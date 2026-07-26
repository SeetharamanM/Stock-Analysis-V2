"""
Scrapes fundamental data for an Indian listed company from Screener.in.

Screener has no public API, so this reads the public company page and the
peers fragment. Requests are deliberately polite: a real User-Agent, a short
cache, and a single page fetch per analysis.
"""

from __future__ import annotations

import re
import time
import threading
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE = "https://www.screener.in"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 15 * 60  # seconds


class ScreenerError(RuntimeError):
    """Raised when Screener.in cannot be reached or the company is not found."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _cached(key: str):
    with _cache_lock:
        hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < CACHE_TTL:
        return hit[1]
    return None


def _store(key: str, value):
    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value


def _get(url: str, headers: dict | None = None, **kw) -> requests.Response:
    try:
        r = requests.get(url, headers={**UA, **(headers or {})}, timeout=25, **kw)
    except requests.RequestException as exc:
        raise ScreenerError(f"Could not reach Screener.in: {exc}") from exc
    return r


def clean_num(text: str):
    """'₹ 1,17,279 Cr.' -> 117279.0 ; '18%' -> 18.0 ; '' -> None"""
    if text is None:
        return None
    t = str(text).replace(",", "").replace("%", "")
    t = t.replace("₹", "").replace("Cr.", "").replace("Cr", "").strip()
    if t in ("", "-", "--"):
        return None
    neg = t.startswith("-")
    t = re.sub(r"[^0-9.]", "", t)
    if t in ("", "."):
        return None
    try:
        val = float(t)
    except ValueError:
        return None
    return -val if neg else val


def fy_label(col: str) -> str:
    """'Mar 2024' -> 'FY24'. Indian FY ends 31 March."""
    m = re.search(r"(\d{4})", col or "")
    if not m:
        return (col or "").strip()
    return "FY" + m.group(1)[2:]


def _row_label(tr) -> str:
    cell = tr.find(["td", "th"])
    if not cell:
        return ""
    label = cell.get_text(" ", strip=True)
    return label.rstrip("+").strip()


def _parse_table(table) -> dict:
    """Turn a Screener data-table into {'columns': [...], 'rows': {label: [vals]}}"""
    if table is None:
        return {"columns": [], "rows": {}}
    heads = [th.get_text(strip=True) for th in table.select("thead th")]
    columns = heads[1:] if heads else []
    rows: dict[str, list] = {}
    for tr in table.select("tbody tr"):
        label = _row_label(tr)
        if not label:
            continue
        cells = tr.select("td")[1:]
        rows[label] = [c.get_text(strip=True) for c in cells]
    return {"columns": columns, "rows": rows}


def _ranges(section) -> dict:
    """Parse the 'Compounded Sales Growth' style CAGR blocks."""
    out: dict[str, dict[str, str]] = {}
    if section is None:
        return out
    for tbl in section.select(".ranges-table"):
        trs = tbl.select("tr")
        if not trs:
            continue
        title = trs[0].get_text(" ", strip=True)
        vals: dict[str, str] = {}
        for tr in trs[1:]:
            cells = [c.get_text(strip=True) for c in tr.select("td,th")]
            if len(cells) >= 2:
                vals[cells[0].rstrip(":")] = cells[1]
        if title:
            out[title] = vals
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def search(query: str) -> list[dict]:
    """Resolve a company name / ticker to Screener entries."""
    key = f"search::{query.lower()}"
    if (hit := _cached(key)) is not None:
        return hit
    r = _get(f"{BASE}/api/company/search/", params={"q": query})
    if r.status_code != 200:
        raise ScreenerError(f"Screener search failed (HTTP {r.status_code}).")
    try:
        data = r.json()
    except ValueError as exc:
        raise ScreenerError("Screener search returned an unexpected response.") from exc
    out = []
    for d in data:
        url = d.get("url", "")
        if not url.startswith("/company/"):
            continue
        parts = [p for p in url.strip("/").split("/") if p]
        # /company/INFY/consolidated/ -> INFY ; /company/id/62890/ -> no ticker
        ticker = parts[1] if len(parts) > 1 and parts[1] != "id" else ""
        out.append({"name": d.get("name", ""), "url": url, "id": d.get("id"), "ticker": ticker})
    return _store(key, out)


def _peers(company_id, referer: str) -> dict:
    """
    Best-effort peer comparison. Not every company has a peer group configured
    on Screener (it 404s), which is fine: the report then falls back to a
    qualitative peer comparison.
    """
    empty = {"columns": [], "rows": [], "median": [], "available": False}
    if not company_id:
        return empty
    try:
        r = _get(
            f"{BASE}/api/company/{company_id}/peers/",
            headers={"Referer": referer, "X-Requested-With": "XMLHttpRequest"},
        )
    except ScreenerError:
        return empty
    if r.status_code != 200:
        return empty

    soup = BeautifulSoup(r.text, "lxml")
    # Headers sit in <th> cells inside tbody — this fragment has no <thead>.
    cols = [th.get_text(" ", strip=True) for th in soup.select("th")]
    rows = []
    for tr in soup.select("tr"):
        if tr.find("th"):  # header row
            continue
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if cells and any(c.strip() for c in cells):
            rows.append(cells)
    # The industry median lives in <tfoot>.
    median = [c.get_text(" ", strip=True) for c in soup.select("tfoot td, tfoot th")]
    if median and rows and median in rows:
        rows.remove(median)
    return {
        "columns": cols,
        "rows": rows[:12],
        "median": median,
        "available": bool(rows),
    }


def fetch_company(query: str) -> dict:
    """
    Resolve `query` and scrape the full fundamental dataset.
    Raises ScreenerError if the company cannot be found.
    """
    matches = search(query)
    if not matches:
        raise ScreenerError(
            f"No company on Screener.in matches '{query}'. "
            "Try the exact listed name or the NSE ticker."
        )
    best = matches[0]
    path = best["url"]
    if not path.endswith("/consolidated/"):
        path = path.rstrip("/") + "/consolidated/"

    key = f"company::{path}"
    if (hit := _cached(key)) is not None:
        return hit

    url = BASE + path
    r = _get(url)
    if r.status_code == 404:  # some companies have no consolidated statements
        url = BASE + best["url"]
        r = _get(url)
    if r.status_code != 200:
        raise ScreenerError(f"Screener.in returned HTTP {r.status_code} for {best['name']}.")

    soup = BeautifulSoup(r.text, "lxml")

    # --- headline ratios -------------------------------------------------
    top: dict[str, str] = {}
    for li in soup.select("#top-ratios li"):
        name = li.select_one(".name")
        value = li.select_one(".value")
        if name and value:
            top[name.get_text(strip=True)] = " ".join(value.get_text().split())

    # --- narrative -------------------------------------------------------
    about_el = soup.select_one(".company-profile .about, .about")
    about = " ".join(about_el.get_text().split()) if about_el else ""

    key_points = ""
    profile = soup.select_one(".company-profile")
    if profile:
        text = " ".join(profile.get_text(" ").split())
        m = re.search(r"Key Points(.*)", text)
        if m:
            key_points = m.group(1).strip()[:2500]

    links = [a.get_text(strip=True) for a in soup.select(".company-links a")]

    # --- statement tables ------------------------------------------------
    sections = {
        "profit_loss": "#profit-loss",
        "balance_sheet": "#balance-sheet",
        "cash_flow": "#cash-flow",
        "ratios": "#ratios",
        "shareholding": "#shareholding",
        "quarters": "#quarters",
    }
    tables = {}
    for name, sel in sections.items():
        node = soup.select_one(sel)
        tables[name] = _parse_table(node.select_one("table") if node else None)

    growth = _ranges(soup.select_one("#profit-loss"))

    cid_match = re.search(r'data-company-id="(\d+)"', r.text)
    company_id = cid_match.group(1) if cid_match else best.get("id")

    data = {
        "name": best["name"],
        "ticker": best.get("ticker", ""),
        "screener_url": url,
        "company_id": company_id,
        "about": about,
        "key_points": key_points,
        "links": links,
        "top_ratios": top,
        "growth": growth,
        "peers": _peers(company_id, url),
        **tables,
    }
    data["derived"] = derive_metrics(data)
    return _store(key, data)


# --------------------------------------------------------------------------
# derived metrics — computed in Python so the model never does the arithmetic
# --------------------------------------------------------------------------

def _series(table: dict, *candidates: str) -> tuple[list[str], list]:
    """Find the first matching row across candidate labels."""
    rows = table.get("rows", {})
    for cand in candidates:
        for label, values in rows.items():
            if label.lower() == cand.lower():
                return table.get("columns", []), [clean_num(v) for v in values]
    for cand in candidates:  # fall back to substring match
        for label, values in rows.items():
            if cand.lower() in label.lower():
                return table.get("columns", []), [clean_num(v) for v in values]
    return table.get("columns", []), []


def _cagr(series: list, years: int):
    """CAGR over the last `years` periods of a numeric series."""
    vals = [v for v in series if v is not None]
    if len(vals) < years + 1:
        return None
    start, end = vals[-(years + 1)], vals[-1]
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / years) - 1) * 100, 2)


def derive_metrics(data: dict) -> dict:
    """
    Compute the ratios the report needs but Screener does not print directly:
    net profit margin, debt-to-equity, asset turnover, and CAGRs.
    """
    pl, bs, cf = data.get("profit_loss", {}), data.get("balance_sheet", {}), data.get("cash_flow", {})

    cols, sales = _series(pl, "Sales", "Revenue")
    _, net_profit = _series(pl, "Net Profit")
    _, op_profit = _series(pl, "Operating Profit")
    _, eps = _series(pl, "EPS in Rs", "EPS")
    _, payout = _series(pl, "Dividend Payout %")

    _, borrowings = _series(bs, "Borrowings")
    _, equity_cap = _series(bs, "Equity Capital")
    _, reserves = _series(bs, "Reserves")
    _, total_assets = _series(bs, "Total Assets")

    _, cfo = _series(cf, "Cash from Operating Activity")

    labels = [fy_label(c) for c in cols]
    n = len(labels)

    def at(seq, i):
        return seq[i] if seq and i < len(seq) else None

    per_year = []
    for i in range(n):
        s, np_, op = at(sales, i), at(net_profit, i), at(op_profit, i)
        borrow = at(borrowings, i)
        eq = at(equity_cap, i)
        res = at(reserves, i)
        ta = at(total_assets, i)
        net_worth = (eq or 0) + (res or 0) if (eq is not None or res is not None) else None

        per_year.append({
            "period": labels[i],
            "sales": s,
            "net_profit": np_,
            "operating_profit": op,
            "eps": at(eps, i),
            "dividend_payout_pct": at(payout, i),
            "cfo": at(cfo, i),
            "net_profit_margin_pct": round(np_ / s * 100, 2) if s and np_ is not None and s != 0 else None,
            "operating_margin_pct": round(op / s * 100, 2) if s and op is not None and s != 0 else None,
            "debt_to_equity": round(borrow / net_worth, 2) if borrow is not None and net_worth else None,
            "asset_turnover": round(s / ta, 2) if s and ta else None,
            "net_worth": round(net_worth, 2) if net_worth else None,
            "total_assets": ta,
            "borrowings": borrow,
        })

    return {
        "per_year": per_year,
        "cagr": {
            "revenue_5y_pct": _cagr(sales, 5),
            "revenue_3y_pct": _cagr(sales, 3),
            "net_profit_5y_pct": _cagr(net_profit, 5),
            "net_profit_3y_pct": _cagr(net_profit, 3),
            "eps_5y_pct": _cagr(eps, 5),
            "eps_3y_pct": _cagr(eps, 3),
        },
        "latest_period": labels[-1] if labels else None,
    }
