"""
Renders a structured report dict into a standalone, self-contained HTML file
and writes it into outputs/.

Each analysis produces its own file — nothing is overwritten.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
CSS_FILE = Path(__file__).resolve().parent / "static" / "styles.css"

TABS = [
    ("overview", "1", "Company Overview"),
    ("quantitative", "2", "Quantitative Analysis"),
    ("qualitative", "3", "Qualitative Analysis"),
    ("shareholding", "4", "Shareholding Pattern"),
    ("thesis", "5", "Investment Thesis"),
    ("valuation", "6", "Valuation &amp; Recommendation"),
    ("conclusion", "7", "Conclusion"),
]

ALLOWED_TAGS = {
    "h3", "h4", "h5", "p", "ul", "ol", "li", "strong", "b", "em", "i", "u", "br",
    "span", "div", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "small", "sup",
}
ALLOWED_ATTRS = {"class", "colspan", "rowspan", "style"}
SAFE_STYLE = re.compile(r"^[a-z0-9\s:;%.,#()\-]*$", re.I)

NEG = re.compile(r"declin|deteriorat|weaken|worsen|contract|erosion|shrink|falling|fell|"
                 r"negative|downside|adverse|pressure|slowdown|deficit", re.I)
POS = re.compile(r"increas|improv|growth|growing|grew|strong|accelerat|expand|robust|healthy|"
                 r"favou?rable|upside|rising|rose|recover|efficient|outperform|positive|record", re.I)
NEU = re.compile(r"stable|steady|consistent|flat|range|fluctuat|mixed|moderat|volatile|"
                 r"unchanged|maintain", re.I)


def sanitize(fragment: str) -> str:
    """Strip anything executable from model-authored HTML."""
    soup = BeautifulSoup(fragment or "", "lxml")
    for node in soup.find_all(True):
        if node.name not in ALLOWED_TAGS:
            node.unwrap()
            continue
        for attr in list(node.attrs):
            if attr not in ALLOWED_ATTRS:
                del node[attr]
            elif attr == "style" and not SAFE_STYLE.match(str(node.get("style", ""))):
                del node["style"]
    # lxml wraps fragments in <html><body>; return the inner markup only
    body = soup.body
    return body.decode_contents() if body else str(soup)


def colour_trends(fragment: str) -> str:
    """Wrap the trend column of each table in a colour-coded chip."""
    soup = BeautifulSoup(fragment or "", "lxml")
    for table in soup.find_all("table"):
        head = table.select_one("thead tr")
        if not head:
            continue
        heads = head.find_all(["th", "td"])
        if len(heads) < 2:
            continue
        last = len(heads) - 1
        label = heads[last].get_text(strip=True).lower()
        if "trend" not in label and "commentary" not in label:
            continue
        for tr in table.select("tbody tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) <= last:
                continue
            cell = cells[last]
            if cell.select_one(".t-up,.t-down,.t-flat"):
                continue
            text = cell.get_text(" ", strip=True)
            if not text:
                continue
            cls = ("t-down" if NEG.search(text)
                   else "t-up" if POS.search(text)
                   else "t-flat" if NEU.search(text) else None)
            if not cls:
                continue
            cell.clear()
            span = soup.new_tag("span")
            span["class"] = cls
            span.string = text
            cell.append(span)
    body = soup.body
    return body.decode_contents() if body else str(soup)


def indian_group(value: float) -> str:
    """1055780 -> '10,55,780' (Indian digit grouping)."""
    neg = value < 0
    whole = str(int(abs(round(value))))
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts) + "," + tail
    return ("-" if neg else "") + whole


def build_charts(company: dict) -> str:
    """
    Render bar charts directly from the scraped figures.

    These are generated in Python rather than asked of the model: the numbers
    are already known exactly, so this guarantees both accuracy and that the
    charts actually appear.
    """
    per_year = (company.get("derived") or {}).get("per_year") or []
    years = [y for y in per_year if str(y.get("period", "")).startswith("FY")][-5:]
    if len(years) < 2:
        return ""

    specs = [
        ("Revenue (Rs Cr)", "sales", lambda v: indian_group(v)),
        ("Net Profit (Rs Cr)", "net_profit", lambda v: indian_group(v)),
        ("Operating Margin (%)", "operating_margin_pct", lambda v: f"{v:.1f}%"),
        ("Cash Flow from Operations (Rs Cr)", "cfo", lambda v: indian_group(v)),
    ]

    charts = []
    for title, key, fmt in specs:
        points = [(y["period"], y.get(key)) for y in years if y.get(key) is not None]
        if len(points) < 2:
            continue
        peak = max(abs(v) for _, v in points) or 1
        rows = []
        for period, value in points:
            width = max(2, round(abs(value) / peak * 100))
            rows.append(
                f'<div class="bar-row"><span class="bar-lbl">{html.escape(period)}</span>'
                f'<span class="bar-track"><span class="bar-fill" style="width:{width}%"></span></span>'
                f'<span class="bar-val">{html.escape(fmt(value))}</span></div>'
            )
        charts.append(
            f'<div class="chart"><h5>{html.escape(title)}</h5>'
            f'<div class="bars">{"".join(rows)}</div></div>'
        )

    if not charts:
        return ""
    return (
        '<h3>Five-Year Visual Summary</h3>'
        '<p><em>Charted directly from the figures scraped from Screener.in.</em></p>'
        f'<div class="chart-grid">{"".join(charts)}</div>'
    )


def _rating_class(rating: str) -> str:
    r = (rating or "").lower()
    if "buy" in r:
        return "rating-buy"
    if "sell" in r:
        return "rating-sell"
    return "rating-hold"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "report").lower()).strip("-")
    return s or "report"


def _as_dict(value) -> dict:
    """Tolerate a nested object arriving as a JSON string (see analyst._as_dict)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def render_html(report: dict, company: dict, live: dict) -> str:
    """Build the complete standalone report document."""
    css = CSS_FILE.read_text(encoding="utf-8")
    rec = _as_dict(report.get("recommendation"))
    name = report.get("companyName") or company.get("name") or "Company"
    sector = report.get("sector") or ""
    ticker = report.get("ticker") or company.get("ticker") or ""
    exchange = report.get("exchange") or "NSE"

    e = html.escape
    title = f"{name} — Equity Analysis Report"

    sub_bits = []
    if ticker:
        sub_bits.append(f"<b>{e(ticker)}</b>")
    if exchange:
        sub_bits.append(e(exchange))
    if sector:
        sub_bits.append(e(sector))

    sources = [f'<a href="{e(company.get("screener_url",""))}" target="_blank" rel="noopener">Screener.in</a>']
    if live.get("symbol"):
        sources.append(
            f'<a href="https://finance.yahoo.com/quote/{e(live["symbol"])}" '
            f'target="_blank" rel="noopener">Yahoo Finance ({e(live["symbol"])})</a>'
        )

    tabs_html, panels_html = [], []
    for i, (key, num, label) in enumerate(TABS):
        selected = "true" if i == 0 else "false"
        tabs_html.append(
            f'<button type="button" id="tab-{key}" role="tab" aria-selected="{selected}" '
            f'aria-controls="panel-{key}" onclick="pick(\'{key}\')">'
            f'<span class="num">{num}</span>{label}</button>'
        )
        raw = _as_dict(report.get("sections")).get(key, "")
        body = colour_trends(sanitize(raw if isinstance(raw, str) else json.dumps(raw)))
        if not body.strip():
            body = "<p><em>No content was returned for this section.</em></p>"
        if key == "quantitative":
            body += build_charts(company)
        hidden = "" if i == 0 else " hidden"
        panels_html.append(
            f'<section class="panel" id="panel-{key}" role="tabpanel" '
            f'aria-labelledby="tab-{key}"{hidden}>{body}</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
{css}
</style>
</head>
<body>
<header class="masthead">
  <div class="masthead-inner">
    <div class="brand">
      <div class="brand-mark">EA</div>
      <div class="brand-text">
        <h1>Equity Analysis Report</h1>
        <p>Fundamental Research &middot; Indian Markets</p>
      </div>
    </div>
    <div class="masthead-meta">
      <strong>By SEETHARAMAN</strong>
      Data: Screener.in &middot; Yahoo Finance
    </div>
  </div>
</header>

<main>
<section class="card report">
  <div class="report-head">
    <p class="eyebrow">{e(sector) or "Equity Research"}</p>
    <h2>{e(title)}</h2>
    <p class="sub">{" &nbsp;·&nbsp; ".join(sub_bits)}</p>
    <p class="byline">By SEETHARAMAN</p>
    <p class="sources">Sources: {" &middot; ".join(sources)}</p>
  </div>

  <div class="rec-box">
    <div class="rec-cell">
      <p class="k">Recommendation</p>
      <p class="v {_rating_class(rec.get('rating',''))}">{e(str(rec.get('rating','—'))).upper()}</p>
    </div>
    <div class="rec-cell">
      <p class="k">Target Price</p>
      <p class="v">{e(str(rec.get('targetPrice','—')))}</p>
      <p class="note">{e(str(rec.get('changePct','')))}</p>
    </div>
    <div class="rec-cell">
      <p class="k">Current Price</p>
      <p class="v">{e(str(rec.get('currentPrice','—')))}</p>
    </div>
    <div class="rec-cell">
      <p class="k">Investment Horizon</p>
      <p class="v small">{e(str(rec.get('horizon','—')))}</p>
    </div>
  </div>

  <div class="report-actions">
    <button type="button" class="btn btn-ghost btn-sm" onclick="window.print()">Print / Save as PDF</button>
  </div>

  <nav class="tabbar" role="tablist">{"".join(tabs_html)}</nav>
  <div class="panels">{"".join(panels_html)}</div>
</section>
</main>

<footer class="site-footer">
  <div class="disclaimer">
    <strong>Disclaimer</strong>
    This report was generated by an AI language model from data scraped from Screener.in and
    Yahoo Finance, for <b>educational and informational purposes only</b>. It does not constitute
    investment advice, a recommendation, or an offer to buy or sell any security. Figures are
    reproduced automatically and <b>may be inaccurate, incomplete, or out of date</b> — verify all
    data against company filings and exchange disclosures before making any investment decision.
    Valuation estimates, target prices and ratings are model-generated opinions, not professional
    advice. Past performance is not indicative of future results. Investing in equities involves
    risk, including loss of principal. Consult a SEBI-registered investment adviser before acting
    on any information contained here.
  </div>
</footer>

<script>
function pick(id){{
  var ids={[t[0] for t in TABS]!r};
  ids.forEach(function(i){{
    var b=document.getElementById("tab-"+i), p=document.getElementById("panel-"+i);
    if(b) b.setAttribute("aria-selected", String(i===id));
    if(p) p.hidden = (i!==id);
  }});
}}
</script>
</body>
</html>
"""


def save(report: dict, company: dict, live: dict) -> dict:
    """Write the report to outputs/ and return its filename and path."""
    sections = _as_dict(report.get("sections"))
    if not any(str(v).strip() for v in sections.values()):
        # Never leave a hollow file on disk that looks like a real report.
        raise ValueError(
            "Refusing to save: the report has no section content "
            "(the model response was most likely truncated)."
        )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    name = report.get("companyName") or company.get("name") or "report"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{slugify(name)}-equity-analysis-{stamp}.html"
    path = OUTPUTS / filename
    path.write_text(render_html(report, company, live), encoding="utf-8")

    # The network share this project lives on stamps newly written files with a
    # fixed, wrong base time (~Dec 2002), so Explorer shows the wrong date. The
    # share honours an explicit modification-time update, so restamp it to now.
    # (The creation time is not settable on this share — it silently ignores the
    # change — but the app reads a report's date from its filename regardless.)
    try:
        now = time.time()
        os.utime(path, (now, now))
    except OSError:
        pass

    return {
        "filename": filename,
        "path": str(path),
        "url": f"/reports/{filename}",
        "bytes": path.stat().st_size,
    }
