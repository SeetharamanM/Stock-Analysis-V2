"""
Live market quote from Yahoo Finance, used to cross-reference Screener's
price with the most recent close.

Yahoo's chart endpoint is unofficial but does not require a key. If it is
unavailable the report still works — it simply falls back to Screener's price.
"""

from __future__ import annotations

import requests

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def quote(ticker: str) -> dict:
    """
    Fetch the latest quote for an NSE ticker. Tries .NS then .BO.
    Returns {} if neither resolves, so callers can degrade gracefully.
    """
    if not ticker:
        return {}
    for suffix in (".NS", ".BO"):
        symbol = f"{ticker}{suffix}"
        try:
            r = requests.get(
                CHART.format(symbol=symbol),
                params={"range": "1y", "interval": "1d"},
                headers=UA,
                timeout=15,
            )
            if r.status_code != 200:
                continue
            result = (r.json().get("chart") or {}).get("result") or []
            if not result:
                continue
            meta = result[0].get("meta") or {}
            price = meta.get("regularMarketPrice")
            if price is None:
                continue

            closes = []
            try:
                raw = result[0]["indicators"]["quote"][0]["close"]
                closes = [c for c in raw if c is not None]
            except (KeyError, IndexError, TypeError):
                pass

            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            change_pct = None
            if prev:
                change_pct = round((price - prev) / prev * 100, 2)

            return {
                "symbol": symbol,
                "source": "Yahoo Finance",
                "long_name": meta.get("longName") or meta.get("shortName"),
                "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
                "currency": meta.get("currency"),
                "price": round(price, 2),
                "previous_close": round(prev, 2) if prev else None,
                "day_change_pct": change_pct,
                "fifty_two_week_high": meta.get("fiftyTwoWeekHigh") or (max(closes) if closes else None),
                "fifty_two_week_low": meta.get("fiftyTwoWeekLow") or (min(closes) if closes else None),
                "one_year_change_pct": (
                    round((closes[-1] - closes[0]) / closes[0] * 100, 2)
                    if len(closes) > 1 and closes[0]
                    else None
                ),
            }
        except (requests.RequestException, ValueError):
            continue
    return {}
