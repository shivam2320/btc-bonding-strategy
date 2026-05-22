"""Data fetchers for the BTC Trade Dashboard."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_SESSION_START = time(21, 30)  # 9:30 PM IST — matches backtest/common.py
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backtest"))

COINGECKO = "https://api.coingecko.com/api/v3"
KRAKEN    = "https://api.kraken.com/0/public"
BYBIT     = "https://api.bybit.com/v5"
GAMMA_API = "https://gamma-api.polymarket.com"

_SESS = requests.Session()
_SESS.headers.update({"User-Agent": "Mozilla/5.0"})


class RateLimitedError(Exception):
    pass


def _get(
    url: str,
    params: dict | None = None,
    timeout: int = 8,
    raise_on_429: bool = False,
) -> dict | list | None:
    try:
        r = _SESS.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            if raise_on_429:
                raise RateLimitedError(url)
            return None
        r.raise_for_status()
        return r.json()
    except RateLimitedError:
        raise
    except Exception:
        return None


# ── BTC Price  (CoinGecko simple/price → Coinbase fallback) ──────────────────

def get_btc_market() -> dict:
    """One lightweight CoinGecko call → spot, 24h change%, vol.
    Falls back to Coinbase for spot if CoinGecko is unavailable.
    24h high/low are derived from Kraken OHLC in load_price_data().
    """
    data = _get(f"{COINGECKO}/simple/price", {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
    })
    if data and "bitcoin" in data:
        btc = data["bitcoin"]
        return {
            "spot":       float(btc.get("usd") or 0) or None,
            "change_pct": float(btc.get("usd_24h_change") or 0),
            "volume_usd": float(btc.get("usd_24h_vol") or 0),
        }
    # Coinbase fallback — spot only, no 24h stats
    cb = _get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    if cb and cb.get("data", {}).get("amount"):
        return {"spot": float(cb["data"]["amount"]), "change_pct": 0.0, "volume_usd": 0.0}
    return {"spot": None, "change_pct": 0.0, "volume_usd": 0.0}


# ── OHLC  (Kraken — no geo restrictions, 720 daily candles) ──────────────────

def get_ohlc(interval: str = "1d", limit: int = 250) -> pd.DataFrame | None:
    """Daily OHLC from Kraken.  `interval` param kept for API compat but only
    daily is used; `limit` is how many most-recent candles to keep."""
    data = _get(f"{KRAKEN}/OHLC", {"pair": "XBTUSD", "interval": 1440})
    if not data or data.get("error"):
        return None
    result = data.get("result", {})
    # Kraken result has one key for the pair (e.g. "XXBTZUSD") + "last"
    pair_key = next((k for k in result if k != "last"), None)
    if not pair_key:
        return None
    candles = result[pair_key]
    # Each candle: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.tail(limit).reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1])


def compute_emas(df: pd.DataFrame, periods: tuple = (20, 50, 200)) -> dict[int, float]:
    if df is None or df.empty:
        return {}
    closes = df["close"]
    return {p: float(closes.ewm(span=p, adjust=False).mean().iloc[-1]) for p in periods}


# ── Volatility (from local CSVs) ──────────────────────────────────────────────

def _vol_metrics_for(daily: "pd.DataFrame") -> dict:
    from btc_volatility_analysis import bar_metrics, max_move_series_and_dates  # type: ignore
    m = bar_metrics(daily)
    max_pct, max_usd, ts_pct, ts_usd, _, _ = max_move_series_and_dates(daily)
    return {
        "avg_oc_pct": m["avg_oc_pct"],
        "avg_hl_pct": m["avg_hl_pct"],
        "avg_oc_usd": m["avg_oc_usd"],
        "avg_hl_usd": m["avg_hl_usd"],
        "max_hl_pct": max_pct,
        "max_hl_usd": max_usd,
        "max_hl_pct_date": str(ts_pct.date()) if ts_pct else "—",
        "max_hl_usd_date": str(ts_usd.date()) if ts_usd else "—",
    }


def get_volatility_metrics() -> dict | None:
    try:
        from btc_volatility_analysis import (  # type: ignore
            daily_ohlc_from_hourly,
            load_ohlc,
        )

        path_1m = ROOT / "BTCUSD_1M_1HOUR_FROM_PERPLEXITY (1).csv"
        path_6m = ROOT / "BTCUSD_6M_1DAY_FROM_PERPLEXITY (1).csv"

        m1 = None
        if path_1m.exists():
            daily_1m = daily_ohlc_from_hourly(load_ohlc(path_1m))
            m1 = _vol_metrics_for(daily_1m)

        m6 = None
        if path_6m.exists():
            m6 = _vol_metrics_for(load_ohlc(path_6m))

        if m1 is None and m6 is None:
            return None
        return {"1m": m1, "6m": m6}
    except Exception:
        return None


# ── ForexFactory ──────────────────────────────────────────────────────────────

def get_forex_factory_events(today: date | None = None) -> list[dict]:
    """Return High-impact FF events for the given IST calendar day.

    FF dates are in US Eastern time. We convert each event to IST and compare
    against the current IST date so events aren't missed when UTC is still the
    previous day (e.g. early morning IST).
    """
    if today is None:
        today = datetime.now(IST).date()
    data = _get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10, raise_on_429=True)
    if not data:
        return []
    events = []
    for e in data:
        if e.get("impact") != "High":
            continue
        raw = e.get("date", "")
        try:
            dt_et = datetime.fromisoformat(raw)          # aware, ET offset
            dt_ist = dt_et.astimezone(IST)               # convert to IST
            event_date_ist = dt_ist.date()
        except Exception:
            continue
        if event_date_ist != today:
            continue
        events.append({
            "time_ist": dt_ist.strftime("%I:%M %p IST"),
            "time_utc": dt_ist.astimezone(timezone.utc).strftime("%H:%M UTC"),
            "title": e.get("title", ""),
            "country": e.get("country", ""),
            "forecast": e.get("forecast") or "—",
            "previous": e.get("previous") or "—",
            "actual": e.get("actual") or "",
        })
    return events


# ── Crypto News ───────────────────────────────────────────────────────────────

def get_crypto_news() -> list[dict]:
    data = _get(
        "https://min-api.cryptocompare.com/data/v2/news/",
        {"lang": "EN", "categories": "BTC,Trading", "sortOrder": "latest"},
    )
    if not data or not data.get("Data"):
        return []
    return [
        {
            "title": item.get("title", ""),
            "source": item.get("source_info", {}).get("name", ""),
            "url": item.get("url", ""),
            "published_on": datetime.fromtimestamp(item.get("published_on", 0), tz=timezone.utc),
        }
        for item in data["Data"][:8]
    ]


# ── Futures / Trend  (Bybit → OKX fallback) ──────────────────────────────────

OKX = "https://www.okx.com/api/v5"


def get_funding_rate() -> dict:
    # Bybit
    data = _get(f"{BYBIT}/market/funding/history", {
        "category": "linear", "symbol": "BTCUSDT", "limit": 1,
    })
    try:
        item = data["result"]["list"][0]
        return {
            "rate_pct":    float(item["fundingRate"]) * 100,
            "next_funding": datetime.fromtimestamp(
                int(item["fundingRateTimestamp"]) / 1000, tz=timezone.utc
            ),
        }
    except Exception:
        pass
    # OKX fallback
    data = _get(f"{OKX}/public/funding-rate", {"instId": "BTC-USD-SWAP"})
    try:
        item = data["data"][0]
        return {
            "rate_pct":    float(item["fundingRate"]) * 100,
            "next_funding": datetime.fromtimestamp(
                int(item["nextFundingTime"]) / 1000, tz=timezone.utc
            ),
        }
    except Exception:
        return {}


def get_open_interest() -> float | None:
    # Bybit
    data = _get(f"{BYBIT}/market/open-interest", {
        "category": "linear", "symbol": "BTCUSDT", "intervalTime": "1h", "limit": 1,
    })
    try:
        return float(data["result"]["list"][0]["openInterest"])
    except Exception:
        pass
    # OKX fallback
    data = _get(f"{OKX}/public/open-interest", {"instId": "BTC-USD-SWAP"})
    try:
        return float(data["data"][0]["oi"])
    except Exception:
        return None


def get_etf_performance() -> list[dict]:
    tickers = ["IBIT", "FBTC", "BITB", "ARKB"]
    results = []
    for ticker in tickers:
        data = _get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            {"interval": "1d", "range": "5d"},
            timeout=8,
        )
        try:
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if len(closes) >= 2:
                results.append({
                    "ticker": ticker,
                    "price": closes[-1],
                    "change_pct": (closes[-1] / closes[-2] - 1) * 100,
                })
        except Exception:
            pass
    return results


# ── Polymarket ────────────────────────────────────────────────────────────────

def _parse_strike(question: str) -> float | None:
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", question)
    return float(m.group(1).replace(",", "")) if m else None


def active_session_date() -> tuple[date, datetime, datetime]:
    """Return (market_date, session_start_ist, session_end_ist) for the currently active session.

    Sessions run 21:30 IST → next day 21:30 IST.
    The market slug date equals the *end* day (when the market resolves).
    """
    now_ist = datetime.now(IST)
    today_ist = now_ist.date()
    session_open_today = datetime.combine(today_ist, _SESSION_START, tzinfo=IST)

    if now_ist >= session_open_today:
        # After 9:30 PM IST → new session started, market resolves tomorrow
        session_start = session_open_today
        market_date = today_ist + timedelta(days=1)
    else:
        # Before 9:30 PM IST → still in session that started yesterday evening
        session_start = session_open_today - timedelta(days=1)
        market_date = today_ist

    session_end = session_start + timedelta(days=1)
    return market_date, session_start, session_end


def _find_polymarket_event(start_date: date) -> tuple[dict | None, date | None]:
    """Try start_date ±3 days to find the nearest available BTC daily event."""
    from itertools import chain
    # Try today, then +1, +2, +3 days ahead, then -1, -2, -3 days back
    offsets = list(range(0, 4)) + list(range(-1, -4, -1))
    for delta in offsets:
        d = start_date + timedelta(days=delta)
        slug = f"bitcoin-above-on-{d.strftime('%B').lower()}-{d.day}"
        data = _get(f"{GAMMA_API}/events/slug/{slug}", timeout=12)
        if data and data.get("markets"):
            return data, d
    return None, None


def get_polymarket_btc_markets(target_date: date | None = None) -> tuple[list[dict], date | None]:
    if target_date is None:
        target_date, _, _ = active_session_date()

    data, found_date = _find_polymarket_event(target_date)
    if not data:
        return [], None

    results = []
    for m in data.get("markets", []):
        question = m.get("question", "")
        strike = _parse_strike(question)

        raw_prices = m.get("outcomePrices", "[]")
        try:
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        except Exception:
            prices = []

        yes_price = float(prices[0]) if prices else None
        no_price = float(prices[1]) if len(prices) > 1 else None

        results.append({
            "question": question,
            "slug": m.get("slug", ""),
            "strike": strike,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": float(m.get("volume") or 0),
            "active": m.get("active", True),
        })

    results.sort(key=lambda x: x["strike"] or 0)
    return results, found_date
