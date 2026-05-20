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

BINANCE_REST = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
GAMMA_API = "https://gamma-api.polymarket.com"

_SESS = requests.Session()
_SESS.headers.update({"User-Agent": "Mozilla/5.0"})


class RateLimitedError(Exception):
    pass


def _get(url: str, params: dict | None = None, timeout: int = 8) -> dict | list | None:
    try:
        r = _SESS.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            raise RateLimitedError(url)
        r.raise_for_status()
        return r.json()
    except RateLimitedError:
        raise
    except Exception:
        return None


# ── BTC Price ─────────────────────────────────────────────────────────────────

def get_spot_price() -> float | None:
    data = _get(f"{BINANCE_REST}/api/v3/ticker/price", {"symbol": "BTCUSDT"})
    return float(data["price"]) if data else None


def get_24h_stats() -> dict:
    data = _get(f"{BINANCE_REST}/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})
    if not data:
        return {}
    return {
        "change_pct": float(data["priceChangePercent"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
        "volume_usd": float(data["quoteVolume"]),
    }


def get_ohlc(interval: str = "1d", limit: int = 50) -> pd.DataFrame | None:
    data = _get(
        f"{BINANCE_REST}/api/v3/klines",
        {"symbol": "BTCUSDT", "interval": interval, "limit": limit},
    )
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades", "tbb", "tbq", "ignore",
    ])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
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

def get_volatility_metrics() -> dict | None:
    try:
        from btc_volatility_analysis import (  # type: ignore
            bar_metrics,
            daily_log_returns_from_close,
            daily_ohlc_from_hourly,
            ewma_volatility_series_pct,
            historical_volatility_from_returns,
            load_ohlc,
        )

        path = ROOT / "BTCUSD_1M_1HOUR_FROM_PERPLEXITY (1).csv"
        if not path.exists():
            return None
        raw = load_ohlc(path)
        daily = daily_ohlc_from_hourly(raw)
        r = daily_log_returns_from_close(daily)
        sigma_hist, _mu, var_dec = historical_volatility_from_returns(r)
        ewma_series = ewma_volatility_series_pct(r, lam=0.94, initial_var=float(var_dec))
        m = bar_metrics(daily)
        return {
            "hist_sigma": sigma_hist,
            "ewma_sigma": float(ewma_series.iloc[-1]),
            "avg_hl_pct": m["avg_hl_pct"],
            "avg_hl_usd": m["avg_hl_usd"],
            "as_of": str(ewma_series.index[-1].date()),
        }
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
    data = _get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)  # raises RateLimitedError on 429
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


# ── Futures / Trend ───────────────────────────────────────────────────────────

def get_funding_rate() -> dict:
    data = _get(f"{BINANCE_FUTURES}/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1})
    if data and isinstance(data, list) and data:
        item = data[-1]
        return {
            "rate_pct": float(item["fundingRate"]) * 100,
            "next_funding": datetime.fromtimestamp(item["fundingTime"] / 1000, tz=timezone.utc),
        }
    return {}


def get_open_interest() -> float | None:
    data = _get(f"{BINANCE_FUTURES}/fapi/v1/openInterest", {"symbol": "BTCUSDT"})
    return float(data["openInterest"]) if data else None


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


def get_polymarket_btc_markets(target_date: date | None = None) -> list[dict]:
    if target_date is None:
        target_date, _, _ = active_session_date()

    month_name = target_date.strftime("%B").lower()
    day = target_date.day
    slug = f"bitcoin-above-on-{month_name}-{day}"

    data = _get(f"{GAMMA_API}/events/slug/{slug}", timeout=12)
    if not data:
        return []

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
    return results
