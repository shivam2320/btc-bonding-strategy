"""Data fetchers for the BTC Trade Dashboard."""

from __future__ import annotations

import json
import math
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

BINANCE      = "https://api.binance.com"
COINGECKO    = "https://api.coingecko.com/api/v3"
KRAKEN       = "https://api.kraken.com/0/public"
BYBIT        = "https://api.bybit.com/v5"
GAMMA_API    = "https://gamma-api.polymarket.com"
NEWSAPI_KEY  = "83994ea8-14b0-404e-9f0f-282a2d0b7c9d"

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


# ── BTC Price  (Binance → CoinGecko → Coinbase fallback) ─────────────────────

def get_btc_market() -> dict:
    """Binance 24h ticker (primary) → CoinGecko → Coinbase fallback.
    Returns spot, 24h change%, vol, high_24h, low_24h.
    """
    # Binance global — single call gives spot + 24h stats + high/low
    data = _get(f"{BINANCE}/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})
    if data and isinstance(data, dict) and data.get("lastPrice"):
        return {
            "spot":       float(data["lastPrice"]),
            "change_pct": float(data.get("priceChangePercent") or 0),
            "volume_usd": float(data.get("quoteVolume") or 0),
            "high_24h":   float(data["highPrice"]) if data.get("highPrice") else None,
            "low_24h":    float(data["lowPrice"])  if data.get("lowPrice")  else None,
        }
    # CoinGecko fallback — no high/low
    data = _get(f"{COINGECKO}/simple/price", {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
    })
    if data and isinstance(data, dict) and "bitcoin" in data:
        btc = data["bitcoin"]
        return {
            "spot":       float(btc.get("usd") or 0) or None,
            "change_pct": float(btc.get("usd_24h_change") or 0),
            "volume_usd": float(btc.get("usd_24h_vol") or 0),
            "high_24h":   None,
            "low_24h":    None,
        }
    # Coinbase fallback — spot only
    cb = _get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
    if cb and isinstance(cb, dict) and cb.get("data", {}).get("amount"):
        return {"spot": float(cb["data"]["amount"]), "change_pct": 0.0, "volume_usd": 0.0, "high_24h": None, "low_24h": None}
    return {"spot": None, "change_pct": 0.0, "volume_usd": 0.0, "high_24h": None, "low_24h": None}


# ── OHLC  (Kraken — no geo restrictions, 720 daily candles) ──────────────────

def get_ohlc(interval: str = "1d", limit: int = 720) -> pd.DataFrame | None:
    """Daily OHLC from Kraken.  `interval` param kept for API compat but only
    daily is used; `limit` is how many most-recent candles to keep (720 ≈ 2 years)."""
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


def get_session_open_price(session_start_ist: "datetime") -> float | None:
    """BTC open price at session start (9:30 PM IST = 16:00 UTC) from Kraken 1h OHLC."""
    utc = session_start_ist.astimezone(timezone.utc)
    ts  = int(utc.timestamp())
    data = _get(f"{KRAKEN}/OHLC", {"pair": "XBTUSD", "interval": 60, "since": ts - 60}, timeout=10)
    try:
        result   = data["result"]
        pair_key = next(k for k in result if k != "last")
        candles  = result[pair_key]
        # First candle at or immediately after session start
        for c in candles:
            if int(c[0]) >= ts - 120:          # within 2 min of session open
                return float(c[1])             # open price
    except Exception:
        return None


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


def get_volatility_metrics(ohlc: "pd.DataFrame | None" = None) -> dict | None:
    """Compute volatility metrics.

    Primary: slice `ohlc` (Kraken live data — always current).
    Fallback: load from local CSVs (stale but better than nothing).
    """
    if ohlc is not None and len(ohlc) >= 30:
        try:
            m1 = _vol_metrics_for(ohlc.tail(30))
            m6 = _vol_metrics_for(ohlc.tail(180)) if len(ohlc) >= 180 else None
            return {"1m": m1, "6m": m6}
        except Exception:
            pass

    # CSV fallback (local dev / Kraken outage)
    try:
        from btc_volatility_analysis import (  # type: ignore
            daily_ohlc_from_hourly,
            load_ohlc,
        )
        path_1m = ROOT / "BTCUSD_1M_1HOUR_FROM_PERPLEXITY (1).csv"
        path_6m = ROOT / "BTCUSD_6M_1DAY_FROM_PERPLEXITY (1).csv"
        m1 = _vol_metrics_for(daily_ohlc_from_hourly(load_ohlc(path_1m))) if path_1m.exists() else None
        m6 = _vol_metrics_for(load_ohlc(path_6m)) if path_6m.exists() else None
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


# ── Crypto News (newsapi.ai) ──────────────────────────────────────────────────

def get_crypto_news() -> list[dict]:
    data = _get(
        "https://newsapi.ai/api/v1/article/getArticles",
        {
            "keyword":           "Bitcoin",
            "keywordLoc":        "title,body",
            "lang":              "eng",
            "sortBy":            "date",
            "sortByAsc":         "false",
            "articlesPage":      1,
            "articlesCount":     8,
            "articlesSortBy":    "date",
            "articlesSortByAsc": "false",
            "resultType":        "articles",
            "apiKey":            NEWSAPI_KEY,
        },
        timeout=12,
    )
    if not data or "articles" not in data:
        return []
    results = data.get("articles", {}).get("results", [])
    items = []
    for item in results[:8]:
        try:
            dt_str = item.get("dateTimePub") or item.get("dateTime", "")
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)
        items.append({
            "title":        item.get("title", ""),
            "source":       (item.get("source") or {}).get("title", ""),
            "url":          item.get("url", ""),
            "published_on": dt,
        })
    return items


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


def get_etf_flows() -> dict | None:
    """BTC ETF aggregate net flows from CoinMarketCap (no API key required)."""
    data = _get(
        "https://api.coinmarketcap.com/data-api/v3/etf/netflow/metrics",
        {"category": "btc"},
        timeout=10,
    )
    try:
        d = data["data"]
        return {
            "today":        d["current"]["value"]       / 1e6,
            "last_week":    d["lastWeek"]["value"]      / 1e6,
            "last_month":   d["lastMonth"]["value"]     / 1e6,
            "three_months": d["threeMonthsAgo"]["value"] / 1e6,
        }
    except Exception:
        return None


# ── Fear & Greed Index ────────────────────────────────────────────────────────

def get_fear_greed() -> dict | None:
    data = _get("https://api.alternative.me/fng/", {"limit": 1})
    try:
        item = data["data"][0]
        return {
            "value":          int(item["value"]),
            "classification": item["value_classification"],
        }
    except Exception:
        return None


# ── Realized Volatility (7d / 30d annualised) ─────────────────────────────────

def get_realized_vol(ohlc: "pd.DataFrame | None") -> dict | None:
    if ohlc is None or len(ohlc) < 31:
        return None
    try:
        closes  = ohlc["close"].astype(float)
        log_ret = np.log(closes / closes.shift(1)).dropna()
        rv7  = float(log_ret.tail(7).std()  * np.sqrt(252) * 100)
        rv30 = float(log_ret.tail(30).std() * np.sqrt(252) * 100)
        return {"rv7": rv7, "rv30": rv30}
    except Exception:
        return None


# ── Deribit DVOL (BTC 30-day constant-maturity implied vol index) ─────────────

def get_deribit_dvol() -> float | None:
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - 2 * 3_600_000          # 2-hour window → 1 candle at 1h resolution
    data = _get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        {"currency": "BTC", "start_timestamp": start_ms,
         "end_timestamp": now_ms, "resolution": 3600},
        timeout=10,
    )
    try:
        candles = data["result"]["data"]        # [[ts, open, high, low, close], ...]
        return float(candles[-1][4])            # close of last candle
    except Exception:
        return None


# ── BTC GEX (Deribit options) ─────────────────────────────────────────────────

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def _bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes gamma (same formula for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1 ** 2) / (S * sigma * math.sqrt(2 * math.pi * T))
    except Exception:
        return 0.0


def _expiry_years(date_code: str) -> float:
    """Parse Deribit date code like '23MAY26' → years to expiry from today."""
    try:
        day   = int(date_code[:2])
        month = _MONTHS[date_code[2:5].upper()]
        year  = 2000 + int(date_code[5:7])
        T = (date(year, month, day) - date.today()).days / 365.0
        return max(T, 0.5 / 365)   # floor at ~12 h so gamma stays finite
    except Exception:
        return 0.0


def get_btc_gex() -> dict | None:
    """Net Gamma Exposure from all active BTC options on Deribit.

    Deribit book summary doesn't expose gamma directly, so we compute it via
    Black-Scholes using mark_iv (available in the response) and the strike /
    expiry parsed from the instrument name.

    GEX = Σ(call_gamma × call_OI − put_gamma × put_OI) × spot²
    Positive GEX → dealers long gamma → stabilising (mean-reversion).
    Negative GEX → dealers short gamma → destabilising (trending/volatile).
    """
    try:
        return _get_btc_gex_inner()
    except Exception:
        return None


def _get_btc_gex_inner() -> dict | None:
    data = _get(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        {"currency": "BTC", "kind": "option"},
        timeout=20,
    )
    if not data or not isinstance(data, dict) or "result" not in data:
        return None
    instruments = data.get("result", [])
    if not isinstance(instruments, list) or not instruments:
        return None

    # Underlying spot from the first instrument that carries it
    spot = None
    for inst in instruments:
        if not isinstance(inst, dict):
            continue
        up = inst.get("underlying_price")
        if up:
            spot = float(up)
            break
    if not spot:
        return None

    call_gex = 0.0
    put_gex  = 0.0
    gex_by_strike: dict[float, float] = {}

    for inst in instruments:
        if not isinstance(inst, dict):
            continue
        name  = inst.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) < 4:
            continue                      # skip if not a valid option name
        opt_type   = parts[-1]            # "C" or "P"
        date_code  = parts[1]             # e.g. "23MAY26"
        try:
            strike = float(parts[-2])
        except ValueError:
            continue

        mark_iv = float(inst.get("mark_iv") or 0) / 100   # % → decimal
        oi      = float(inst.get("open_interest") or 0)
        if mark_iv <= 0 or oi <= 0:
            continue

        T     = _expiry_years(date_code)
        gamma = _bs_gamma(spot, strike, T, mark_iv)
        gex   = gamma * oi * spot * spot

        if opt_type == "C":
            call_gex += gex
            gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) + gex
        else:
            put_gex  += gex
            gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) - gex

    net_gex = call_gex - put_gex

    # Gamma flip — strike nearest to spot where per-strike net GEX changes sign
    near = sorted(s for s in gex_by_strike if 0.8 * spot <= s <= 1.2 * spot)
    flip_level = None
    for i in range(1, len(near)):
        g1, g2 = gex_by_strike[near[i - 1]], gex_by_strike[near[i]]
        if (g1 < 0 <= g2) or (g1 >= 0 > g2):
            denom = abs(g1) + abs(g2)
            w = abs(g1) / denom if denom > 0 else 0.5
            flip_level = near[i - 1] + (near[i] - near[i - 1]) * w
            break

    # Per-strike data for chart — filter to ±35% around spot, convert to $M, sort
    chart_strikes = {k: v / 1e6 for k, v in gex_by_strike.items()
                     if abs(k - spot) <= spot * 0.35}
    chart_strikes = dict(sorted(chart_strikes.items()))

    return {
        "net_gex_bn":    net_gex  / 1e9,
        "call_gex_bn":   call_gex / 1e9,
        "put_gex_bn":    put_gex  / 1e9,
        "regime":        "positive" if net_gex >= 0 else "negative",
        "flip_level":    flip_level,
        "gex_by_strike": chart_strikes,   # {strike: gex_in_$M}
    }


def get_gex_for_strikes(strikes: list, expiry_date: "date", spot: float) -> dict:
    """Fetch per-strike GEX using Deribit public/ticker (returns actual greeks).

    Calls ticker once per option (call + put) for each strike.
    Returns {strike: {"call_gex_m", "put_gex_m", "net_gex_m"}} — all in $M.
    """
    day_str = str(expiry_date.day)               # Deribit: non-zero-padded day
    mon_str = expiry_date.strftime("%b").upper()  # "JUN"
    yr_str  = expiry_date.strftime("%y")          # "26"
    scale   = spot * spot / 1e6                   # gamma × OI × spot² → $M

    per_strike: dict = {}
    for raw_strike in strikes:
        strike_int = int(round(float(raw_strike)))
        call_g = call_oi = put_g = put_oi = 0.0

        for opt_type in ("C", "P"):
            name = f"BTC-{day_str}{mon_str}{yr_str}-{strike_int}-{opt_type}"
            data = _get(
                "https://www.deribit.com/api/v2/public/ticker",
                {"instrument_name": name},
                timeout=8,
            )
            try:
                r  = data["result"]
                g  = float((r.get("greeks") or {}).get("gamma") or 0)
                oi = float(r.get("open_interest") or 0)
                if opt_type == "C":
                    call_g, call_oi = g, oi
                else:
                    put_g,  put_oi  = g, oi
            except Exception:
                pass

        per_strike[float(raw_strike)] = {
            "call_gex_m": call_g * call_oi * scale,
            "put_gex_m":  put_g  * put_oi  * scale,
            "net_gex_m":  (call_g * call_oi - put_g * put_oi) * scale,
        }

    return per_strike


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
    """Try slug patterns ±3 days to find the nearest available BTC daily event.
    Tries both 'bitcoin-above-on-may-22' and 'bitcoin-above-on-may-22-2026' variants.
    Falls back to Gamma market search API if all slug attempts fail.
    """
    offsets = list(range(0, 4)) + list(range(-1, -4, -1))
    for delta in offsets:
        d = start_date + timedelta(days=delta)
        month = d.strftime('%B').lower()
        for slug in (
            f"bitcoin-above-on-{month}-{d.day}",
            f"bitcoin-above-on-{month}-{d.day}-{d.year}",
        ):
            data = _get(f"{GAMMA_API}/events/slug/{slug}", timeout=12)
            if data and data.get("markets"):
                return data, d

    # Last resort: keyword search on the Gamma markets API for start_date
    month = start_date.strftime('%B').lower()
    mdata = _get(
        f"{GAMMA_API}/markets",
        {"search": f"bitcoin above {month} {start_date.day}", "active": "true", "limit": 30},
        timeout=12,
    )
    if mdata and isinstance(mdata, list):
        btc_markets = [m for m in mdata if _parse_strike(m.get("question", "")) is not None]
        if btc_markets:
            return {"markets": btc_markets}, start_date

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
