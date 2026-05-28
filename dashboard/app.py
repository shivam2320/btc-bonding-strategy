"""BTC Trade Dashboard — Streamlit app.

Run:
    cd dashboard && streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
import fetchers

IST = ZoneInfo("Asia/Kolkata")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Trade Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg:     #060A0F;
    --card:   #0A1018;
    --border: #141E2A;
    --green:  #00D4A8;
    --red:    #FF3D54;
    --amber:  #E8B000;
    --blue:   #4E9FD4;
    --muted:  #3D5066;
    --text:   #A8BDD0;
    --bright: #CDE0F0;
  }

  /* Global */
  .stApp { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; }
  .block-container { padding: 0.6rem 2rem 2rem; max-width: 100% !important; }

  /* Metric overrides */
  [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700; color: var(--bright) !important; letter-spacing: -0.02em; }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.6rem !important; text-transform: uppercase; letter-spacing: 0.14em; }
  [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

  /* Section header */
  .sec-hdr {
    color: var(--muted);
    font-size: 0.57rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin-bottom: 8px;
    margin-top: 4px;
  }

  /* Card */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 1px;
    padding: 6px 10px;
    margin-bottom: 5px;
    font-size: 0.78rem;
    line-height: 1.45;
    color: var(--text);
  }

  /* Signal dot */
  .dot-green { color: var(--green); font-size: 0.6rem; }
  .dot-red   { color: var(--red);   font-size: 0.6rem; }
  .dot-yel   { color: var(--amber); font-size: 0.6rem; }

  /* Color helpers */
  .green  { color: var(--green); }
  .red    { color: var(--red); }
  .yellow { color: var(--amber); }
  .muted  { color: var(--muted); }
  .blue   { color: var(--blue); }

  /* News */
  .news-card {
    border-left: 2px solid var(--border);
    padding: 5px 10px;
    margin: 4px 0;
    font-size: 0.75rem;
    line-height: 1.4;
  }
  .news-card.high { border-left-color: var(--red); }
  .news-card.info { border-left-color: var(--blue); }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 8px 0; }

  /* Hide streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 1px; }

  /* Terminal header */
  .term-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--green);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    line-height: 1;
  }
  .term-prompt { color: var(--muted); margin-right: 6px; }
  .term-cursor {
    display: inline-block;
    width: 6px;
    height: 0.85em;
    background: var(--green);
    vertical-align: text-bottom;
    margin-left: 3px;
    animation: blink 1.1s step-end infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* Buttons */
  .stButton > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 1px !important;
    padding: 4px 14px !important;
    text-transform: uppercase !important;
  }
  .stButton > button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_ist = datetime.now(IST)

hc1, hc2, hc3 = st.columns([4, 2, 1])
with hc1:
    st.markdown(
        "<div class='term-header'>"
        "<span class='term-prompt'>▶</span>"
        "BTC Bonding Dashboard"
        "<span class='term-cursor'></span>"
        "</div>",
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown(
        f"<span class='muted' style='font-size:0.8rem'>"
        f"IST {now_ist.strftime('%d %b %Y · %H:%M:%S')}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"UTC {now_utc.strftime('%H:%M')}"
        f"</span>",
        unsafe_allow_html=True,
    )
with hc3:
    refresh = st.button("↺ Refresh", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── Fast cache: price/futures data — 60s TTL, cleared on Refresh ──────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_price_data() -> dict:
    ohlc    = fetchers.get_ohlc()
    market  = fetchers.get_btc_market()
    _, session_start, _ = fetchers.active_session_date()
    session_open = fetchers.get_session_open_price(session_start)

    # 24h high/low: prefer Binance (real-time), fall back to last Kraken OHLC candle
    h24 = market.get("high_24h") or 0.0
    l24 = market.get("low_24h")  or 0.0
    if not h24 and ohlc is not None and not ohlc.empty:
        last = ohlc.iloc[-1]
        h24, l24 = float(last["high"]), float(last["low"])

    return {
        "spot":    market.get("spot"),
        "stats24": {
            "change_pct": market.get("change_pct", 0.0),
            "high":        h24,
            "low":         l24,
            "volume_usd":  market.get("volume_usd", 0.0),
        },
        "ohlc":    ohlc,
        "atr":     fetchers.compute_atr(ohlc),
        "emas":    fetchers.compute_emas(ohlc),
        "session_open": session_open,
        "vol":     fetchers.get_volatility_metrics(ohlc),
        "rv":      fetchers.get_realized_vol(ohlc),
        "funding": fetchers.get_funding_rate(),
        "oi":      fetchers.get_open_interest(),
        "poly":    fetchers.get_polymarket_btc_markets(),   # returns (list, found_date)
    }


# ── Slow cache: news/FF/ETF — 5-min TTL, NOT cleared on Refresh ──────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_slow_data() -> dict:
    ff, ff_error = [], False
    try:
        ff = fetchers.get_forex_factory_events()
    except fetchers.RateLimitedError:
        ff_error = True

    return {
        "ff":         ff,
        "ff_error":   ff_error,
        "news":       fetchers.get_crypto_news(),
        "etf_flows":  fetchers.get_etf_flows(),
        "fear_greed": fetchers.get_fear_greed(),
        "dvol":       fetchers.get_deribit_dvol(),
    }


if refresh:
    load_price_data.clear()

with st.spinner("Loading market data…"):
    p = load_price_data()
    s = load_slow_data()

d = {**p, **s}

spot    = d["spot"]
stats24 = d["stats24"]
atr     = d["atr"]
emas    = d["emas"]
vol     = d["vol"]

# Session info — computed fresh each render (no API call)
market_date, session_start, session_end = fetchers.active_session_date()
session_open = d.get("session_open")

# ── Session strip ─────────────────────────────────────────────────────────────
remaining     = session_end - now_ist
total_secs    = max(0, int(remaining.total_seconds()))
hours, r      = divmod(total_secs, 3600)
mins          = r // 60
countdown_str = f"{hours}h {mins:02d}m"

ss1, ss2, ss3, ss4 = st.columns(4)
with ss1:
    st.metric("Session Open (BTC)", f"${session_open:,.0f}" if session_open else "—")
with ss2:
    if session_open and spot:
        sess_diff     = spot - session_open
        sess_diff_pct = sess_diff / session_open * 100
        st.metric("Move Since Open", f"${sess_diff:+,.0f}", delta=f"{sess_diff_pct:+.2f}%")
    else:
        st.metric("Move Since Open", "—")
with ss3:
    st.metric("Session Ends", session_end.strftime("%d %b %H:%M IST"))
with ss4:
    st.metric("Time Remaining", countdown_str)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Intraday chart — TradingView ──────────────────────────────────────────────
st.markdown("<div class='sec-hdr'>Intraday Price Action · BINANCE:BTCUSDT · 1H</div>", unsafe_allow_html=True)
components.html("""
<!DOCTYPE html>
<html>
<head><style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #060A0F; overflow: hidden; }
  #tv_btc { width: 100%; height: 100%; }
</style></head>
<body>
  <div id="tv_btc"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({
    "container_id":        "tv_btc",
    "width":               "100%",
    "height":              "100%",
    "symbol":              "BINANCE:BTCUSDT",
    "interval":            "60",
    "timezone":            "Asia/Kolkata",
    "theme":               "dark",
    "style":               "1",
    "locale":              "en",
    "backgroundColor":     "#060A0F",
    "gridColor":           "rgba(15,26,40,0.9)",
    "toolbar_bg":          "#0A1018",
    "withdateranges":      true,
    "range":               "1D",
    "hide_side_toolbar":   true,
    "allow_symbol_change": false,
    "hide_volume":         false,
    "save_image":          false,
    "enable_publishing":   false,
    "studies":             [],
    "overrides": {
      "paneProperties.background":            "#060A0F",
      "paneProperties.backgroundType":        "solid",
      "paneProperties.vertGridProperties.color": "rgba(15,26,40,0.8)",
      "paneProperties.horzGridProperties.color": "rgba(15,26,40,0.8)",
      "scalesProperties.textColor":           "#3D5066",
      "scalesProperties.backgroundColor":    "#060A0F",
      "candleStyle.upColor":                  "#00D4A8",
      "candleStyle.downColor":                "#FF3D54",
      "candleStyle.borderUpColor":            "#00D4A8",
      "candleStyle.borderDownColor":          "#FF3D54",
      "candleStyle.wickUpColor":              "#00D4A8",
      "candleStyle.wickDownColor":            "#FF3D54"
    }
  });
  </script>
</body>
</html>
""", height=460)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 1: Key price metrics ──────────────────────────────────────────────────
mc = st.columns(6)
with mc[0]:
    st.metric(
        "BTC Spot",
        f"${spot:,.0f}" if spot else "—",
        delta=f"{stats24.get('change_pct', 0):+.2f}%" if stats24 else None,
    )
with mc[1]:
    st.metric("24h High", f"${stats24.get('high', 0):,.0f}" if stats24 else "—")
with mc[2]:
    st.metric("24h Low",  f"${stats24.get('low', 0):,.0f}"  if stats24 else "—")
with mc[3]:
    st.metric("ATR (14d)", f"${atr:,.0f}" if atr else "—")
vol_1m = vol.get("1m") if vol else None
with mc[4]:
    st.metric("Avg H-L % (1M)", f"{vol_1m['avg_hl_pct']:.2f}%" if vol_1m else "—")
with mc[5]:
    st.metric("Avg H-L $ (1M)", f"${vol_1m['avg_hl_usd']:,.0f}" if vol_1m else "—")

st.markdown("<hr>", unsafe_allow_html=True)


# ── Row 2: Three columns ──────────────────────────────────────────────────────
col_trend, col_verdict, col_vol = st.columns([1, 1.1, 1.2], gap="medium")


# ── COLUMN 1 · Trend Analysis ─────────────────────────────────────────────────
with col_trend:
    st.markdown("<div class='sec-hdr'>Trend Analysis · EMA</div>", unsafe_allow_html=True)

    if emas:
        for period, val in emas.items():
            if spot:
                diff_pct = (spot - val) / val * 100
                color = "green" if diff_pct > 0 else "red"
                arrow = "▲" if diff_pct > 0 else "▼"
                diff_str = f"&nbsp;&nbsp;<span class='{color}'>{arrow} {diff_pct:+.2f}%</span>"
            else:
                diff_str = ""
            st.markdown(
                f"<div class='card'>"
                f"<span class='muted'>EMA {period}</span>&nbsp;&nbsp;"
                f"<b>${val:,.0f}</b>{diff_str}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='card muted'>EMA data unavailable</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='sec-hdr'>Futures Metrics</div>", unsafe_allow_html=True)

    funding = d["funding"]
    if funding:
        rate = funding.get("rate_pct", 0)
        color = "green" if rate > 0 else "red"
        next_ts = funding.get("next_funding")
        next_str = next_ts.strftime("%H:%M UTC") if next_ts else ""
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Funding Rate</span>&nbsp;&nbsp;"
            f"<span class='{color}'><b>{rate:+.4f}%</b></span>"
            f"<br><span class='muted' style='font-size:0.75rem'>Next: {next_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    oi = d["oi"]
    if oi:
        oi_str = f"{oi:,.0f} BTC"
        if spot:
            oi_usd = oi * spot
            oi_str = f"{oi:,.0f} BTC (${oi_usd/1e9:.2f}B)"
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Open Interest</span>&nbsp;&nbsp;<b>{oi_str}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br><div class='sec-hdr'>BTC ETF Net Flows</div>", unsafe_allow_html=True)

    etf_flows = d.get("etf_flows")
    if etf_flows:
        rows = [
            ("Today",      etf_flows["today"]),
            ("Last Week",  etf_flows["last_week"]),
            ("Last Month", etf_flows["last_month"]),
            ("3 Months",   etf_flows["three_months"]),
        ]
        for label, val in rows:
            color = "green" if val > 0 else "red"
            arrow = "▲" if val > 0 else "▼"
            is_today = label == "Today"
            weight = "font-size:1rem;" if is_today else "font-size:0.85rem;"
            st.markdown(
                f"<div class='card' style='padding:7px 14px;display:flex;"
                f"justify-content:space-between;align-items:center'>"
                f"<span class='muted'>{label}</span>"
                f"<span class='{color}' style='{weight}'><b>{arrow} ${val:,.1f}M</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='card muted'>ETF flow data unavailable</div>", unsafe_allow_html=True)


# ── COLUMN 2 · Trade Verdict ──────────────────────────────────────────────────
with col_verdict:
    st.markdown("<div class='sec-hdr'>Trade Verdict · High Impact Events Today</div>", unsafe_allow_html=True)

    ff = d["ff"]
    ff_error = d.get("ff_error", False)

    if ff_error:
        st.markdown(
            "<div class='news-card high'><span class='yellow'>⚠ FOREX FACTORY RATE-LIMITED — "
            "auto-retry in ~5 min</span></div>",
            unsafe_allow_html=True,
        )
    elif ff:
        FLAGS = {"USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵",
                 "CAD": "🇨🇦", "AUD": "🇦🇺", "NZD": "🇳🇿", "CHF": "🇨🇭"}
        for e in ff:
            flag = FLAGS.get(e["country"], "🌐")
            actual_str = f" &nbsp;·&nbsp; <b>Actual: {e['actual']}</b>" if e.get("actual") else ""
            st.markdown(
                f"<div class='news-card high'>"
                f"<b>{e['title']}</b><br>"
                f"<span class='muted'>{flag} {e['country']} &nbsp;·&nbsp; "
                f"{e['time_ist']} ({e['time_utc']})"
                f" &nbsp;·&nbsp; F: {e['forecast']} &nbsp;·&nbsp; P: {e['previous']}"
                f"{actual_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='news-card info'><span class='muted'>NO HIGH-IMPACT MACRO EVENTS TODAY</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br><div class='sec-hdr'>BTC News</div>", unsafe_allow_html=True)
    news = d["news"]
    if news:
        for item in news:
            pub = item["published_on"].strftime("%H:%M") if "published_on" in item else ""
            st.markdown(
                f"<div class='news-card info'>"
                f"<a href='{item['url']}' target='_blank' "
                f"style='color:#4E9FD4;text-decoration:none;font-weight:600'>{item['title']}</a>"
                f"<br><span class='muted'>{item['source']} · {pub} UTC</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='news-card muted'>News unavailable</div>", unsafe_allow_html=True)


# ── COLUMN 3 · Volatility Details ────────────────────────────────────────────
with col_vol:
    st.markdown("<div class='sec-hdr'>Live Volatility Signals</div>", unsafe_allow_html=True)

    # ── Fear & Greed ──────────────────────────────────────────────────────────
    fg = d.get("fear_greed")
    if fg:
        val   = fg["value"]
        label = fg["classification"]
        if val <= 25:
            fg_color = "red"
        elif val <= 45:
            fg_color = "yellow"
        elif val <= 55:
            fg_color = "muted"
        else:
            fg_color = "green"
        filled = round(val / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Fear & Greed</span>&nbsp;&nbsp;"
            f"<span class='{fg_color}'><b>{val} — {label}</b></span><br>"
            f"<span style='letter-spacing:2px;font-size:0.8rem;color:#3D5066'>{bar}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Realised vol ──────────────────────────────────────────────────────────
    rv = d.get("rv")
    if rv:
        rv7, rv30     = rv["rv7"], rv["rv30"]
        expanding     = rv7 > rv30
        rv_color      = "red" if expanding else "green"
        rv_arrow      = "▲" if expanding else "▼"
        rv_label      = "Expanding" if expanding else "Contracting"
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Realised Vol</span><br>"
            f"<span class='muted' style='font-size:0.75rem'>7d</span>&nbsp;"
            f"<b>{rv7:.1f}%</b>"
            f"&nbsp;&nbsp;<span class='muted' style='font-size:0.75rem'>30d</span>&nbsp;"
            f"<b>{rv30:.1f}%</b>"
            f"<br><span class='{rv_color}' style='font-size:0.75rem'>"
            f"{rv_arrow} Vol {rv_label} (7d {'>' if expanding else '<'} 30d)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Deribit DVOL ──────────────────────────────────────────────────────────
    dvol = d.get("dvol")
    if dvol:
        if dvol < 50:
            dvol_color, dvol_label = "green",  "Low"
        elif dvol < 80:
            dvol_color, dvol_label = "yellow", "Moderate"
        else:
            dvol_color, dvol_label = "red",    "High"
        daily_move_pct = dvol / (365 ** 0.5)
        daily_move_usd = f"&nbsp;·&nbsp; ${daily_move_pct / 100 * spot:,.0f}" if spot else ""
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Deribit DVOL (30d IV)</span>"
            f"&nbsp;&nbsp;<b>{dvol:.1f}</b>"
            f"&nbsp;&nbsp;<span class='{dvol_color}'>{dvol_label}</span><br>"
            f"<span class='muted' style='font-size:0.75rem'>Expected daily move: "
            f"<b class='blue'>{daily_move_pct:.2f}%</b>{daily_move_usd}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not fg and not rv and not dvol:
        st.markdown("<div class='card muted'>Live signals unavailable</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='sec-hdr'>Historical Volatility</div>", unsafe_allow_html=True)

    def _vol_table(m: dict, label: str) -> None:
        st.markdown(f"<div class='sec-hdr' style='margin-top:6px'>{label}</div>", unsafe_allow_html=True)
        rows = [
            ("Avg volatility % (open/close)", f"{m['avg_oc_pct']:.2f}%"),
            ("Avg volatility % (high/low)",   f"{m['avg_hl_pct']:.2f}%"),
            ("Avg volatility USD (open/close)", f"${m['avg_oc_usd']:,.0f}"),
            ("Avg volatility USD (high/low)",   f"${m['avg_hl_usd']:,.0f}"),
            (f"Max % move in a day", f"{m['max_hl_pct']:.2f}% <span class='muted' style='font-size:0.73rem'>({m['max_hl_pct_date']})</span>"),
            (f"Max USD move in a day", f"${m['max_hl_usd']:,.0f} <span class='muted' style='font-size:0.73rem'>({m['max_hl_usd_date']})</span>"),
        ]
        for label_row, value in rows:
            st.markdown(
                f"<div class='card' style='padding:7px 14px;display:flex;justify-content:space-between;align-items:center'>"
                f"<span class='muted'>{label_row}</span>"
                f"<b>{value}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if vol:
        if vol.get("1m"):
            _vol_table(vol["1m"], "1 Month")
        if vol.get("6m"):
            _vol_table(vol["6m"], "6 Months")
    else:
        st.markdown(
            "<div class='card muted'>Local CSVs not found.<br>"
            "Place BTCUSD CSVs in project root.</div>",
            unsafe_allow_html=True,
        )

    if atr and spot:
        st.markdown("<br>", unsafe_allow_html=True)
        atr_pct = atr / spot * 100
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>ATR as % of spot</span>&nbsp;&nbsp;"
            f"<b>{atr_pct:.2f}%</b>"
            f"<br><span class='muted' style='font-size:0.75rem'>"
            f"Expected 1-day move: ${atr:,.0f} (1×ATR) — ${atr*1.5:,.0f} (1.5×ATR)"
            f"</span></div>",
            unsafe_allow_html=True,
        )



# Unpack poly data — market_date/session_start/session_end already computed above
poly_list, found_date = d["poly"]


# ── Row 3: AI Analysis ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

ai_hdr, ai_btn = st.columns([5, 1])
with ai_hdr:
    st.markdown("<div class='sec-hdr'>AI Analysis · Bonding Strategy</div>", unsafe_allow_html=True)
with ai_btn:
    run_ai = st.button("🤖 Analyze", use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_ai_analysis(
    spot, change_pct,
    ema20, ema50, ema200,
    atr,
    vol1m_hl_pct, vol6m_hl_pct,
    dvol, daily_move_pct,
    rv7, rv30,
    fg_val, fg_label,
    funding_rate,
    etf_flow_today,
    ff_count,
    news_headlines,
    poly_summary,
) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "⚠ Add `OPENROUTER_API_KEY` to Streamlit secrets to enable AI analysis."

    def _fmt(val, fmt, prefix="", suffix=""):
        return f"{prefix}{val:{fmt}}{suffix}" if val is not None else "N/A"

    context = f"""BTC MARKET SNAPSHOT
Spot: {_fmt(spot, ",.0f", "$")}  |  24h change: {_fmt(change_pct, "+.2f", suffix="%")}
EMA20: {_fmt(ema20, ",.0f", "$")}  |  EMA50: {_fmt(ema50, ",.0f", "$")}  |  EMA200: {_fmt(ema200, ",.0f", "$")}
ATR (14d): {_fmt(atr, ",.0f", "$")}
Avg daily H-L% 1M: {_fmt(vol1m_hl_pct, ".2f", suffix="%")}  |  6M: {_fmt(vol6m_hl_pct, ".2f", suffix="%")}
Deribit DVOL: {_fmt(dvol, ".1f")}  |  Expected daily move: {_fmt(daily_move_pct, ".2f", suffix="%")}
Realised vol — 7d: {_fmt(rv7, ".1f", suffix="%")}  |  30d: {_fmt(rv30, ".1f", suffix="%")}{"  (EXPANDING)" if rv7 and rv30 and rv7 > rv30 else "  (CONTRACTING)" if rv7 and rv30 else ""}
Fear & Greed: {fg_val if fg_val is not None else "N/A"} — {fg_label}
Funding rate: {_fmt(funding_rate, "+.4f", suffix="%")}
ETF net flow today: {_fmt(etf_flow_today, "+,.1f", "$", "M")}
High-impact macro events today: {ff_count}

BTC NEWS HEADLINES (latest):
{news_headlines}

POLYMARKET BTC MARKETS (today's session)
{poly_summary}"""

    prompt = f"""{context}

STRATEGY: Polymarket BTC daily BONDING strategy.
- Target YES or NO tokens priced between 99¢ and 99.9¢ (marked ★ in the market data).
- YES bond: buy YES in (99¢, 99.9¢) when spot is well ABOVE strike → profit if BTC stays above strike.
- NO bond: buy NO in (99¢, 99.9¢) when spot is well BELOW strike → profit if BTC stays below strike.
- Profit = 100¢ − entry price (0.1¢ to 1¢ per token). Session ends 9:30 PM IST.
- Prices at 100¢ offer NO profit margin — ignore them. Prices below 99¢ carry too much risk.
- This is NOT directional speculation. We exploit near-certain outcomes with defined cushion.

CUSHION ANALYSIS: Safety ratio = (spot − strike) ÷ expected daily move
  > 2.0 → strong  |  1.5–2.0 → acceptable  |  < 1.5 → avoid

Respond in EXACTLY this format (2–3 sentences each, specific numbers required):

**TRADE DECISION**: [TRADE / SKIP / CAUTION]
[Lead with the news headlines — do any suggest imminent shock (regulation, macro event, whale move, exchange issue)? Then assess: DVOL level, vol regime (expanding/contracting), ETF flows, Fear & Greed, macro event count. TRADE = calm news + calm vol; CAUTION = one risk factor; SKIP = news risk OR multiple vol risks stacking up OR no valid bond exists.]

**TREND**:
[Start with what the news headlines say about current market narrative — bullish catalyst, bearish pressure, or neutral? Then: where is spot vs EMA20/50/200? Is today's 24h move a continuation or reversal? What does funding rate suggest? Conclude which side (YES or NO bond) has more tailwind given both price structure and news.]

**RISK ASSESSMENT**:
[What specific events or conditions could cause BTC to breach the target strike during this session? Quantify: how many expected daily moves would need to occur? Reference ATR, historical max H-L, and any macro catalysts. This is about TAIL RISK, not trend.]

**STRIKE SELECTION**:
[Name the exact strike, YES or NO, and its entry price in ¢ (must be between 99¢ and 99.9¢). State the cushion in USD and the safety ratio. Explain in one sentence what single event would invalidate this trade.]"""

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={"X-Title": "BTC Trade Dashboard"},
        )
        resp = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Analysis unavailable: {e}"


if run_ai:
    load_ai_analysis.clear()

# Build poly summary string for the prompt — flag bonding opportunities explicitly
if poly_list and spot:
    _poly_lines = []
    for m in poly_list[:15]:
        if m["strike"] and m["yes_price"] is not None:
            diff   = spot - m["strike"]
            yes_c  = m["yes_price"] * 100
            no_c   = m["no_price"]  * 100 if m["no_price"] else 0
            if 99 < yes_c < 99.9:
                flag = "  ★ YES BOND"
            elif 99 < no_c < 99.9:
                flag = "  ★ NO BOND"
            elif 97 <= yes_c <= 99:
                flag = "  (YES near-bond)"
            elif 97 <= no_c <= 99:
                flag = "  (NO near-bond)"
            else:
                flag = ""
            _poly_lines.append(
                f"  Strike ${m['strike']:,.0f}"
                f"  spot {'above' if diff > 0 else 'below'} by ${abs(diff):,.0f}"
                f"  YES={yes_c:.0f}¢  NO={no_c:.0f}¢"
                f"  Vol=${m['volume']:,.0f}{flag}"
            )
    _poly_summary = "\n".join(_poly_lines) or "No markets"
else:
    _poly_summary = "No markets available"

_rv   = d.get("rv") or {}
_fg   = d.get("fear_greed") or {}
_vol1 = (d.get("vol") or {}).get("1m") or {}
_vol6 = (d.get("vol") or {}).get("6m") or {}
_fund = d.get("funding") or {}
_etff = d.get("etf_flows") or {}

_news = d.get("news") or []
_news_summary = "\n".join(
    f"• {item['title']} ({item['source']})" for item in _news[:8]
) or "No headlines available"

with st.spinner("Running AI analysis…"):
    analysis = load_ai_analysis(
        spot            = spot,
        change_pct      = (d.get("stats24") or {}).get("change_pct"),
        ema20           = emas.get(20),
        ema50           = emas.get(50),
        ema200          = emas.get(200),
        atr             = atr,
        vol1m_hl_pct    = _vol1.get("avg_hl_pct"),
        vol6m_hl_pct    = _vol6.get("avg_hl_pct"),
        dvol            = d.get("dvol"),
        daily_move_pct  = d["dvol"] / (365 ** 0.5) if d.get("dvol") else None,
        rv7             = _rv.get("rv7"),
        rv30            = _rv.get("rv30"),
        fg_val          = _fg.get("value"),
        fg_label        = _fg.get("classification", ""),
        funding_rate    = _fund.get("rate_pct"),
        etf_flow_today  = _etff.get("today"),
        ff_count        = len(d.get("ff") or []),
        news_headlines  = _news_summary,
        poly_summary    = _poly_summary,
    )

st.markdown(
    f"<div class='card' style='padding:14px 18px;line-height:1.9;font-size:0.8rem;border-left:2px solid #00D4A8'>"
    f"{analysis.replace(chr(10), '<br>')}"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Row 4: Polymarket Markets ─────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sec-hdr'>"
    f"Polymarket · BTC Markets — resolves {market_date.strftime('%d %b %Y')} &nbsp;·&nbsp; "
    f"Session {session_start.strftime('%d %b %H:%M')} → {session_end.strftime('%d %b %H:%M')} IST"
    f"</div>",
    unsafe_allow_html=True,
)

if found_date and found_date != market_date:
    st.markdown(
        f"<div class='news-card' style='border-left-color:#E8B000;margin-bottom:8px'>"
        f"<span class='yellow'>⚠ {market_date.strftime('%d %b').upper()} MARKETS NOT YET LISTED — "
        f"SHOWING {found_date.strftime('%d %b %Y').upper()} INSTEAD</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

if poly_list and spot:
    rows       = []
    bond_types = []
    for m in poly_list:
        strike   = m["strike"]
        diff_usd = (spot - strike) if strike else None
        diff_pct = (diff_usd / strike * 100) if strike else None
        yes_p    = m["yes_price"]
        no_p     = m["no_price"]
        yes_c    = yes_p * 100 if yes_p is not None else 0.0
        no_c     = no_p  * 100 if no_p  is not None else 0.0
        direction = "▲ ITM" if diff_usd and diff_usd > 0 else "▼ OTM"
        if 99 < yes_c < 99.9:
            bond_types.append("yes")
        elif 99 < no_c < 99.9:
            bond_types.append("no")
        else:
            bond_types.append("")
        rows.append({
            "Strike ($)":    f"${strike:,.0f}"        if strike           else "—",
            "Spot − Strike": f"${diff_usd:+,.0f}"     if diff_usd is not None else "—",
            "Diff %":        f"{diff_pct:+.2f}%"       if diff_pct is not None else "—",
            "Position":      direction,
            "YES (¢)":       f"{yes_c:.1f}¢"           if yes_p is not None else "—",
            "NO (¢)":        f"{no_c:.1f}¢"            if no_p  is not None else "—",
            "Volume ($)":    f"${m['volume']:,.0f}",
            "Question":      m["question"],
        })

    df = pd.DataFrame(rows)

    def _style_bonds(row):
        bt = bond_types[row.name] if row.name < len(bond_types) else ""
        if bt == "yes":
            return ["background-color: #062018; color: #00D4A8; font-weight: 700"] * len(row)
        if bt == "no":
            return ["background-color: #061428; color: #4E9FD4; font-weight: 700"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_style_bonds, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

elif poly_list:
    st.info("Spot price unavailable — showing raw Polymarket data.")
    rows = [
        {
            "Strike ($)": f"${m['strike']:,.0f}"         if m["strike"]    else "—",
            "YES (¢)":    f"{m['yes_price']*100:.1f}¢"  if m["yes_price"] else "—",
            "NO (¢)":     f"{m['no_price']*100:.1f}¢"   if m["no_price"]  else "—",
            "Volume ($)": f"${m['volume']:,.0f}",
            "Question":   m["question"],
        }
        for m in poly_list
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info(
        f"No Polymarket BTC markets found near {market_date.strftime('%d %b %Y')} "
        "(checked ±3 days). Markets may not be listed yet."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<span class='muted' style='font-size:0.72rem'>"
    f"Data sources: Binance · Kraken · Bybit/OKX · Deribit · Alternative.me · CMC · ForexFactory · CryptoCompare · Polymarket Gamma API"
    f"&nbsp;|&nbsp; AI: Gemini 2.0 Flash &nbsp;|&nbsp; Cache TTL: 60s &nbsp;|&nbsp; "
    f"Last loaded: {now_utc.strftime('%H:%M:%S')} UTC"
    f"</span>",
    unsafe_allow_html=True,
)
