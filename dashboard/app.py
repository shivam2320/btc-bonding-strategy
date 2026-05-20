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
  /* Global */
  .stApp { background: #0d1117; color: #e6edf3; font-family: 'SF Mono', 'Fira Code', monospace; }
  .block-container { padding: 1rem 2rem 2rem; }

  /* Metric overrides */
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; letter-spacing: -0.02em; }
  [data-testid="stMetricLabel"] { color: #7d8590 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
  [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

  /* Section header */
  .sec-hdr {
    color: #7d8590;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 5px;
    margin-bottom: 10px;
    margin-top: 4px;
  }

  /* Card */
  .card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    line-height: 1.5;
  }

  /* Signal dot */
  .dot-green { color: #3fb950; font-size: 0.65rem; }
  .dot-red   { color: #f85149; font-size: 0.65rem; }
  .dot-yel   { color: #d29922; font-size: 0.65rem; }

  /* Color helpers */
  .green  { color: #3fb950; }
  .red    { color: #f85149; }
  .yellow { color: #d29922; }
  .muted  { color: #7d8590; }
  .blue   { color: #58a6ff; }

  /* News */
  .news-card {
    border-left: 2px solid #21262d;
    padding: 6px 12px;
    margin: 5px 0;
    font-size: 0.82rem;
    line-height: 1.45;
  }
  .news-card.high { border-left-color: #f85149; }
  .news-card.info { border-left-color: #388bfd; }

  /* Divider */
  hr { border-color: #21262d !important; margin: 10px 0; }

  /* Hide streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_ist = datetime.now(IST)

hc1, hc2, hc3 = st.columns([4, 2, 1])
with hc1:
    st.markdown("## ₿ BTC Trade Dashboard")
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


# ── Fast cache: price/futures data — 30s TTL, cleared on Refresh ──────────────
@st.cache_data(ttl=30, show_spinner=False)
def load_price_data() -> dict:
    ohlc = fetchers.get_ohlc(interval="1d", limit=50)
    return {
        "spot":    fetchers.get_spot_price(),
        "stats24": fetchers.get_24h_stats(),
        "ohlc":    ohlc,
        "atr":     fetchers.compute_atr(ohlc),
        "emas":    fetchers.compute_emas(ohlc),
        "funding": fetchers.get_funding_rate(),
        "oi":      fetchers.get_open_interest(),
        "poly":    fetchers.get_polymarket_btc_markets(),
    }


# ── Slow cache: news/FF/ETF/vol — 5-min TTL, NOT cleared on Refresh ───────────
@st.cache_data(ttl=300, show_spinner=False)
def load_slow_data() -> dict:
    ff, ff_error = [], False
    try:
        ff = fetchers.get_forex_factory_events()
    except fetchers.RateLimitedError:
        ff_error = True

    return {
        "vol":      fetchers.get_volatility_metrics(),
        "ff":       ff,
        "ff_error": ff_error,
        "news":     fetchers.get_crypto_news(),
        "etf":      fetchers.get_etf_performance(),
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

    if emas and spot:
        for period, val in emas.items():
            diff_pct = (spot - val) / val * 100
            color = "green" if diff_pct > 0 else "red"
            arrow = "▲" if diff_pct > 0 else "▼"
            st.markdown(
                f"<div class='card'>"
                f"<span class='muted'>EMA {period}</span>&nbsp;&nbsp;"
                f"<b>${val:,.0f}</b>&nbsp;&nbsp;"
                f"<span class='{color}'>{arrow} {diff_pct:+.2f}%</span>"
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

    st.markdown("<br><div class='sec-hdr'>BTC ETF Performance (1d)</div>", unsafe_allow_html=True)

    etf = d["etf"]
    if etf:
        for e in etf:
            color = "green" if e["change_pct"] > 0 else "red"
            arrow = "▲" if e["change_pct"] > 0 else "▼"
            st.markdown(
                f"<div class='card'>"
                f"<b>{e['ticker']}</b>&nbsp;&nbsp;${e['price']:.2f}"
                f"&nbsp;&nbsp;<span class='{color}'>{arrow} {e['change_pct']:+.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='card muted'>ETF data unavailable</div>", unsafe_allow_html=True)


# ── COLUMN 2 · Trade Verdict ──────────────────────────────────────────────────
with col_verdict:
    st.markdown("<div class='sec-hdr'>Trade Verdict · High Impact Events Today</div>", unsafe_allow_html=True)

    ff = d["ff"]
    ff_error = d.get("ff_error", False)

    if ff_error:
        st.markdown(
            "<div class='news-card high'><span class='yellow'>⚠ ForexFactory rate-limited. "
            "Data will auto-retry in ~5 min.</span></div>",
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
            "<div class='news-card info'><span class='muted'>No high-impact macro events today.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br><div class='sec-hdr'>Signal Summary</div>", unsafe_allow_html=True)

    signals: list[tuple[str, str, str]] = []
    if spot and emas:
        ema20  = emas.get(20,  0)
        ema50  = emas.get(50,  0)
        ema200 = emas.get(200, 0)
        signals.append(("Above EMA 20",  "green" if spot > ema20  else "red",  "Bullish short-term" if spot > ema20  else "Bearish short-term"))
        signals.append(("Above EMA 50",  "green" if spot > ema50  else "red",  "Bullish mid-term"   if spot > ema50  else "Bearish mid-term"))
        signals.append(("Above EMA 200", "green" if spot > ema200 else "red",  "Bull market"        if spot > ema200 else "Bear market"))

    if funding:
        rate = funding.get("rate_pct", 0)
        if abs(rate) > 0.05:
            color = "red" if rate > 0 else "green"
            label = "High longs (caution)" if rate > 0 else "High shorts (squeeze risk)"
            signals.append(("Funding Extreme", color, label))
        else:
            signals.append(("Funding Neutral", "yellow", "Balanced positioning"))

    if ff:
        signals.append(("⚠ High Impact News", "yellow", f"{len(ff)} event(s) — expect volatility"))

    if vol_1m:
        avg_hl = vol_1m["avg_hl_pct"]
        if avg_hl > 3:
            signals.append(("High Volatility", "yellow", f"Avg H-L {avg_hl:.2f}% — wide moves expected"))
        else:
            signals.append(("Normal Volatility", "green", f"Avg H-L {avg_hl:.2f}%"))

    for label, color, note in signals:
        dot_class = f"dot-{color}"
        st.markdown(
            f"<div class='card' style='padding:8px 14px'>"
            f"<span class='{dot_class}'>●</span>&nbsp;"
            f"<b>{label}</b>"
            f"<br><span class='muted' style='font-size:0.75rem;padding-left:14px'>{note}</span>"
            f"</div>",
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
                f"style='color:#58a6ff;text-decoration:none;font-weight:600'>{item['title']}</a>"
                f"<br><span class='muted'>{item['source']} · {pub} UTC</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='news-card muted'>News unavailable</div>", unsafe_allow_html=True)


# ── COLUMN 3 · Volatility Details ────────────────────────────────────────────
with col_vol:
    st.markdown("<div class='sec-hdr'>Volatility Analysis</div>", unsafe_allow_html=True)

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

    if spot and atr:
        st.markdown("<br><div class='sec-hdr'>Strike Distance Guide</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='card'>"
            f"<span class='muted'>Spot</span> ${spot:,.0f}<br>"
            f"<span class='muted'>+1 ATR</span> ${spot+atr:,.0f}&nbsp;&nbsp;"
            f"<span class='muted'>−1 ATR</span> ${spot-atr:,.0f}<br>"
            f"<span class='muted'>+2 ATR</span> ${spot+2*atr:,.0f}&nbsp;&nbsp;"
            f"<span class='muted'>−2 ATR</span> ${spot-2*atr:,.0f}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Row 3: Polymarket Markets ─────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

market_date, session_start, session_end = fetchers.active_session_date()
st.markdown(
    f"<div class='sec-hdr'>"
    f"Polymarket · BTC Markets — resolves {market_date.strftime('%d %b %Y')} &nbsp;·&nbsp; "
    f"Session {session_start.strftime('%d %b %H:%M')} → {session_end.strftime('%d %b %H:%M')} IST"
    f"</div>",
    unsafe_allow_html=True,
)

poly = d["poly"]

if poly and spot:
    rows = []
    for m in poly:
        strike = m["strike"]
        diff_usd = (spot - strike) if strike else None
        diff_pct = (diff_usd / strike * 100) if strike else None
        yes_p = m["yes_price"]
        no_p  = m["no_price"]

        direction = "—"
        if diff_usd is not None:
            direction = "▲ ITM" if diff_usd > 0 else "▼ OTM"

        rows.append({
            "Strike ($)":    f"${strike:,.0f}"        if strike           else "—",
            "Spot − Strike": f"${diff_usd:+,.0f}"     if diff_usd is not None else "—",
            "Diff %":        f"{diff_pct:+.2f}%"       if diff_pct is not None else "—",
            "Position":      direction,
            "YES (¢)":       f"{yes_p*100:.1f}¢"       if yes_p is not None else "—",
            "NO (¢)":        f"{no_p*100:.1f}¢"        if no_p  is not None else "—",
            "Volume ($)":    f"${m['volume']:,.0f}",
            "Question":      m["question"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)

elif poly:
    st.info("Spot price unavailable — showing raw Polymarket data.")
    rows = [
        {
            "Strike ($)": f"${m['strike']:,.0f}" if m["strike"] else "—",
            "YES (¢)":    f"{m['yes_price']*100:.1f}¢" if m["yes_price"] else "—",
            "NO (¢)":     f"{m['no_price']*100:.1f}¢"  if m["no_price"]  else "—",
            "Volume ($)": f"${m['volume']:,.0f}",
            "Question":   m["question"],
        }
        for m in poly
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info(
        f"No Polymarket BTC markets found for {market_date.strftime('%d %b %Y')}. "
        "Markets may not be listed yet or have already expired."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<span class='muted' style='font-size:0.72rem'>"
    f"Data sources: Binance REST · ForexFactory · CryptoCompare · Polymarket Gamma API · Yahoo Finance · "
    f"Local OHLC CSVs &nbsp;|&nbsp; Cache TTL: 60s &nbsp;|&nbsp; "
    f"Last loaded: {now_utc.strftime('%H:%M:%S')} UTC"
    f"</span>",
    unsafe_allow_html=True,
)
