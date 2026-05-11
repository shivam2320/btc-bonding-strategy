"""Shared helpers for backtests on btc_one_day_price_history.csv.

Each CSV row is one outcome token (any YES/NO spelling). All entry, resolution,
and stop-loss rules use only that row's ``price`` series for the
(event, market, outcome) group — never the other side's prices.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd

IST_TZ = "Asia/Kolkata"
SESSION_START_CLOCK = time(21, 30)

# Non-overlapping bands; 98c matches user spec [0.975, 0.985).
PRICE_TIER_BANDS: tuple[tuple[str, float, float], ...] = (
    ("98c", 0.975, 0.985),
    ("98.5c", 0.985, 0.990),
    ("99c", 0.990, 0.995),
    ("99.5c", 0.995, 0.999),
    ("99.9c", 0.999, 1.0000001),
)

WIN_THRESHOLD = 0.99
LOSE_THRESHOLD = 0.01

DEFAULT_THRESHOLDS = (0.98, 0.985, 0.99, 0.995)
DEFAULT_SL_PCTS = (5, 10, 20, 30, 50)


def normalize_outcome_label(outcome: str) -> str:
    """Collapse yes/No/YES etc. to YES or NO; other labels kept stripped."""
    s = str(outcome).strip()
    u = s.upper()
    if u == "YES":
        return "YES"
    if u == "NO":
        return "NO"
    return s


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_csv_path() -> Path:
    return repo_root() / "btc_one_day_price_history.csv"


def load_price_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["datetime_utc"],
        dtype={
            "event_slug": str,
            "market_slug": str,
            "outcome": str,
            "timestamp": "int64",
            "price": "float64",
        },
    )
    df["outcome"] = df["outcome"].map(normalize_outcome_label)
    df.sort_values(
        ["market_slug", "outcome", "timestamp"],
        kind="mergesort",
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def terminal_label(last_price: float) -> str | None:
    if last_price >= WIN_THRESHOLD:
        return "win"
    if last_price <= LOSE_THRESHOLD:
        return "lose"
    return None


def ensure_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def price_tier(price: float) -> str | None:
    for name, lo, hi in PRICE_TIER_BANDS:
        if lo <= price < hi:
            return name
    return None


def ist_session_columns(dt_utc: pd.Series) -> pd.DataFrame:
    """IST time, session [21:30, next day 21:30) IST, and hour floor in IST."""
    utc = ensure_utc(dt_utc)
    ist = utc.dt.tz_convert(IST_TZ)
    midnight = ist.dt.normalize()
    msm = (
        ist.dt.hour * 60
        + ist.dt.minute
        + ist.dt.second / 60.0
        + ist.dt.microsecond / 60_000_000.0
    )
    cutoff = SESSION_START_CLOCK.hour * 60 + SESSION_START_CLOCK.minute
    on_or_after_open = msm >= cutoff
    anchor_midnight = midnight.where(on_or_after_open, midnight - pd.Timedelta(days=1))
    session_start = anchor_midnight + pd.Timedelta(
        hours=SESSION_START_CLOCK.hour, minutes=SESSION_START_CLOCK.minute
    )
    session_end = session_start + pd.Timedelta(days=1)
    hour_ist = ist.dt.floor("h")
    return pd.DataFrame(
        {
            "dt_ist": ist,
            "session_start_ist": session_start,
            "session_end_ist": session_end,
            "hour_ist": hour_ist,
        }
    )


def first_tick_at_or_above(
    g: pd.DataFrame, threshold: float
) -> tuple[pd.Timestamp, int, float] | None:
    """
    First chronological row where price >= threshold (no hourly bucketing).
    ``g`` must already be sorted by time for this group.
    """
    if g.empty:
        return None
    m = g["price"] >= threshold
    if not m.any():
        return None
    row = g.loc[m].iloc[0]
    return (
        pd.Timestamp(row["datetime_utc"]),
        int(row["timestamp"]),
        float(row["price"]),
    )


def first_hourly_entry(
    g: pd.DataFrame, threshold: float
) -> tuple[pd.Timestamp, int, float] | None:
    """
    First UTC hour where max(price) in that hour >= threshold.
    Returns (entry_datetime of first tick in that hour with price >= threshold,
             entry_timestamp, observed price at that tick).
    """
    if g.empty:
        return None
    hours = g["datetime_utc"].dt.floor("h")
    for hour in sorted(hours.unique()):
        mask = hours == hour
        part = g.loc[mask]
        if part["price"].max() < threshold:
            continue
        row = part.loc[part["price"] >= threshold].iloc[0]
        return (
            pd.Timestamp(row["datetime_utc"]),
            int(row["timestamp"]),
            float(row["price"]),
        )
    return None
