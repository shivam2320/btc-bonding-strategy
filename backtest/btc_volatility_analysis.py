#!/usr/bin/env python3
"""
Volatility stats from Perplexity BTC OHLC CSVs (1Y daily, 6M daily, 1M hourly).

**Close-based volatility**
1. Daily closing prices (daily bars; 1M file is hourly resampled to daily close).
2. Log returns: r_t = ln(P_t / P_{t-1}).
3. Mean return μ = mean(r).
4. Sample variance (N−1): sum((r_t − μ)²) / (N−1).
5. Historical daily σ = sqrt(variance), reported as % (σ × 100).
6. EWMA: σ₀² = historical variance; σ_t² = λ σ_{t-1}² + (1−λ) r_{t-1}² with default λ=0.94;
   report latest σ_t in %.

Metrics 1–4: mean daily |O−C|/O and (H−L)/O (same as before).

Max day: largest daily (H−L)/open and H−L USD, with dates and optional plots.

Weekday/weekend splits use the daily bar date. Close-to-close log returns are
classified by the ending close date: Saturday/Sunday returns are weekend returns.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parent
ROOT = BACKTEST_DIR.parent 
PLOT_DIR = BACKTEST_DIR / "btc_volatility_plots"

CSV_CONFIG: list[tuple[str, str, Path, str]] = [
    (
        "1y_1d",
        "1Y (1 day bars)",
        ROOT / "BTCUSD_1Y_1DAY_FROM_PERPLEXITY (1).csv",
        "daily",
    ),
    (
        "6m_1d",
        "6M (1 day bars)",
        ROOT / "BTCUSD_6M_1DAY_FROM_PERPLEXITY (1).csv",
        "daily",
    ),
    (
        "1m_1h",
        "1M (1 hour bars)",
        ROOT / "BTCUSD_1M_1HOUR_FROM_PERPLEXITY (1).csv",
        "hourly",
    ),
]


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.loc[df["open"] > 0]
    df = df.sort_values("date")
    return df


def daily_ohlc_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day in the series' timezone (from date index)."""
    d = df.set_index("date").sort_index()
    agg = d.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna(subset=["open", "high", "low", "close"]).loc[lambda x: x["open"] > 0]


def ensure_dt_index(daily: pd.DataFrame) -> pd.DataFrame:
    if isinstance(daily.index, pd.DatetimeIndex):
        return daily
    return daily.set_index("date")


def bar_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "avg_oc_pct": float("nan"),
            "avg_hl_pct": float("nan"),
            "avg_oc_usd": float("nan"),
            "avg_hl_usd": float("nan"),
        }
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    oc_pct = ((c - o).abs() / o * 100).mean()
    hl_pct = ((h - l) / o * 100).mean()
    oc_usd = (c - o).abs().mean()
    hl_usd = (h - l).mean()
    return {
        "avg_oc_pct": float(oc_pct),
        "avg_hl_pct": float(hl_pct),
        "avg_oc_usd": float(oc_usd),
        "avg_hl_usd": float(hl_usd),
    }


def daily_log_returns_from_close(daily: pd.DataFrame) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}) on daily closes; index = date of P_t."""
    dd = ensure_dt_index(daily)
    c = dd["close"].astype(float)
    r = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return r.rename("log_return")


def historical_volatility_from_returns(r: pd.Series) -> tuple[float, float, float]:
    """
    Steps 3–5: μ, (N−1) variance, σ in %.
    Returns (sigma_pct, mu_as_pct, variance_decimal).
    """
    r = r.astype(float).dropna()
    n = len(r)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mu = float(r.mean())
    var = float(((r - mu) ** 2).sum() / (n - 1))
    sigma_pct = float(np.sqrt(var) * 100.0)
    return sigma_pct, float(mu * 100.0), var


def ewma_volatility_series_pct(
    r: pd.Series, lam: float = 0.94, initial_var: float | None = None
) -> pd.Series:
    """
    Step 8–9: σ_t² = λ σ_{t-1}² + (1−λ) r_{t-1}² with r in decimal log returns.
    σ_0² = historical sample variance (same as Step 4) unless initial_var is set.
    Returns daily EWMA σ in percent, one value per return date (after each r is applied).
    """
    r = r.astype(float).dropna()
    n = len(r)
    if n < 1:
        return pd.Series(dtype=float)
    if initial_var is None:
        if n < 2:
            initial_var = float(r.iloc[0] ** 2)
        else:
            mu = float(r.mean())
            initial_var = float(((r - mu) ** 2).sum() / (n - 1))
    sigma2_prev = initial_var
    out_pct: list[float] = []
    for i in range(n):
        sigma2_prev = lam * sigma2_prev + (1.0 - lam) * (float(r.iloc[i]) ** 2)
        out_pct.append(np.sqrt(sigma2_prev) * 100.0)
    return pd.Series(out_pct, index=r.index, name="ewma_sigma_pct")


def max_move_series_and_dates(
    daily: pd.DataFrame,
) -> tuple[float, float, pd.Timestamp | None, pd.Timestamp | None, pd.Series, pd.Series]:
    """
    Returns max hl%, max hl USD, date of max %, date of max USD,
    and the full daily series (indexed by datetime) for plotting.
    """
    dd = ensure_dt_index(daily)
    if dd.empty:
        empty = pd.Series(dtype=float)
        return float("nan"), float("nan"), None, None, empty, empty
    o = dd["open"]
    h = dd["high"]
    l = dd["low"]
    hl_pct = (h - l) / o * 100
    hl_usd = h - l
    idx_pct = hl_pct.idxmax()
    idx_usd = hl_usd.idxmax()
    return (
        float(hl_pct.max()),
        float(hl_usd.max()),
        pd.Timestamp(idx_pct),
        pd.Timestamp(idx_usd),
        hl_pct,
        hl_usd,
    )


def split_daily_by_day_type(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dd = ensure_dt_index(daily)
    weekday_mask = dd.index.dayofweek < 5
    return {
        "all": dd,
        "weekday": dd.loc[weekday_mask],
        "weekend": dd.loc[~weekday_mask],
    }


def split_returns_by_day_type(r: pd.Series) -> dict[str, pd.Series]:
    weekday_mask = r.index.dayofweek < 5
    return {
        "all": r,
        "weekday": r.loc[weekday_mask],
        "weekend": r.loc[~weekday_mask],
    }


def _fmt_ts(ts: pd.Timestamp) -> str:
    if ts.tzinfo is not None:
        return ts.isoformat()
    return pd.Timestamp(ts).isoformat()


def print_metric_block(
    title: str,
    daily: pd.DataFrame,
    returns: pd.Series,
    *,
    ewma_lambda: float,
) -> None:
    m = bar_metrics(daily)
    max_pct, max_usd, ts_pct, ts_usd, _, _ = max_move_series_and_dates(daily)
    sigma_hist, mu_pct, var_dec = historical_volatility_from_returns(returns)
    if len(returns) >= 2 and np.isfinite(var_dec):
        ewma_series = ewma_volatility_series_pct(
            returns, lam=ewma_lambda, initial_var=float(var_dec)
        )
        latest_ewma = float(ewma_series.iloc[-1])
        ewma_asof = ewma_series.index[-1]
    else:
        latest_ewma = float("nan")
        ewma_asof = None

    print(f"  --- {title} ---")
    print(f"      Daily bars: {len(daily)}  |  close-to-close returns: {len(returns)}")
    if len(returns) >= 2:
        print(f"      μ (mean return):                    {mu_pct:.6f}%  per day")
        print(f"      Variance (N−1, demeaned):           {var_dec:.8f}")
        print(f"      Historical σ = √var × 100:        {sigma_hist:.4f}%")
        print(
            f"      EWMA σ (λ={ewma_lambda}, σ₀²=hist var):  latest = {latest_ewma:.4f}%"
            + (f"  (as of {_fmt_ts(pd.Timestamp(ewma_asof))})" if ewma_asof is not None else "")
        )
    else:
        print("      (Not enough closes to compute log-return volatility.)")
    print(f"      Avg volatility % (open/close):       {m['avg_oc_pct']:.4f}%")
    print(f"      Avg volatility % (high/low):         {m['avg_hl_pct']:.4f}%")
    print(f"      Avg volatility USD (open/close):     ${m['avg_oc_usd']:,.2f}")
    print(f"      Avg volatility USD (high/low):       ${m['avg_hl_usd']:,.2f}")
    print(f"      Max % movement in a day:             {max_pct:.4f}%")
    if ts_pct is not None:
        print(f"        date: {_fmt_ts(ts_pct)}")
    print(f"      Max USD movement in a day:           ${max_usd:,.2f}")
    if ts_usd is not None:
        print(f"        date: {_fmt_ts(ts_usd)}")
    if ts_pct is not None and ts_usd is not None and ts_pct.date() != ts_usd.date():
        print("        (max % and max USD fall on different calendar days.)")
    print()


def plot_max_moves(
    slug: str,
    label: str,
    hl_pct: pd.Series,
    hl_usd: pd.Series,
    ts_pct: pd.Timestamp,
    ts_usd: pd.Timestamp,
    out_dir: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", slug).strip("_")
    out_path = out_dir / f"max_daily_range_{safe}.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{label} — daily range & max-move dates", fontsize=12)

    ax0 = axes[0]
    ax0.plot(hl_pct.index, hl_pct.values, color="steelblue", linewidth=1, label="(H−L)/open %")
    ax0.scatter([ts_pct], [hl_pct.loc[ts_pct]], color="crimson", s=80, zorder=5, label=f"Max %: {_fmt_ts(ts_pct)}")
    ax0.set_ylabel("Range %")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(hl_usd.index, hl_usd.values, color="darkgreen", linewidth=1, label="H−L (USD)")
    ax1.scatter([ts_usd], [hl_usd.loc[ts_usd]], color="darkorange", s=80, zorder=5, label=f"Max USD: {_fmt_ts(ts_usd)}")
    ax1.set_ylabel("Range USD")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_one(
    slug: str,
    label: str,
    path: Path,
    bar_kind: str,
    *,
    plot: bool,
    plot_dir: Path,
    ewma_lambda: float,
) -> None:
    if not path.is_file():
        print(f"\n=== {label} ===\nMissing file: {path}\n")
        return

    df = load_ohlc(path)
    if bar_kind == "hourly":
        daily = daily_ohlc_from_hourly(df)
        bar_note = (
            f"{len(df)} hourly rows → {len(daily)} daily bars; "
            "metrics 1–4 and close-based vol use daily OHLC"
        )
        n_raw = len(df)
    else:
        daily = df
        bar_note = f"{len(df)} daily bars"
        n_raw = len(df)

    r = daily_log_returns_from_close(daily)
    daily_splits = split_daily_by_day_type(daily)
    return_splits = split_returns_by_day_type(r)
    _, _, ts_pct, ts_usd, hl_pct_s, hl_usd_s = max_move_series_and_dates(daily)

    print(f"\n=== {label} ===")
    print(f"Rows (raw file): {n_raw}  |  {bar_note}\n")
    print("  Close-based vol uses r_t = ln(close_t/close_{t-1}).")
    print("  Weekday/weekend returns are classified by the ending close date.\n")
    print_metric_block(
        "All days",
        daily_splits["all"],
        return_splits["all"],
        ewma_lambda=ewma_lambda,
    )
    print_metric_block(
        "Weekdays (Mon-Fri)",
        daily_splits["weekday"],
        return_splits["weekday"],
        ewma_lambda=ewma_lambda,
    )
    print_metric_block(
        "Weekends (Sat-Sun)",
        daily_splits["weekend"],
        return_splits["weekend"],
        ewma_lambda=ewma_lambda,
    )

    if plot and ts_pct is not None and ts_usd is not None:
        p = plot_max_moves(slug, label, hl_pct_s, hl_usd_s, ts_pct, ts_usd, plot_dir)
        if p:
            print(f"\n  Plot saved: {p}")
        else:
            print("\n  (Install matplotlib to save plots: pip install matplotlib)")


def main() -> None:
    mpl_cfg = BACKTEST_DIR / ".mplconfig"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG charts",
    )
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=PLOT_DIR,
        help=f"Directory for PNG output (default: {PLOT_DIR})",
    )
    ap.add_argument(
        "--ewma-lambda",
        type=float,
        default=0.94,
        help="λ for EWMA variance update (default: 0.94)",
    )
    args = ap.parse_args()
    plot = not args.no_plots

    print(
        "BTC volatility from Perplexity CSVs\n"
        "Close-based vol: log returns → μ → (N−1) variance → σ; then EWMA with σ₀² = that variance.\n"
        "Metrics are shown for all days, weekdays, and weekends using daily bars.\n"
        f"Max-move charts: {'enabled' if plot else 'disabled'}."
    )
    for slug, label, path, kind in CSV_CONFIG:
        run_one(
            slug,
            label,
            path,
            kind,
            plot=plot,
            plot_dir=args.plot_dir,
            ewma_lambda=args.ewma_lambda,
        )


if __name__ == "__main__":
    main()
