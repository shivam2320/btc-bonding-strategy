#!/usr/bin/env python3
"""
Win rate when buying an outcome token at high prices (YES+NO combined).

Outcome labels are normalized (yes/YES -> YES, no/NO -> NO). Each trade uses
only that leg's ``price`` history; YES and NO are not mixed for pricing, but
reported win rates pool all legs together.

Entry: first chronological tick where price >= threshold (no UTC hour buckets).
Fill assumed at ``threshold``.

Resolution for that leg: last ``price`` >= 0.99 => win, <= 0.01 => lose.
Ambiguous terminals are excluded from the win-rate denominator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_THRESHOLDS,
    default_csv_path,
    first_tick_at_or_above,
    load_price_history,
    terminal_label,
)


def build_trades(df: pd.DataFrame, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = df.groupby(["event_slug", "market_slug", "outcome"], sort=False)
    for (event_slug, market_slug, outcome), g in grouped:
        last_price = float(g["price"].iloc[-1])
        term = terminal_label(last_price)
        res_end = pd.Timestamp(g["datetime_utc"].iloc[-1])
        for thr in thresholds:
            ent = first_tick_at_or_above(g, thr)
            if ent is None:
                continue
            entry_dt, entry_ts, _observed = ent
            won = term == "win"
            rows.append(
                {
                    "event_slug": event_slug,
                    "market_slug": market_slug,
                    "outcome": outcome,
                    "threshold": thr,
                    "entry_datetime_utc": entry_dt,
                    "entry_timestamp": entry_ts,
                    "resolution_label": term,
                    "last_price": last_price,
                    "resolution_datetime_utc": res_end,
                    "won": won if term is not None else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=default_csv_path(),
        help="Path to btc_one_day_price_history.csv",
    )
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    df = load_price_history(args.csv)
    trades = build_trades(df, DEFAULT_THRESHOLDS)
    print(f"Simulated trades (all thresholds, incl. ambiguous): {len(trades)}\n")

    clean = trades.dropna(subset=["won"])
    summary = (
        clean.groupby("threshold", as_index=False)
        .agg(n_trades=("won", "size"), wins=("won", "sum"))
        .assign(win_rate=lambda x: x["wins"] / x["n_trades"])
    )
    print(
        "Win rate — YES + NO combined (each leg: first tick >= threshold, "
        "terminal from that leg's last price; ambiguous excluded)\n"
    )
    print(summary.to_string(index=False))

    ambiguous = trades[trades["resolution_label"].isna()]
    if len(ambiguous):
        cols = [
            "event_slug",
            "market_slug",
            "outcome",
            "threshold",
            "last_price",
            "resolution_datetime_utc",
        ]
        print(f"\nAmbiguous terminal (last price between lose/win bands): {len(ambiguous)}\n")
        print(ambiguous[cols].to_string(index=False))

    losses = clean[~clean["won"].astype(bool)]
    loss_cols = [
        "outcome",
        "threshold",
        "market_slug",
        "entry_datetime_utc",
        "resolution_datetime_utc",
        "last_price",
    ]
    print(f"\nLosses (clear resolution, lost): {len(losses)}\n")
    if len(losses):
        print(losses[loss_cols].to_string(index=False))


if __name__ == "__main__":
    main()
