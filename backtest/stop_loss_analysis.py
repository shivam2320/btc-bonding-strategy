#!/usr/bin/env python3
"""
Stop-loss on the same entries as win_rate_thresholds.

Each row uses only that outcome leg's ``price`` path (YES or NO, any casing).
Entry = first tick with price >= threshold; fill = threshold; SL when
price <= entry * (1 - sl_pct/100) after entry.

Reports SL hits and whipsaws (SL hit but that leg's last price still >= 0.99).
Terminal output: YES + NO combined only, one table per entry threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_SL_PCTS,
    DEFAULT_THRESHOLDS,
    default_csv_path,
    first_tick_at_or_above,
    load_price_history,
    terminal_label,
)


def sl_hit(
    prices: pd.Series, entry_idx: int, entry_fill: float, sl_pct: float
) -> bool:
    floor_px = entry_fill * (1.0 - sl_pct / 100.0)
    tail = prices.iloc[entry_idx:]
    return bool((tail <= floor_px).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=default_csv_path())
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    df = load_price_history(args.csv)

    rows: list[dict] = []
    grouped = df.groupby(["event_slug", "market_slug", "outcome"], sort=False)
    for (event_slug, market_slug, outcome), g in grouped:
        prices = g["price"].reset_index(drop=True)
        last_price = float(prices.iloc[-1])
        term = terminal_label(last_price)
        if term is None:
            continue
        won = term == "win"
        idx_by_ts = {int(ts): i for i, ts in enumerate(g["timestamp"].values)}

        for thr in DEFAULT_THRESHOLDS:
            ent = first_tick_at_or_above(g, thr)
            if ent is None:
                continue
            _dt, entry_ts, _obs = ent
            if entry_ts not in idx_by_ts:
                continue
            entry_i = idx_by_ts[entry_ts]
            entry_fill = float(thr)

            for sl_pct in DEFAULT_SL_PCTS:
                hit = sl_hit(prices, entry_i, entry_fill, sl_pct)
                whipsaw = hit and won
                rows.append(
                    {
                        "event_slug": event_slug,
                        "market_slug": market_slug,
                        "outcome": outcome,
                        "threshold": thr,
                        "sl_pct": sl_pct,
                        "entry_timestamp": entry_ts,
                        "sl_hit": hit,
                        "whipsaw_win_after_sl": whipsaw,
                        "resolution_terminal": term,
                        "last_price": last_price,
                    }
                )

    detail = pd.DataFrame(rows)
    print(f"Trade × SL simulations: {len(detail)}\n")
    print(
        "Stop-loss (YES + NO combined): sl_hits = path touched SL before resolution; "
        "whipsaws = SL hit but that leg's last price still a win (>= 0.99).\n"
        "One table per entry threshold (same layout idea as IST tier tables).\n"
    )

    pooled = (
        detail.groupby(["threshold", "sl_pct"], as_index=False)
        .agg(
            n_trades=("sl_hit", "size"),
            sl_hits=("sl_hit", "sum"),
            whipsaws=("whipsaw_win_after_sl", "sum"),
        )
        .assign(
            sl_hit_rate=lambda x: x["sl_hits"] / x["n_trades"],
            whipsaw_rate_given_sl=lambda x: x["whipsaws"] / x["sl_hits"].replace(0, pd.NA),
        )
        .sort_values(["threshold", "sl_pct"])
    )

    for thr in DEFAULT_THRESHOLDS:
        sub = pooled[pooled["threshold"] == thr].drop(columns=["threshold"])
        if sub.empty:
            continue
        print(f"=== Entry threshold {thr} (YES + NO combined) ===\n")
        print(sub.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
