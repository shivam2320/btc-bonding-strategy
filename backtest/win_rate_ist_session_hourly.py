#!/usr/bin/env python3
"""
Win rate by price tier and IST clock hour (no fixed 21:30 session).

Uses each series' actual timestamps only: convert to IST, bucket by IST hour
(floor to hour). Per bucket and tier: if any tick's price lies in the tier band,
count one trade (first such tick = notional entry). Win/lose from that leg's
last price (same rules as other backtests).

Output: YES+NO combined only, one table per tier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    PRICE_TIER_BANDS,
    default_csv_path,
    ensure_utc,
    load_price_history,
    terminal_label,
)


def hour_label(h: int) -> str:
    return f"{h:02d}:00–{h:02d}:59 IST"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=default_csv_path())
    args = ap.parse_args()

    print("Price tiers (half-open intervals):")
    for name, lo, hi in PRICE_TIER_BANDS:
        print(f"  {name}: {lo} <= price < {hi}")
    print(
        "\nBuckets: each tick's time in IST, floored to the start of the clock hour "
        "(uses only data between first and last row of each market outcome series).\n"
        "Per tier: one trade per (market, outcome, IST hour-bucket) if any tick in "
        "that bucket lies in the tier band (entry = first such tick).\n"
        "Report: YES + NO combined, separate table per tier.\n"
    )

    print(f"Loading {args.csv} ...")
    df = load_price_history(args.csv)

    rows: list[dict] = []
    grouped = df.groupby(["event_slug", "market_slug", "outcome"], sort=False)
    for (event_slug, market_slug, outcome), g in grouped:
        last_price = float(g["price"].iloc[-1])
        term = terminal_label(last_price)
        won = term == "win" if term is not None else None

        g1 = g.reset_index(drop=True)
        utc = ensure_utc(g1["datetime_utc"])
        dt_ist = utc.dt.tz_convert("Asia/Kolkata")
        hour_floor_ist = dt_ist.dt.floor("h")
        g2 = pd.concat(
            [g1, pd.DataFrame({"dt_ist": dt_ist, "hour_floor_ist": hour_floor_ist})],
            axis=1,
        )
        g2 = g2.loc[g2["dt_ist"].notna()]
        if g2.empty:
            continue

        keyed = g2.groupby("hour_floor_ist", sort=False)
        for hour_floor, part in keyed:
            h = int(hour_floor.hour)
            for tier_name, lo, hi in PRICE_TIER_BANDS:
                hit = part[(part["price"] >= lo) & (part["price"] < hi)]
                if hit.empty:
                    continue
                row0 = hit.iloc[0]
                rows.append(
                    {
                        "event_slug": event_slug,
                        "market_slug": market_slug,
                        "outcome": outcome,
                        "tier": tier_name,
                        "ist_hour": h,
                        "ist_hour_label": hour_label(h),
                        "entry_dt_ist": row0["dt_ist"],
                        "entry_price": float(row0["price"]),
                        "resolution_label": term,
                        "won": won,
                    }
                )

    trades = pd.DataFrame(rows)
    print(f"Trades (any tick in an IST hour-bucket hit tier band): {len(trades)}\n")

    clean = trades.dropna(subset=["won"])
    if clean.empty:
        print("No trades with clear terminal resolution.")
        return

    pooled = (
        clean.groupby(["tier", "ist_hour", "ist_hour_label"], as_index=False)
        .agg(n_trades=("won", "size"), wins=("won", "sum"))
        .assign(win_rate=lambda x: x["wins"] / x["n_trades"])
    )

    tier_order = [name for name, _, _ in PRICE_TIER_BANDS]
    for tier_name in tier_order:
        sub = pooled[pooled["tier"] == tier_name].sort_values("ist_hour")
        if sub.empty:
            continue
        show = sub.drop(columns=["tier"])
        print(f"=== {tier_name} (YES + NO combined) ===\n")
        print(show.to_string(index=False))
        print()

    amb = len(trades) - len(clean)
    if amb:
        print(f"Excluded ambiguous terminal: {amb} trades")


if __name__ == "__main__":
    main()
