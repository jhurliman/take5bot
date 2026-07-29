#!/usr/bin/env python3
"""Take 5 arena: pit bots against each other and report penalties and win
rates with confidence intervals.

Examples:
  .venv/bin/python arena/run_arena.py --bots greedy,random,random,random --games 2000
  .venv/bin/python arena/run_arena.py --bots mc:64,greedy,greedy,greedy --games 500

Bot specs: random | lowest | greedy | mc | mc:<worlds>
One spec per seat; seats rotate every game so position doesn't bias results.
Deterministic for a given (--seed, --games).
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))

import take5_engine  # noqa: E402


def mean_ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bots",
        default="greedy,random,random,random",
        help="comma-separated bot specs, one per seat",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    args = parser.parse_args()

    specs = args.bots.split(",")
    start = time.time()
    results = take5_engine.run_arena(specs, args.games, args.seed, args.threads)
    elapsed = time.time() - start

    per_bot_pens: list[list[float]] = [[] for _ in specs]
    per_bot_wins = [0.0 for _ in specs]
    for seat_bots, penalties in results:
        best = min(penalties)
        winners = [i for i, p in enumerate(penalties) if p == best]
        for seat, bot in enumerate(seat_bots):
            per_bot_pens[bot].append(float(penalties[seat]))
            if seat in winners:
                per_bot_wins[bot] += 1.0 / len(winners)

    print(
        f"\n{args.games} games, seats rotated, seed {args.seed} "
        f"({elapsed:.2f}s, {args.games / elapsed:,.0f} games/s)\n"
    )
    print(f"{'bot':<12} {'mean pen':>9} {'95% ci':>8} {'win rate':>9}")
    print("-" * 42)
    order = sorted(range(len(specs)), key=lambda b: sum(per_bot_pens[b]) / len(per_bot_pens[b]))
    for b in order:
        mean, ci = mean_ci95(per_bot_pens[b])
        win = per_bot_wins[b] / args.games
        print(f"{specs[b]:<12} {mean:>9.2f} {ci:>8.2f} {win:>8.1%}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
