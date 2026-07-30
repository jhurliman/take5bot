#!/usr/bin/env python3
"""Elo ladder: round-robin every pair of bot specs and fit ratings.

Each pair plays head-to-head in 2v2 seating ([a, b, a, b], seats rotated by
the arena), a game's winner being the bot with the lower summed penalty
across its two seats (ties split). Ratings come from averaging pairwise
log-odds (Elo-style, 400/ln(10) scale), anchored so greedy = 1000 when
present (else the first spec).

Example:
  .venv/bin/python arena/ladder.py \\
      --bots random,lowest,greedy,mc:16,mc:64,neural:training/runs/m4-v1/net.t5n:32 \\
      --games 500
"""

import argparse
import itertools
import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py")
)

import take5_engine


def pair_win_rate(a: str, b: str, games: int, seed: int, threads: int) -> float:
    """Fraction of games bot `a` wins against `b` in 2v2 seating."""
    results = take5_engine.run_arena([a, b, a, b], games, seed, threads)
    wins = 0.0
    for seat_bots, penalties in results:
        totals = [0, 0]
        for seat, bot in enumerate(seat_bots):
            totals[bot % 2] += penalties[seat]
        if totals[0] < totals[1]:
            wins += 1.0
        elif totals[0] == totals[1]:
            wins += 0.5
    return wins / games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bots", default="random,lowest,greedy,mc:16,mc:64")
    parser.add_argument("--games", type=int, default=500, help="games per pairing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be >= 1")

    specs = args.bots.split(",")
    if len(specs) < 2:
        parser.error("need at least two bot specs")

    n = len(specs)
    win = [[0.5] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        w = pair_win_rate(specs[i], specs[j], args.games, args.seed, args.threads)
        win[i][j], win[j][i] = w, 1.0 - w
        print(f"{specs[i]:>40} vs {specs[j]:<40} {w:6.1%}", flush=True)

    # Elo from mean pairwise log-odds (clamped away from 0/1).
    scale = 400.0 / math.log(10.0)
    diffs = []
    for i in range(n):
        d = 0.0
        for j in range(n):
            if i != j:
                w = min(max(win[i][j], 0.01), 0.99)
                d += scale * math.log(w / (1.0 - w))
        diffs.append(d / (n - 1))

    anchor = specs.index("greedy") if "greedy" in specs else 0
    ratings = [1000.0 + d - diffs[anchor] for d in diffs]

    print(f"\n{'bot':<44} {'elo':>6}")
    print("-" * 52)
    for idx in sorted(range(n), key=lambda k: -ratings[k]):
        print(f"{specs[idx]:<44} {ratings[idx]:>6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
