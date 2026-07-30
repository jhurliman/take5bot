#!/usr/bin/env python3
"""Evaluate a trained PPO checkpoint against engine bots.

Example:
  .venv/bin/python training/eval_arena.py --ckpt training/runs/latest/best.pt \
      --opponents mc:64,mc:64,mc:64 --games 1000
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ppo import PolicyNet, eval_vs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--opponents", default="greedy,greedy,greedy")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be >= 1")

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    net = PolicyNet(cfg.get("width", 512), cfg.get("blocks", 2)).to(device)
    net.load_state_dict(ckpt["model"], strict=False)

    opponents = args.opponents.split(",")
    result = eval_vs(net, opponents, args.games, args.seed, device)
    print(
        f"policy vs {args.opponents} over {args.games} games (seed {args.seed}):\n"
        f"  policy mean penalty: {result['policy_pen']:.2f}\n"
        f"  opponent mean penalty: {result['opp_pen']:.2f}\n"
        f"  policy win rate: {result['win_rate']:.1%}"
        f"  ({1 / (len(opponents) + 1):.1%} = parity)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
