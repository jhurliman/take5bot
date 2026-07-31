#!/usr/bin/env python3
"""Legacy bridge: play the original v1 MuZero checkpoint against v2 engine
bots, for a before/after strength comparison.

The legacy net (LightZero MuZero, 253-dim observation, policy-head argmax —
exactly how v1's play_take5.py used it) drives seat 0 of a VecGames pool;
its observation is reconstructed from the v2 observation, and forced row
choices use the cheapest-row heuristic (v1 never learned that decision).

Example:
  .venv/bin/python arena/legacy_bridge.py --games 1000 \\
      --opponents greedy,greedy,greedy
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))

import take5_engine

TURNS = 10
CARDS = 104
LEGACY_OBS = 253
V2_OBS = take5_engine.obs_len()
ROW_SLOTS_OFF = 2 * CARDS  # v2 layout: per-slot normalized row card ids
PENALTY_OFF = ROW_SLOTS_OFF + 20 + 12  # own penalty /66 comes first


def bullheads(card: int) -> int:
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


BULLS = np.array([0] + [bullheads(c) for c in range(1, CARDS + 1)], dtype=np.float32)


def legacy_obs_from_v2(v2: np.ndarray) -> np.ndarray:
    """Rebuild the v1 253-dim observation batch from v2 observations."""
    n = v2.shape[0]
    out = np.zeros((n, LEGACY_OBS), dtype=np.float32)
    hand = v2[:, :CARDS]
    out[:, :CARDS] = hand  # presence
    out[:, CARDS : 2 * CARDS] = hand * (BULLS[1:] / 7.0)  # hand penalties

    slots = np.rint(v2[:, ROW_SLOTS_OFF : ROW_SLOTS_OFF + 20] * CARDS).astype(int)
    out[:, 208:228] = slots / CARDS  # row card numbers
    slot_bulls = BULLS[slots] / 7.0
    slot_bulls[slots == 0] = 0.0
    out[:, 228:248] = slot_bulls  # row card penalties
    row_totals = (slot_bulls * 7.0).reshape(n, 4, 5).sum(axis=2) / 35.0
    out[:, 248:252] = row_totals  # row penalty totals
    out[:, 252] = v2[:, PENALTY_OFF]  # own penalty /66 (same normalization)
    return out


def load_legacy_policy(ckpt_path: str):
    from easydict import EasyDict
    from lzero.policy.muzero import MuZeroPolicy

    config = MuZeroPolicy.default_config()
    config["model"].update(
        {
            "observation_shape": LEGACY_OBS,
            "action_space_size": 108,
            "model_type": "mlp",
            "lstm_hidden_size": 256,
            "latent_state_dim": 256,
            "discrete_action_encoding_type": "one_hot",
            "norm_type": "BN",
        }
    )
    config.update(
        {
            "cuda": False,
            "env_type": "not_board_games",
            "use_wandb": False,
            "device": "cpu",
            "action_type": "varied_action_space",
        }
    )
    policy = MuZeroPolicy(EasyDict(config))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy._model.load_state_dict(checkpoint["model"])
    policy._model.eval()
    return policy._model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="v1 checkpoint (default: newest data_muzero/**/ckpt_best.pth.tar)",
    )
    parser.add_argument("--opponents", default="greedy,greedy,greedy")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be >= 1")

    ckpt = args.ckpt
    if not ckpt:
        pattern = os.path.join(ROOT, "data_muzero", "**", "ckpt_best.pth.tar")
        found = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
        if not found:
            print(f"no legacy checkpoint found under {pattern}", file=sys.stderr)
            return 1
        ckpt = found[-1]
    print(f"legacy checkpoint: {ckpt}")

    model = load_legacy_policy(ckpt)
    opponents = args.opponents.split(",")
    env = take5_engine.VecGames(args.games, [None] + opponents, args.seed)
    n = env.num_players()

    with torch.no_grad():
        for _ in range(TURNS):
            v2_obs, mask = env.observe()
            v2_obs = v2_obs.reshape(args.games, V2_OBS)
            mask = mask.reshape(args.games, CARDS)
            legacy = torch.from_numpy(legacy_obs_from_v2(v2_obs))
            result = model.initial_inference(legacy)
            logits = result.policy_logits[:, :CARDS].numpy()
            logits[mask < 0.5] = -np.inf
            acts = logits.argmax(axis=1) + 1
            _, dones, finals, _, _ = env.step(acts.astype(np.int64))
    assert dones.all()

    pens = finals.reshape(args.games, n)
    best = pens.min(axis=1, keepdims=True)
    winners = pens == best
    wins = float((winners[:, 0] / winners.sum(axis=1)).sum())
    print(
        f"legacy MuZero vs {args.opponents} over {args.games} games:\n"
        f"  legacy mean penalty: {pens[:, 0].mean():.2f}\n"
        f"  opponent mean penalty: {pens[:, 1:].mean():.2f}\n"
        f"  legacy win rate: {wins / args.games:.1%}"
        f"  ({1 / (len(opponents) + 1):.1%} = parity)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
