#!/usr/bin/env python3
"""Sanity checks for take5_engine.VecGames (vectorized self-play env).

Run: .venv/bin/python tests/test_vec_games.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))

import take5_engine

OBS_LEN = take5_engine.obs_len()
TURNS = 10
CARDS = 104


def random_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random legal card id per row of a (rows, 104) mask."""
    acts = np.empty(mask.shape[0], dtype=np.int64)
    for i, row in enumerate(mask):
        acts[i] = rng.choice(np.flatnonzero(row)) + 1
    return acts


def run_deal(env, games: int, seats: int, rng: np.random.Generator):
    reward_sum = np.zeros(games * seats, dtype=np.float64)
    for t in range(TURNS):
        obs, mask = env.observe()
        obs = obs.reshape(games * seats, OBS_LEN)
        mask = mask.reshape(games * seats, CARDS)
        assert np.array_equal(mask, obs[:, :CARDS]), "mask must equal hand bits"
        assert mask.sum(axis=1).min() == TURNS - t, "hand size mismatch"
        rewards, dones, finals = env.step(random_legal(mask, rng))
        reward_sum += rewards
        expected_done = 1 if t == TURNS - 1 else 0
        assert (dones == expected_done).all(), f"bad dones at turn {t}"
    return reward_sum, finals


def test_self_play() -> None:
    games, seats = 16, 4
    env = take5_engine.VecGames(games, [None] * seats, seed=7)
    assert env.policy_seats() == [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    reward_sum, finals = run_deal(env, games, seats, rng)

    # Relative rewards are zero-sum across the four policy seats of a game.
    per_game = reward_sum.reshape(games, seats).sum(axis=1)
    assert np.abs(per_game).max() < 1e-4, "rewards must be zero-sum per game"

    # Summed rewards equal final relative scores.
    pens = finals.reshape(games, seats)
    relative = pens.mean(axis=1, keepdims=True) * seats / (seats - 1) - pens * seats / (
        seats - 1
    )
    assert np.allclose(reward_sum.reshape(games, seats), relative, atol=1e-3)

    # Fresh deals after auto-reset: full hands again.
    _, mask = env.observe()
    assert mask.reshape(-1, CARDS).sum(axis=1).min() == TURNS


def test_determinism() -> None:
    games, seats = 8, 4
    outs = []
    for _ in range(2):
        env = take5_engine.VecGames(games, [None] * seats, seed=42)
        rng = np.random.default_rng(1)
        outs.append(run_deal(env, games, seats, rng))
    assert np.array_equal(outs[0][0], outs[1][0]), "rewards must be deterministic"
    assert np.array_equal(outs[0][1], outs[1][1]), "finals must be deterministic"


def test_bot_seats() -> None:
    games = 8
    env = take5_engine.VecGames(games, [None, "greedy", "greedy", "greedy"], seed=3)
    assert env.policy_seats() == [0]
    rng = np.random.default_rng(2)
    _, finals = run_deal(env, games, 1, rng)
    pens = finals.reshape(games, 4)
    assert (pens.sum(axis=1) > 0).all(), "someone always collects bulls"


def test_rejects_illegal() -> None:
    env = take5_engine.VecGames(2, [None] * 4, seed=0)
    try:
        env.step(np.zeros(8, dtype=np.int64))
    except ValueError:
        return
    raise AssertionError("expected ValueError for illegal card 0")


def main() -> int:
    test_self_play()
    test_determinism()
    test_bot_seats()
    test_rejects_illegal()
    print("VEC GAMES OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
