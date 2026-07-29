#!/usr/bin/env python3
"""Rules-parity check: the Rust engine must reproduce the legacy OpenSpiel
implementation move-for-move.

For each seed we let the legacy Python game deal, mirror that deal into the
Rust engine via Game.from_state, then drive both with the same random actions
and compare rows, penalty totals, phases, and terminal returns after every
action.

Run: .venv/bin/python tests/parity_check.py [--games 200]
Requires py/take5_engine.so (scripts/build_engine.sh) and the legacy deps
(pyspiel, numpy) on the import path.
"""

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
sys.path.insert(0, os.path.join(ROOT, "take5bot"))

import take5_engine  # noqa: E402
import openspiel_take5  # noqa: E402
import pyspiel  # noqa: E402


def check(cond: bool, msg: str, seed: int, step: int) -> None:
    if not cond:
        raise AssertionError(f"seed={seed} step={step}: {msg}")


def compare(py_state, rust, seed: int, step: int) -> None:
    py_rows = py_state._rows
    rust_rows = rust.rows()
    check(py_rows == rust_rows, f"rows differ: {py_rows} vs {rust_rows}", seed, step)

    py_pen = py_state._collect_bullheads()
    rust_pen = rust.penalties()
    check(
        py_pen == list(rust_pen), f"penalties differ: {py_pen} vs {rust_pen}", seed, step
    )

    py_hands = py_state._hands
    rust_hands = rust.hands()
    check(
        py_hands == rust_hands, f"hands differ: {py_hands} vs {rust_hands}", seed, step
    )


def run_one_game(seed: int) -> int:
    rng = random.Random(seed)
    game = openspiel_take5.TakeFiveGame()
    py_state = game.new_initial_state()

    num_players = len(py_state._hands)
    rust = take5_engine.Game.from_state(
        py_state._hands,
        [row[0] for row in py_state._rows],
        [0] * num_players,
        0,
    )

    steps = 0
    while not py_state.is_terminal():
        steps += 1
        current = py_state.current_player()
        if current == pyspiel.PlayerId.SIMULTANEOUS:
            check(rust.phase() == "select", f"phase: rust={rust.phase()}", seed, steps)
            cards = []
            for p in range(num_players):
                actions = py_state.legal_actions(p)
                check(
                    sorted(a + 1 for a in actions) == rust.legal_cards(p),
                    "legal cards differ",
                    seed,
                    steps,
                )
                cards.append(rng.choice(actions) + 1)
            py_state.apply_actions([c - 1 for c in cards])
            rust.play_cards(cards)
        else:
            check(
                rust.phase() == "choose_row",
                f"phase: py=choose_row rust={rust.phase()}",
                seed,
                steps,
            )
            ctx = rust.choose_row_context()
            check(ctx[0] == current, f"chooser: py={current} rust={ctx[0]}", seed, steps)
            check(
                ctx[1] == py_state._row_choice_needed_for,
                f"forced card: py={py_state._row_choice_needed_for} rust={ctx[1]}",
                seed,
                steps,
            )
            row = rng.randrange(4)
            py_state.apply_action(104 + row)
            rust.choose_row(row)

        compare(py_state, rust, seed, steps)

    check(rust.is_terminal(), "rust not terminal when python is", seed, steps)
    py_returns = py_state.returns()
    rust_returns = rust.returns()
    check(
        [float(r) for r in py_returns] == list(rust_returns),
        f"returns differ: {py_returns} vs {rust_returns}",
        seed,
        steps,
    )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    args = parser.parse_args()

    total_steps = 0
    for seed in range(args.games):
        total_steps += run_one_game(seed)
    print(f"PARITY OK: {args.games} games, {total_steps} actions compared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
