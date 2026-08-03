#!/usr/bin/env python3
"""Torch-vs-Rust inference parity: export a randomly initialized PolicyNet,
load it through the engine, and compare outputs on real observations.

Run: .venv/bin/python tests/test_neural_parity.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
sys.path.insert(0, os.path.join(ROOT, "training"))

import take5_engine
from export_net import export
from train_ppo import AttnNet, PolicyNet

OBS_LEN = take5_engine.obs_len()


def main() -> int:
    torch.manual_seed(3)
    nets = [PolicyNet(width=64, blocks=2), AttnNet(d_model=64, layers=2)]
    for n in nets:
        n.eval()

    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        for arch, net in zip(("mlp", "attn"), nets):
            for dtype, atol in (("f32", 1e-3), ("f16", 5e-2)):
                path = os.path.join(tmp, f"net-{arch}-{dtype}.t5n")
                export(net, path, dtype)

                # Real observations from a few dealt games.
                for seed in range(5):
                    game = take5_engine.Game.deal(4, seed)
                    for player in range(4):
                        obs = np.asarray(game.observe(player), dtype=np.float32)
                        with torch.no_grad():
                            t_logits, t_value, t_belief = net(
                                torch.from_numpy(obs).unsqueeze(0)
                            )
                        r_logits, r_value, r_belief = take5_engine.debug_neural_eval(
                            path, obs.tolist()
                        )
                        ctx = f"({arch}, {dtype}, seed {seed} player {player})"
                        assert np.allclose(
                            t_logits.squeeze(0).numpy(), np.array(r_logits), atol=atol
                        ), f"policy logits diverge {ctx}"
                        assert (
                            abs(float(t_value) - r_value) < atol
                        ), f"value diverges {ctx}"
                        assert np.allclose(
                            t_belief.squeeze(0).numpy().reshape(-1),
                            np.array(r_belief),
                            atol=atol,
                        ), f"belief logits diverge {ctx}"
                        checks += 1

    print(f"NEURAL PARITY OK: torch and Rust agree on {checks} observations (mlp+attn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
