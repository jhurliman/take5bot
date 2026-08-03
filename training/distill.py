#!/usr/bin/env python3
"""Distill a teacher checkpoint into a (different-architecture) student.

Rollouts come from the teacher playing all four seats of a VecGames pool
(sampling from its softmax for state coverage). The student trains on
soft cross-entropy to the teacher's legal-card distribution, MSE to the
teacher's value, and the env's exact belief targets. This measures
whether the student architecture can *represent* the teacher's policy,
separately from whether PPO can find it from scratch.

Example:
  .venv/bin/python training/distill.py \\
      --teacher training/runs/m8-v1/best.pt \\
      --arch attn --width 192 --blocks 4 \\
      --iters 600 --out training/runs/m11-distill
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py")
)

import take5_engine
from train_ppo import (
    BELIEF_CLASSES,
    NUM_CARDS,
    OBS_LEN,
    TURNS,
    build_net,
    eval_vs,
    masked_categorical,
)


def load_ckpt(path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = build_net(
        cfg.get("arch", "mlp"), cfg.get("width", 512), cfg.get("blocks", 2)
    ).to(device)
    net.load_state_dict(ckpt["model"], strict=False)
    net.eval()
    return net


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--arch", choices=["mlp", "attn"], default="attn")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--belief-coef", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="training/runs/distill")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.games < 1 or args.iters < 1:
        parser.error("--games and --iters must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    teacher = load_ckpt(args.teacher, device)
    student = build_net(args.arch, args.width, args.blocks).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    env = take5_engine.VecGames(args.games, [None] * 4, args.seed + 1)
    rows = args.games * 4
    best_pen = float("inf")

    def save(path: str, iteration: int) -> None:
        torch.save(
            {
                "model": student.state_dict(),
                "opt": opt.state_dict(),
                "iter": iteration,
                "best_pen": best_pen,
                "config": {
                    "arch": args.arch,
                    "width": args.width,
                    "blocks": args.blocks,
                    "belief_classes": BELIEF_CLASSES,
                },
            },
            path,
        )

    for it in range(args.iters):
        t0 = time.time()
        obs_buf = torch.empty(TURNS, rows, OBS_LEN, device=device)
        mask_buf = torch.empty(TURNS, rows, NUM_CARDS, device=device)
        tlogit_buf = torch.empty(TURNS, rows, NUM_CARDS, device=device)
        tval_buf = torch.empty(TURNS, rows, device=device)
        belief_buf = torch.empty(
            TURNS, rows, NUM_CARDS, dtype=torch.long, device=device
        )

        with torch.no_grad():
            for t in range(TURNS):
                obs, mask = env.observe()
                obs_t = torch.as_tensor(obs, device=device).view(rows, OBS_LEN)
                mask_t = torch.as_tensor(mask, device=device).view(rows, NUM_CARDS)
                logits, value, _ = teacher(obs_t)
                dist = masked_categorical(logits, mask_t)
                acts = dist.sample()
                obs_buf[t] = obs_t
                mask_buf[t] = mask_t
                tlogit_buf[t] = logits
                tval_buf[t] = value
                belief_buf[t] = torch.as_tensor(
                    env.belief_targets(), device=device
                ).view(rows, NUM_CARDS)
                _, dones, _, _, _ = env.step((acts.cpu().numpy() + 1).astype(np.int64))
            assert dones.all()

        flat_obs = obs_buf.view(-1, OBS_LEN)
        flat_mask = mask_buf.view(-1, NUM_CARDS)
        flat_tlogit = tlogit_buf.view(-1, NUM_CARDS)
        flat_tval = tval_buf.view(-1)
        flat_belief = belief_buf.view(-1, NUM_CARDS)
        total = flat_obs.shape[0]

        stats = {"policy": 0.0, "value": 0.0, "belief": 0.0}
        batches = 0
        for _ in range(args.epochs):
            perm = torch.randperm(total, device=device)
            for i in range(0, total, args.minibatch):
                idx = perm[i : i + args.minibatch]
                logits, value, belief = student(flat_obs[idx])
                # Finite fill, not -inf: exp underflows to exactly 0.0 on
                # illegal slots so the CE product stays finite (0 * -inf
                # would be NaN).
                neg = -1e9
                t_lp = F.log_softmax(
                    flat_tlogit[idx].masked_fill(flat_mask[idx] < 0.5, neg), dim=-1
                )
                s_lp = F.log_softmax(
                    logits.masked_fill(flat_mask[idx] < 0.5, neg), dim=-1
                )
                policy_loss = -(t_lp.exp() * s_lp).sum(-1).mean()
                value_loss = (value - flat_tval[idx]).pow(2).mean()
                belief_loss = F.cross_entropy(
                    belief.reshape(-1, BELIEF_CLASSES),
                    flat_belief[idx].reshape(-1),
                    ignore_index=-100,
                )
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    + args.belief_coef * belief_loss
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                opt.step()
                stats["policy"] += float(policy_loss)
                stats["value"] += float(value_loss)
                stats["belief"] += float(belief_loss)
                batches += 1
        for k in stats:
            stats[k] /= batches

        dt = time.time() - t0
        line = (
            f"iter {it:4d}  ce {stats['policy']:.4f}  v {stats['value']:6.2f}  "
            f"bel {stats['belief']:.4f}  {total / dt:,.0f} samp/s"
        )
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            greedy = eval_vs(
                student, ["greedy"] * 3, args.eval_games, 10_000 + it, device
            )
            line += (
                f"  | vs greedy {greedy['policy_pen']:.2f}/{greedy['opp_pen']:.2f} "
                f"win {greedy['win_rate']:.1%}"
            )
            if greedy["policy_pen"] < best_pen:
                best_pen = greedy["policy_pen"]
                save(os.path.join(args.out, "best.pt"), it)
                line += "  [best]"
            save(os.path.join(args.out, "last.pt"), it)
        print(line, flush=True)

    save(os.path.join(args.out, "last.pt"), args.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
