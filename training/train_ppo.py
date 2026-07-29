#!/usr/bin/env python3
"""PPO self-play trainer for Take 5 (M3 of docs/ARCHITECTURE.md).

All four seats share one policy network and learn from per-seat relative
rewards (mean of others' bull deltas minus own), which sum to the seat's
final relative score over a deal. Deals are exactly 10 simultaneous turns,
so rollouts are rectangular: (turns, games * seats).

Examples:
  .venv/bin/python training/train_ppo.py --iters 500 --out training/runs/v1
  .venv/bin/python training/train_ppo.py --iters 2000 --games 2048 --wandb

Evaluate a checkpoint: training/eval_arena.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py")
)

import take5_engine

NUM_CARDS = 104
TURNS = 10
OBS_LEN = take5_engine.obs_len()


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(x + self.body(x)))


class PolicyNet(nn.Module):
    """Shared trunk with a masked 104-way card policy head and a value head."""

    def __init__(self, width: int = 512, blocks: int = 2):
        super().__init__()
        self.width = width
        self.blocks = blocks
        layers: list[nn.Module] = [nn.Linear(OBS_LEN, width), nn.ReLU()]
        layers += [Residual(width) for _ in range(blocks)]
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(width, NUM_CARDS)
        self.value = nn.Linear(width, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy(h), self.value(h).squeeze(-1)


def masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(
        logits=logits.masked_fill(mask < 0.5, float("-inf"))
    )


@torch.no_grad()
def eval_vs(
    net: PolicyNet,
    opponents: list[str],
    games: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Play `games` deals with the policy (argmax) in seat 0 vs bot seats.

    Returns mean penalties and the policy's win rate (ties split evenly).
    """
    net.eval()
    env = take5_engine.VecGames(games, [None] + opponents, seed)
    n = env.num_players()
    for _ in range(TURNS):
        obs, mask = env.observe()
        obs_t = torch.as_tensor(obs, device=device).view(-1, OBS_LEN)
        mask_t = torch.as_tensor(mask, device=device).view(-1, NUM_CARDS)
        logits, _ = net(obs_t)
        acts = (
            logits.masked_fill(mask_t < 0.5, float("-inf")).argmax(dim=-1).cpu().numpy()
        )
        _, dones, finals = env.step((acts + 1).astype(np.int64))
    assert dones.all(), "deals must finish in exactly TURNS steps"
    pens = finals.reshape(games, n)
    best = pens.min(axis=1, keepdims=True)
    winners = pens == best
    wins = (winners[:, 0] / winners.sum(axis=1)).sum()
    net.train()
    return {
        "policy_pen": float(pens[:, 0].mean()),
        "opp_pen": float(pens[:, 1:].mean()),
        "win_rate": float(wins / games),
    }


def collect_rollout(
    net: PolicyNet,
    env,
    num_games: int,
    seats: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = num_games * seats
    obs_buf = torch.empty(TURNS, rows, OBS_LEN, device=device)
    mask_buf = torch.empty(TURNS, rows, NUM_CARDS, device=device)
    act_buf = torch.empty(TURNS, rows, dtype=torch.long, device=device)
    logp_buf = torch.empty(TURNS, rows, device=device)
    val_buf = torch.empty(TURNS, rows, device=device)
    rew_buf = torch.empty(TURNS, rows, device=device)

    with torch.no_grad():
        for t in range(TURNS):
            obs, mask = env.observe()
            obs_t = torch.as_tensor(obs, device=device).view(rows, OBS_LEN)
            mask_t = torch.as_tensor(mask, device=device).view(rows, NUM_CARDS)
            logits, value = net(obs_t)
            dist = masked_categorical(logits, mask_t)
            action = dist.sample()
            rewards, dones, _ = env.step((action.cpu().numpy() + 1).astype(np.int64))

            obs_buf[t] = obs_t
            mask_buf[t] = mask_t
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value
            rew_buf[t] = torch.as_tensor(rewards, device=device)
        assert dones.all(), "deals must finish in exactly TURNS steps"

    return {
        "obs": obs_buf,
        "mask": mask_buf,
        "act": act_buf,
        "logp": logp_buf,
        "val": val_buf,
        "rew": rew_buf,
    }


def gae(
    rewards: torch.Tensor, values: torch.Tensor, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gamma=1 GAE over a rectangular rollout that terminates at the end."""
    turns = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(turns)):
        next_val = values[t + 1] if t + 1 < turns else torch.zeros_like(values[0])
        delta = rewards[t] + next_val - values[t]
        running = delta + lam * running
        adv[t] = running
    return adv, adv + values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1024, help="parallel deals")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.015)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--out", default="training/runs/latest")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # Opponent mixture: pure mirror self-play converges to an equilibrium
    # that transfers poorly to other styles, so half the pool trains against
    # engine bots (the poor-man's league; full league lands in M4).
    pool_specs: list[list[str | None]] = [
        [None, None, None, None],
        [None, None, None, None],
        [None, "greedy", "greedy", "greedy"],
        [None, None, "greedy", "greedy"],
        [None, "greedy", "random", "mc:8"],
    ]
    envs = []
    for i, spec in enumerate(pool_specs):
        n = max(args.games // len(pool_specs), 1)
        k = sum(1 for s in spec if s is None)
        envs.append((take5_engine.VecGames(n, spec, args.seed + i), n, k))

    net = PolicyNet(args.width, args.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_iter = 0
    best_pen = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_iter = ckpt.get("iter", 0)
        best_pen = ckpt.get("best_pen", float("inf"))
        print(f"resumed from {args.resume} at iter {start_iter}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="take5-ppo", config=vars(args))

    def save(path: str, iteration: int) -> None:
        torch.save(
            {
                "model": net.state_dict(),
                "opt": opt.state_dict(),
                "iter": iteration,
                "best_pen": best_pen,
                "config": {"width": args.width, "blocks": args.blocks},
            },
            path,
        )

    rows = sum(n * k for _, n, k in envs)
    for it in range(start_iter, args.iters):
        t0 = time.time()
        flats: dict[str, list[torch.Tensor]] = {}
        advs: list[torch.Tensor] = []
        rets: list[torch.Tensor] = []
        for env, n, k in envs:
            roll = collect_rollout(net, env, n, k, device)
            adv, ret = gae(roll["rew"], roll["val"], args.gae_lambda)
            for key, v in roll.items():
                flats.setdefault(key, []).append(v.reshape(TURNS * n * k, *v.shape[2:]))
            advs.append(adv.reshape(-1))
            rets.append(ret.reshape(-1))
        flat = {key: torch.cat(v) for key, v in flats.items()}
        adv_f = torch.cat(advs)
        ret_f = torch.cat(rets)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        batches = 0
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(TURNS * rows, device=device)
            for i in range(0, len(perm), args.minibatch):
                idx = perm[i : i + args.minibatch]
                logits, value = net(flat["obs"][idx])
                dist = masked_categorical(logits, flat["mask"][idx])
                logp = dist.log_prob(flat["act"][idx])
                ratio = (logp - flat["logp"][idx]).exp()
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                policy_loss = -torch.min(
                    ratio * adv_f[idx], clipped * adv_f[idx]
                ).mean()
                value_loss = (value - ret_f[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss + args.value_coef * value_loss - args.entropy * entropy
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy)
                    stats["kl"] += float((flat["logp"][idx] - logp).mean())
                batches += 1
        for k in stats:
            stats[k] /= batches

        dt = time.time() - t0
        line = (
            f"iter {it:4d}  pi {stats['policy_loss']:+.4f}  "
            f"v {stats['value_loss']:7.2f}  ent {stats['entropy']:.3f}  "
            f"kl {stats['kl']:+.4f}  {rows * TURNS / dt:,.0f} samp/s"
        )

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            greedy = eval_vs(net, ["greedy"] * 3, args.eval_games, 10_000 + it, device)
            mc = eval_vs(
                net, ["mc:16"] * 3, max(args.eval_games // 5, 100), 20_000 + it, device
            )
            line += (
                f"  | vs greedy {greedy['policy_pen']:.2f}/{greedy['opp_pen']:.2f} "
                f"win {greedy['win_rate']:.1%}"
                f"  | vs mc:16 {mc['policy_pen']:.2f}/{mc['opp_pen']:.2f} "
                f"win {mc['win_rate']:.1%}"
            )
            if run:
                run.log(
                    {
                        "iter": it,
                        **stats,
                        **{f"greedy/{k}": v for k, v in greedy.items()},
                        **{f"mc16/{k}": v for k, v in mc.items()},
                    }
                )
            if greedy["policy_pen"] < best_pen:
                best_pen = greedy["policy_pen"]
                save(os.path.join(args.out, "best.pt"), it)
                line += "  [best]"
            save(os.path.join(args.out, "last.pt"), it)
        elif run:
            run.log({"iter": it, **stats})

        print(line, flush=True)

    save(os.path.join(args.out, "last.pt"), args.iters)
    if run:
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
