#!/usr/bin/env python3
"""PPO league self-play trainer for Take 5 (M3+M4 of docs/ARCHITECTURE.md).

Seats share one learner network. Rollouts come from a mixed pool:
pure self-play, games against engine bots (greedy/random/mc), and league
games against frozen snapshots of the learner's past selves. Per-seat
rewards are relative bull deltas (mean of others' minus own), which sum to
the seat's final relative score over a deal.

The network also carries a belief head trained (auxiliary cross-entropy)
to predict, for every card the seat cannot see, whether each opponent
holds it or it sits in the undealt stock. Targets come from
`VecGames.belief_targets()` — training-only supervision that never feeds
the observation path. The belief head powers M5's determinized search.

Examples:
  .venv/bin/python training/train_ppo.py --iters 1500 --out training/runs/m4
  .venv/bin/python training/train_ppo.py --init-from training/runs/m3/best.pt

Evaluate a checkpoint: training/eval_arena.py.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py")
)

import take5_engine

NUM_CARDS = 104
TURNS = 10
SEATS = 4
BELIEF_CLASSES = SEATS  # 3 opponents (relative seat order) + undealt stock
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
    """Trunk with masked card policy, value, and opponent-belief heads."""

    def __init__(self, width: int = 512, blocks: int = 2):
        super().__init__()
        self.width = width
        self.blocks = blocks
        layers: list[nn.Module] = [nn.Linear(OBS_LEN, width), nn.ReLU()]
        layers += [Residual(width) for _ in range(blocks)]
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(width, NUM_CARDS)
        self.value = nn.Linear(width, 1)
        self.belief = nn.Linear(width, NUM_CARDS * BELIEF_CLASSES)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        belief = self.belief(h).view(-1, NUM_CARDS, BELIEF_CLASSES)
        return self.policy(h), self.value(h).squeeze(-1), belief


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


class AttnNet(nn.Module):
    """Card-token transformer encoder (out-of-class probe vs the MLP).

    One token per card carrying [in-hand, played, bulls, value, on-table,
    row, slot, is-row-tail] plus a learned card embedding, and one CLS
    token fed the non-card observation tail (row summaries, penalties,
    seats, scalars, standings). Policy logits read per-card tokens (a
    natural fit for the 104-card action space), value reads CLS, belief
    reads per-card tokens.
    """

    CARD_FEATS = 8
    GLOBAL_OFF = 2 * NUM_CARDS  # everything past the two card masks

    def __init__(self, d_model: int = 192, layers: int = 4):
        super().__init__()
        self.width = d_model
        self.blocks = layers
        self.card_emb = nn.Embedding(NUM_CARDS, d_model)
        self.feat = nn.Linear(self.CARD_FEATS, d_model)
        self.glob = nn.Linear(OBS_LEN - self.GLOBAL_OFF, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=max(d_model // 32, 1),
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        # Pre-LN encoders need a final norm (PyTorch omits it by default).
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.policy = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, 1)
        self.belief = nn.Linear(d_model, BELIEF_CLASSES)
        self.register_buffer(
            "bulls",
            torch.tensor([bullheads(c) for c in range(1, NUM_CARDS + 1)]) / 7.0,
        )
        self.register_buffer(
            "card_val", torch.arange(1, NUM_CARDS + 1, dtype=torch.float32) / NUM_CARDS
        )
        slot = torch.arange(20)
        self.register_buffer("slot_row", (slot // 5).float() / 3.0)
        self.register_buffer("slot_pos", (slot % 5).float() / 4.0)
        self.register_buffer("slot_row_idx", slot // 5)

    def _card_features(self, obs: torch.Tensor) -> torch.Tensor:
        # Sync-free on purpose: no data-dependent boolean indexing, only
        # scatter/gather (empty slots scatter into a discarded 0th row).
        b_sz = obs.shape[0]
        f = torch.zeros(b_sz, NUM_CARDS, self.CARD_FEATS, device=obs.device)
        f[:, :, 0] = obs[:, :NUM_CARDS]
        f[:, :, 1] = obs[:, NUM_CARDS : 2 * NUM_CARDS]
        f[:, :, 2] = self.bulls
        f[:, :, 3] = self.card_val
        # Table membership from the 4x5 row slots (normalized card ids).
        ids = torch.round(obs[:, self.GLOBAL_OFF : self.GLOBAL_OFF + 20] * NUM_CARDS)
        ids = ids.long()
        valid = (ids > 0).float()
        row_len = valid.view(b_sz, 4, 5).sum(-1)
        len_at_slot = row_len.gather(1, self.slot_row_idx.expand(b_sz, -1))
        tail = (self.slot_pos * 4.0 + 1.0 == len_at_slot).float() * valid
        vals = torch.stack(
            [
                valid,
                self.slot_row.expand_as(valid),
                self.slot_pos.expand_as(valid),
                tail,
            ],
            dim=-1,
        )
        scat = torch.zeros(b_sz, NUM_CARDS + 1, 4, device=obs.device)
        scat.scatter_(1, ids.unsqueeze(-1).expand(-1, -1, 4), vals)
        f[:, :, 4:] = scat[:, 1:]
        return f

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # bf16 autocast: fp32 attention activations at PPO batch sizes are
        # multi-GB (105 tokens/sample); heads are cast back for loss math.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=obs.is_cuda):
            tok = self.feat(self._card_features(obs)) + self.card_emb.weight
            cls = self.cls + self.glob(obs[:, self.GLOBAL_OFF :]).unsqueeze(1)
            h = self.encoder(torch.cat([cls, tok], dim=1))
            logits = self.policy(h[:, 1:]).squeeze(-1)
            value = self.value(h[:, 0]).squeeze(-1)
            belief = self.belief(h[:, 1:])
        return logits.float(), value.float(), belief.float()


def build_net(arch: str, width: int, blocks: int) -> nn.Module:
    if arch == "attn":
        return AttnNet(width, blocks)
    return PolicyNet(width, blocks)


def masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(
        logits=logits.masked_fill(mask < 0.5, float("-inf"))
    )


@torch.no_grad()
def eval_vs(
    net: nn.Module,
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
        logits, _, _ = net(obs_t)
        acts = (
            logits.masked_fill(mask_t < 0.5, float("-inf")).argmax(dim=-1).cpu().numpy()
        )
        _, dones, finals, _, _ = env.step((acts + 1).astype(np.int64))
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


@torch.no_grad()
def eval_match(
    net: nn.Module,
    opponents: list[str],
    matches: int,
    seed: int,
    device: torch.device,
    match_to: int = 66,
) -> dict[str, float]:
    """Play full matches to `match_to` (policy argmax in seat 0). Returns
    the policy's match win rate (ties split) and mean final totals."""
    net.eval()
    env = take5_engine.VecGames(matches, [None] + opponents, seed, match_to)
    n = env.num_players()
    won = 0.0
    finished = np.zeros(matches, dtype=bool)
    pol_total = 0.0
    opp_total = 0.0
    while not finished.all():
        obs, mask = env.observe()
        obs_t = torch.as_tensor(obs, device=device).view(-1, OBS_LEN)
        mask_t = torch.as_tensor(mask, device=device).view(-1, NUM_CARDS)
        logits, _, _ = net(obs_t)
        acts = (
            logits.masked_fill(mask_t < 0.5, float("-inf")).argmax(dim=-1).cpu().numpy()
        )
        _, _, _, match_dones, match_finals = env.step((acts + 1).astype(np.int64))
        for g in np.flatnonzero(match_dones):
            if finished[g]:
                continue
            finished[g] = True
            totals = match_finals[g * n : (g + 1) * n]
            best = totals.min()
            winners = (totals == best).sum()
            if totals[0] == best:
                won += 1.0 / winners
            pol_total += float(totals[0])
            opp_total += float(totals[1:].mean())
    net.train()
    return {
        "match_win": float(won / matches),
        "policy_total": pol_total / matches,
        "opp_total": opp_total / matches,
    }


class EnvSlot:
    """One rollout env plus who drives each policy seat.

    `driver_slots[j]` indexes into the nets list passed to `collect`:
    slot 0 is the learner (trainable); higher slots are frozen league nets.
    """

    def __init__(
        self,
        num_games: int,
        specs: list[str | None],
        driver_slots: list[int],
        seed: int,
        device: torch.device,
        match_to: int = 0,
    ):
        self.env = take5_engine.VecGames(num_games, specs, seed, match_to)
        self.num_games = num_games
        self.match_to = match_to
        self.driver_slots = driver_slots
        k = len(driver_slots)
        assert k == len(self.env.policy_seats())
        self.rows = num_games * k
        # Constant flat row indices (game-major) per driver slot.
        self.driver_rows = {
            slot: torch.tensor(
                [
                    g * k + j
                    for g in range(num_games)
                    for j, s in enumerate(driver_slots)
                    if s == slot
                ],
                dtype=torch.long,
                device=device,
            )
            for slot in sorted(set(driver_slots))
        }
        self.train_rows = self.driver_rows[0]


@torch.no_grad()
def collect(
    slot: EnvSlot, nets: list[nn.Module], device: torch.device
) -> dict[str, torch.Tensor]:
    """Roll one full deal through `slot`; returns trainable-row buffers."""
    rows = slot.rows
    t_rows = len(slot.train_rows)
    buf = {
        "obs": torch.empty(TURNS, t_rows, OBS_LEN, device=device),
        "mask": torch.empty(TURNS, t_rows, NUM_CARDS, device=device),
        "act": torch.empty(TURNS, t_rows, dtype=torch.long, device=device),
        "logp": torch.empty(TURNS, t_rows, device=device),
        "val": torch.empty(TURNS, t_rows, device=device),
        "rew": torch.empty(TURNS, t_rows, device=device),
        "belief": torch.empty(
            TURNS, t_rows, NUM_CARDS, dtype=torch.long, device=device
        ),
    }
    for t in range(TURNS):
        obs, mask = slot.env.observe()
        obs_t = torch.as_tensor(obs, device=device).view(rows, OBS_LEN)
        mask_t = torch.as_tensor(mask, device=device).view(rows, NUM_CARDS)
        belief_t = torch.as_tensor(slot.env.belief_targets(), device=device).view(
            rows, NUM_CARDS
        )

        actions = torch.empty(rows, dtype=torch.long, device=device)
        for s, rid in slot.driver_rows.items():
            logits, value, _ = nets[s](obs_t[rid])
            dist = masked_categorical(logits, mask_t[rid])
            # Slot 3 is a frozen champion under exploitability probing:
            # play argmax, exactly like the deployed raw policy.
            act = dist.probs.argmax(dim=-1) if s >= 3 else dist.sample()
            actions[rid] = act
            if s == 0:
                buf["obs"][t] = obs_t[rid]
                buf["mask"][t] = mask_t[rid]
                buf["act"][t] = act
                buf["logp"][t] = dist.log_prob(act)
                buf["val"][t] = value
                buf["belief"][t] = belief_t[rid]

        rewards, dones, _, match_dones, _ = slot.env.step(
            (actions.cpu().numpy() + 1).astype(np.int64)
        )
        buf["rew"][t] = torch.as_tensor(rewards, device=device)[slot.train_rows]
    assert dones.all(), "deals must finish in exactly TURNS steps"

    if slot.match_to > 0:
        # A deal boundary is not match-terminal: the env has already dealt
        # the next deal with standings carried, so bootstrap its value,
        # zeroed only where the match actually ended.
        obs, _ = slot.env.observe()
        obs_t = torch.as_tensor(obs, device=device).view(rows, OBS_LEN)
        _, boot, _ = nets[0](obs_t[slot.train_rows])
        k = len(slot.driver_slots)
        cont = 1.0 - torch.as_tensor(
            match_dones.astype(np.float32), device=device
        ).repeat_interleave(k)
        buf["boot"] = boot * cont[slot.train_rows]
    else:
        buf["boot"] = torch.zeros(t_rows, device=device)
    return buf


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    lam: float,
    boot: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gamma=1 GAE over a rectangular rollout.

    `boot` is the value after the final turn — zero for terminal deals,
    the fresh-deal value for nonterminal match-mode deal boundaries.
    """
    turns = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    last = boot if boot is not None else torch.zeros_like(values[0])
    for t in reversed(range(turns)):
        next_val = values[t + 1] if t + 1 < turns else last
        delta = rewards[t] + next_val - values[t]
        running = delta + lam * running
        adv[t] = running
    return adv, adv + values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1536, help="deals across the pool")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.015)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--belief-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument(
        "--arch",
        choices=["mlp", "attn"],
        default="mlp",
        help="mlp = residual MLP (exportable to the Rust/WASM engine); "
        "attn = card-token transformer (torch-only experiment; --width is "
        "d_model, try 192; --blocks is encoder layers, try 4)",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--league-size", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--out", default="training/runs/latest")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--init-from", default=None, help="warm-start weights (non-strict load)"
    )
    parser.add_argument(
        "--opponent-net",
        default=None,
        help="exported .t5n weights: adds pool slots where seats are driven "
        "by this frozen champion (the search-league anchor)",
    )
    parser.add_argument(
        "--match-to",
        type=int,
        default=0,
        help="train in match mode: deals accumulate to this bull total "
        "(66 = real rules), with a zero-sum win bonus at match end",
    )
    parser.add_argument(
        "--opponent-ckpt",
        default=None,
        help="torch checkpoint (any arch) driven as frozen opponent seats "
        "on the GPU — much faster than engine-bot opponents when the "
        "champion is an attention net; with --exploit it replaces the "
        "engine-bot champion seats",
    )
    parser.add_argument(
        "--exploit",
        action="store_true",
        help="train a pure best-response to --opponent-net (exploitability probe)",
    )
    parser.add_argument(
        "--opponent-worlds",
        type=int,
        default=0,
        help="search worlds for champion opponents (0 = raw policy, cheap; "
        ">0 adds real search pressure but costs rollout time)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # Rollout pool: self-play, bot anchors (style transfer), and league games
    # against frozen snapshots (driver slots 1 and 2).
    n = max(args.games // 6, 1)
    pool = [
        EnvSlot(n, [None] * 4, [0, 0, 0, 0], args.seed + 1, device, args.match_to),
        EnvSlot(n, [None] * 4, [0, 0, 0, 0], args.seed + 2, device, args.match_to),
        EnvSlot(
            n,
            [None, "greedy", "greedy", "greedy"],
            [0],
            args.seed + 3,
            device,
            args.match_to,
        ),
        EnvSlot(
            n,
            [None, "greedy", "random", "mc:8"],
            [0],
            args.seed + 4,
            device,
            args.match_to,
        ),
        EnvSlot(n, [None] * 4, [0, 0, 1, 1], args.seed + 5, device, args.match_to),
        EnvSlot(n, [None] * 4, [0, 2, 2, 2], args.seed + 6, device, args.match_to),
    ]
    if args.opponent_ckpt and args.exploit:
        # Champion seats driven by the frozen torch net (driver slot 3):
        # batched on the GPU, argmax like the deployed policy.
        pool = [
            EnvSlot(
                n,
                [None] * 4,
                [0, 3, 3, 3],
                args.seed + 7 + i,
                device,
                args.match_to,
            )
            for i in range(3)
        ]
    elif args.opponent_net:
        champ = f"neural:{args.opponent_net}:{args.opponent_worlds}"
        if args.exploit:
            # Best-response mode: a single learner seat, every opponent the
            # frozen champion — never two learner seats in one game, which
            # would dilute the best response and understate exploitability.
            # The learner's win rate above parity measures exploitability.
            pool = [
                EnvSlot(
                    n,
                    [None, champ, champ, champ],
                    [0],
                    args.seed + 7 + i,
                    device,
                    args.match_to,
                )
                for i in range(3)
            ]
        else:
            pool.append(
                EnvSlot(
                    n,
                    [None, champ, champ, champ],
                    [0],
                    args.seed + 7,
                    device,
                    args.match_to,
                )
            )
            pool.append(
                EnvSlot(
                    n,
                    [None, None, champ, "greedy"],
                    [0, 0],
                    args.seed + 8,
                    device,
                    args.match_to,
                )
            )

    learner = build_net(args.arch, args.width, args.blocks).to(device)
    frozen = [
        build_net(args.arch, args.width, args.blocks).to(device).requires_grad_(False)
        for _ in range(2)
    ]
    nets = [learner, *frozen]
    if args.opponent_ckpt:
        ckpt = torch.load(args.opponent_ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        champ_net = build_net(
            cfg.get("arch", "mlp"), cfg.get("width", 512), cfg.get("blocks", 2)
        ).to(device)
        champ_net.load_state_dict(ckpt["model"], strict=False)
        champ_net.eval().requires_grad_(False)
        nets.append(champ_net)
        print(f"champion seats driven by {args.opponent_ckpt} ({cfg.get('arch')})")
    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
    league: list[dict] = []
    start_iter = 0
    best_pen = float("inf")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        state = ckpt["model"]
        stem_w = state.get("trunk.0.weight")
        if stem_w is not None and stem_w.shape[1] < OBS_LEN:
            # Observation schema grew (append-only): pad new input columns
            # with zeros so the net starts as an exact function of the old
            # features and learns the new ones from there.
            pad = torch.zeros(
                stem_w.shape[0],
                OBS_LEN - stem_w.shape[1],
                dtype=stem_w.dtype,
                device=stem_w.device,
            )
            state["trunk.0.weight"] = torch.cat([stem_w, pad], dim=1)
            print(f"padded stem {stem_w.shape[1]} -> {OBS_LEN} inputs")
        missing, _unexpected = learner.load_state_dict(state, strict=False)
        print(f"warm start from {args.init_from} (missing={missing})")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        learner.load_state_dict(ckpt["model"])
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
                "model": learner.state_dict(),
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

    def snapshot() -> dict:
        return {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()}

    for it in range(start_iter, args.iters):
        t0 = time.time()

        # Refresh league opponents. Before the first snapshot exists the
        # frozen nets mirror the learner, i.e. plain self-play.
        for f in frozen:
            f.load_state_dict(random.choice(league) if league else learner.state_dict())

        flats: dict[str, list[torch.Tensor]] = {}
        advs: list[torch.Tensor] = []
        rets: list[torch.Tensor] = []
        for slot in pool:
            roll = collect(slot, nets, device)
            boot = roll.pop("boot")
            adv, ret = gae(roll["rew"], roll["val"], args.gae_lambda, boot)
            t_rows = roll["rew"].shape[1]
            for key, v in roll.items():
                flats.setdefault(key, []).append(
                    v.reshape(TURNS * t_rows, *v.shape[2:])
                )
            advs.append(adv.reshape(-1))
            rets.append(ret.reshape(-1))
        flat = {key: torch.cat(v) for key, v in flats.items()}
        adv_f = torch.cat(advs)
        ret_f = torch.cat(rets)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        total_rows = adv_f.shape[0]

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "belief_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
        }
        batches = 0
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(total_rows, device=device)
            for i in range(0, total_rows, args.minibatch):
                idx = perm[i : i + args.minibatch]
                logits, value, belief = learner(flat["obs"][idx])
                dist = masked_categorical(logits, flat["mask"][idx])
                logp = dist.log_prob(flat["act"][idx])
                ratio = (logp - flat["logp"][idx]).exp()
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                policy_loss = -torch.min(
                    ratio * adv_f[idx], clipped * adv_f[idx]
                ).mean()
                value_loss = (value - ret_f[idx]).pow(2).mean()
                belief_loss = F.cross_entropy(
                    belief.reshape(-1, BELIEF_CLASSES),
                    flat["belief"][idx].reshape(-1),
                    ignore_index=-100,
                )
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    + args.belief_coef * belief_loss
                    - args.entropy * entropy
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner.parameters(), 0.5)
                opt.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["belief_loss"] += float(belief_loss)
                stats["entropy"] += float(entropy)
                stats["kl"] += float((flat["logp"][idx] - logp).mean())
                batches += 1
        for k in stats:
            stats[k] /= batches

        if (it + 1) % args.snapshot_every == 0:
            if len(league) >= args.league_size:
                league[random.randrange(len(league))] = snapshot()
            else:
                league.append(snapshot())

        dt = time.time() - t0
        line = (
            f"iter {it:4d}  pi {stats['policy_loss']:+.4f}  "
            f"v {stats['value_loss']:6.2f}  bel {stats['belief_loss']:.4f}  "
            f"ent {stats['entropy']:.3f}  kl {stats['kl']:+.4f}  "
            f"league {len(league)}  {total_rows / dt:,.0f} samp/s"
        )

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            greedy = eval_vs(
                learner, ["greedy"] * 3, args.eval_games, 10_000 + it, device
            )
            mc = eval_vs(
                learner,
                ["mc:16"] * 3,
                max(args.eval_games // 5, 100),
                20_000 + it,
                device,
            )
            line += (
                f"  | vs greedy {greedy['policy_pen']:.2f}/{greedy['opp_pen']:.2f} "
                f"win {greedy['win_rate']:.1%}"
                f"  | vs mc:16 {mc['policy_pen']:.2f}/{mc['opp_pen']:.2f} "
                f"win {mc['win_rate']:.1%}"
            )
            match_stats: dict[str, float] = {}
            if args.match_to:
                # A standings-aware policy may trade deal score for match
                # wins, so match mode gates best.pt on the match metric.
                match_opps = (
                    [f"neural:{args.opponent_net}:0"] * 3
                    if args.opponent_net
                    else ["greedy"] * 3
                )
                match_stats = eval_match(
                    learner,
                    match_opps,
                    max(args.eval_games // 4, 200),
                    30_000 + it,
                    device,
                    args.match_to,
                )
                line += (
                    f"  | match win {match_stats['match_win']:.1%} "
                    f"tot {match_stats['policy_total']:.1f}"
                    f"/{match_stats['opp_total']:.1f}"
                )
                score = -match_stats["match_win"]
            else:
                score = greedy["policy_pen"]
            if run:
                run.log(
                    {
                        "iter": it,
                        **stats,
                        **{f"greedy/{k}": v for k, v in greedy.items()},
                        **{f"mc16/{k}": v for k, v in mc.items()},
                        **{f"match/{k}": v for k, v in match_stats.items()},
                    }
                )
            if score < best_pen:
                best_pen = score
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
