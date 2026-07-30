# take5bot v2 architecture

Goal: a genuinely strong Take 5 (6 nimmt!) bot, playable in the browser.

## Why not MuZero (the v1 approach)

Take 5 is an imperfect-information, simultaneous-move game. MuZero assumes
perfect information and sequential moves; making it fit required wrapping the
4-player game as a single-agent env stepped round-robin, which broke reward
credit assignment (penalties resolved in a joint step were attributed to
whichever seat happened to be "current"). The v1 stack lives on in `take5bot/`
and its checkpoints remain arena baselines.

## The v2 recipe

1. **One rules engine, three targets.** `engine/take5-core` (Rust, zero
   dependencies) is the single source of truth for rules and observation
   encoding. It builds natively for training and the arena (via
   `engine/take5-py`, PyO3) and will build to WASM for the browser. The
   legacy OpenSpiel implementation is the reference: `tests/parity_check.py`
   drives both engines through identical games and asserts identical state
   after every action.
2. **Model-free self-play (PPO) for training.** Each seat sees only its own
   `View`, so hidden information is handled correctly by construction and
   simultaneous moves need no serialization hack. League play (checkpoint
   pool + heuristic exploiters) guards against self-play overfitting.
3. **Search only at inference.** Determinized rollouts/IS-MCTS on top of the
   trained net, with opponent hands sampled from a learned belief head.
   Strength scales with think time; runs in the browser via WASM.
4. **The arena is the referee.** Fixed baselines (random, lowest, greedy,
   `mc:<worlds>` determinized rollout search, legacy MuZero checkpoint);
   seats rotate; every game reproducible from `(seed, index)`. A candidate is
   only "better" when it beats the incumbent with statistical significance.

## Observation encoding (v2, 264 dims)

Defined once in `take5-core/src/obs.rs` (see the layout table there). Key
additions over v1: the played-cards mask (card counting), all players'
penalty totals seat-relative, turn/hand counters, and forced-row-choice
context. Bullhead values are not encoded — they are a deterministic function
of card id. The schema supports 2-10 players from day one.

## Components

| Path | What |
| --- | --- |
| `engine/take5-core` | Rules, `View` (per-seat info barrier), obs encoding, SplitMix64 RNG, baseline bots, match runner |
| `engine/take5-py` | `take5_engine` Python module: `Game`, `run_arena` (multithreaded, GIL-released) |
| `scripts/build_engine.sh` | Builds `py/take5_engine.so` (abi3, py>=3.10) |
| `arena/run_arena.py` | CLI: mean penalty ± 95% CI and win rates per bot |
| `tests/parity_check.py` | Move-for-move parity vs the legacy OpenSpiel implementation |

## Baseline results (20k games for heuristics, 4k for mc:64, seed 0)

| matchup | mean penalty | win rate |
| --- | --- | --- |
| greedy vs lowest + 2x random | 8.3 vs 13.0 / 15.9 / 16.1 | 47% |
| mc:64 vs 3x greedy | 7.6 vs ~14.7 | 45% (25% = parity) |

Heuristic-only games run at ~1.1M games/s on a desktop CPU; `mc:64` plays
~700 full games/s including its internal search.

## Milestones

- [x] M1: Rust engine, parity with legacy implementation, Python bindings
- [x] M2: Arena + baselines including determinized MC rollout search
- [x] M3: PPO self-play training (`training/train_ppo.py` on a vectorized
      `VecGames` env with mixed self-play/bot opponent pools;
      `training/eval_arena.py` evaluates checkpoints). The raw policy beats
      greedy (11.6 vs 13.2 mean penalty over 5k games) and matches mc:16;
      mc:64 still wins — that gap is M4/M5's job (league + search).
- [x] M4: League training (frozen past-self snapshots in the rollout pool)
      + belief head: auxiliary cross-entropy predicting, per unseen card,
      which opponent holds it or whether it is in the stock. Targets come
      from `VecGames.belief_targets()` (training-only hidden-state read).
- [x] M5a: Belief-guided determinized search at inference. Pure-Rust
      inference of the trained net (`take5-core/src/neural.rs`, exported by
      `training/export_net.py`, torch-parity-tested); `NeuralSearchBot` does
      one-ply expectimax over belief-sampled worlds with value bootstrap.
      Arena spec: `neural:<weights>[:worlds]` (`:0` = raw policy).
      **Result: beats 3x mc:64** (11.2 vs 12.2-12.9 mean penalty, 29.3% win
      over 1500 games); mixed field: neural:32 9.5 > mc:64 10.6 >
      neural:0 12.6 > greedy 16.4.
- [x] M5b: WASM build (`engine/take5-wasm`, built by
      `scripts/build_wasm.sh` into `web/src/engine/pkg`) and browser
      integration: the web UI's bot difficulty setting now offers Random /
      Greedy (TS heuristics) and Search (mc:64) / Neural (trained net +
      belief search, weights fetched from `public/net.t5n`) running the
      real engine in WASM. Verified in a Node runtime smoke test.

### Training notes (M3)

`VecGames` (in `take5-py`) steps N deals in lockstep; every deal is exactly
10 simultaneous turns, so rollouts are rectangular. Per-seat rewards are
relative bull deltas (mean of others minus own) attributed at resolution
time — they sum to the final relative score and are zero-sum across seats.
Policy-seat forced row choices use the cheapest-row heuristic in v1 (a
row-choice head is M4 work). Pure mirror self-play plateaued without
transferring to other styles, so training mixes pools: half pure self-play,
half with greedy/random/mc:8 bot seats. M4 adds league envs (frozen
past-self snapshots) and the belief head; its best checkpoint reaches
11.0 vs greedy's 13.4 (5k games, 27.9% win), parity with mc:16, and
still trails mc:64 (13.4 vs 11.8) — closing that gap is M5's
belief-guided search at inference.
