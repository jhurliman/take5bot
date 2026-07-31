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

## Performance (browser WASM, and why not WebGPU)

Measured on the release WASM build (Node 22 runtime, f16 `net.t5n`,
`neural:16`, i.e. 16 belief-sampled worlds — the browser's setting), 20
timed `choose_card` calls per scenario:

| scenario | per-move latency (mean, tight spread) |
| --- | --- |
| mid-game, 5-card hand (turn 5) | ~63 ms |
| worst case: 10-card hand (turn 0) | ~103 ms |

Enabling WASM SIMD (`RUSTFLAGS='-C target-feature=+simd128'`) was measured
and made no difference (62.6 vs 62.5 ms mean), so it is not part of the
build. The net is a 512-wide 2-block MLP (~1.4 MFLOPs/forward; a move runs
a few hundred forwards), which scalar WASM already handles in ~100 ms worst
case.

**Verdict: WebGPU is unnecessary at this model size.** GPU dispatch
overhead and weight upload would likely cost more than they save for
1.4 MFLOP forwards, and the latency is already well under perceptible
"thinking time" for a card game. If UI smoothness ever matters (the search
currently runs on the main thread and can block a frame for ~100 ms), the
right next step is moving the bot into a Web Worker — not GPU inference.

## Strength program: closed at this scale

Generational self-improvement (train vs frozen previous champion) was run
twice: M6 beat M4 52.6% head-to-head, M7 beat M6 51.6% (6000 games) — gains
are flattening, indicating convergence for this net size and training
scale. Think-time scaling still works and is a runtime knob: 64 worlds
beats 16 worlds 52.7% h2h, so the efficient axis is more determinizations
(breadth), not deeper search — which is why SM-ISMCTS was evaluated and
rejected: at any fixed latency it would trade away worlds in a game where
hidden-hand uncertainty dominates. Web worker and WebGPU were likewise
resolved by measurement (63-103 ms/move needs neither). Next frontiers, if
ever revisited: larger nets, multi-hour search-pressure training, and
2-10-player conditioning.

## Legacy comparison

`arena/legacy_bridge.py` plays the original v1 MuZero checkpoint (LightZero,
253-dim obs, policy-head argmax — exactly how v1's play script used it)
against v2 engine bots by reconstructing its observation from the v2
encoding. Result: v1 loses even to the greedy heuristic (16.1 vs 11.8 mean
penalty, 12.4% win over 1000 games) — roughly random/lowest tier. The v2
raw policy scores ~10.6 vs greedy (~30% win) and its search mode beats
mc:64, i.e. the rewrite moved the bot from below-heuristic to
above-search-baseline strength.

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
      real engine in WASM. Verified in a Node runtime smoke test. Coach
      mode (toolbar lightbulb) scores the human's hand with the neural
      search bot (`NeuralSearchBot::analyze`) and badges each card with its
      expected bull cost relative to the bot's pick.

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
