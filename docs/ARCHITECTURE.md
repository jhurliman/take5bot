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

## Ceiling probes and match awareness (M8)

Exploitability: a best-response net warm-started from the M7 champion and
trained 1000 iters purely against it reaches only 26.2% win vs 3 raw-M7
seats (25% = parity, 3000 games) and loses to the search champion (18.2%)
— near-unexploitable within this model class, corroborating the flattening
self-play curves. (Caveat: that probe's rollout pool gave the learner two
seats in a third of its games, diluting the best response and possibly
understating exploitability; --exploit now uses a single learner seat in
every game, and the probe re-run is tracked under M11.) Match awareness: obs v3 appends carried standings
(append-only; old nets read their prefix), VecGames match mode plays deals
to 66 with a zero-sum outcome bonus, and the M8 generation trained with
--match-to 66 wins 28.1% of matches vs standings-blind M7 seats (26.3%
self-play calibration; ~1.8 fewer bulls per match) while holding a 51.6%
per-deal search edge. The web app passes live standings to bots and coach.

## Capacity experiment (M9): bigger nets don't help

A 1024-wide, 3-block net (4.5x parameters) trained from scratch under the
full modern curriculum (match mode, M8 champion anchor, league) loses to
the 512x2 champion head-to-head with search: 41.1% after 3000 iters, still
42.9% after 6000 (~ the small-net lineage's cumulative budget). Its
vs-greedy eval never reached the lineage's level. What this shows: under a
matched curriculum-and-iteration budget, extra MLP capacity alone is not
the lever — the champion's strength lives in the accumulated multi-
generation curriculum. What it does not show: a hard capacity ceiling.
The big net trained from scratch while the champion is the product of
warm-started generations, so a capacity gain via a better-controlled
protocol (e.g. distill champion -> big net, then continue training)
remains possible but unmotivated while the exploiter probe finds so
little to exploit. Note the probe's 26.2% bound is in-class (its best
response is another instance of the same MLP family); architectures
outside that class — attention/transformer encoders — are not bounded by
it and are tested separately (M11 below).

## Match-credit fixes and out-of-class probes (M11)

Review feedback on the M8 trainer surfaced three real issues, fixed here:

1. Match mode treated every deal boundary as terminal: `collect` discarded
   `match_dones` and GAE bootstrapped zero after turn 10, so actions in
   earlier deals earned no credit for the eventual match outcome and the
   value head trained on truncated returns. GAE now bootstraps the value
   of the fresh deal the env has already dealt (standings carried), zeroed
   only where the match actually ended.
2. `--exploit` gave the learner two seats in a third of its games. The
   pool is now strictly one learner seat vs three frozen champion seats.
3. `best.pt` was selected on per-deal greedy penalty even in match mode;
   match-mode runs now gate on `eval_match` win rate against the anchor.

**M10 (control): match-credit fix alone doesn't gate.** The champion
512x2 MLP warm-started from M8 and retrained 2000 iters with corrected
match credit is a coin flip against M8 in search h2h (50.2%, 2000 games)
and only +0.7 above self-calibration in match win (25.2% vs 24.5%, 5000
matches). M8 stayed champion — until M11.

**M11: the attention encoder is a real, decisive win.** Architecture:
one token per card (in-hand, played, bulls, value, on-table, row, slot,
is-row-tail + learned card embedding), a CLS token carrying the non-card
observation tail, pre-LN transformer (d192, 4 layers, 6 heads, 1.8M
params — same scale as the champion MLP), per-card-token policy logits,
CLS value, per-token belief. Three findings, in order:

1. *From-scratch PPO cannot train it.* Two independent runs (lr 3e-4;
   final-LayerNorm + lr 1e-4) both plateau at greedy-inferior play
   (~14 bulls vs greedy) for 1500-2000 iters while the belief head
   converges normally — the encoder learns the game's information
   structure but PPO never finds a strong policy.
2. *The architecture represents champion play easily.* Supervised
   distillation of M8 (training/distill.py: soft CE on legal-card
   distribution + value MSE + belief CE) reaches champion-level raw
   play in ~100 iters and exact parity vs raw M8 (24.6% with 25% =
   parity, 4000 deals / 2000 matches). The from-scratch failure is an
   optimization pathology, not a capacity or expressiveness limit.
3. *Distill-then-finetune breaks the MLP ceiling.* PPO finetuning the
   distilled student (lr 3e-5, clip 0.1 — at lr 1e-4 PPO destroys the
   distilled policy within 250 iters; the MLP tolerates 3e-4) under the
   match-mode champion-anchored curriculum climbs monotonically for
   3000 iters. Final gates: 28.8% deal win vs 3x raw M8 (4000 deals,
   25% = parity); 35.0% match win vs 24.5% self-calibration (5000
   matches, ~15 sigma); and it beats even M8's *search* mode — 26.3%
   deal win / 29.2% match win vs 3 seats of neural:16, and 54.8% h2h
   (700 games) in the Rust engine 2v2 ladder (~ +67 Elo) — with zero
   search of its own.

Deployment: "T5N3" weight format (attention tensors + heads count),
pure-Rust transformer forward in neural.rs (card features rebuilt from
the observation, parity-tested against torch on real states, f32+f16),
~67 ms/move raw-policy in WASM — the same latency as the MLP search bot,
so the web app's Neural difficulty now plays the M11 transformer raw
policy. The coach stays on the MLP+search net: its scores are in bull
units (analyze() with worlds>0), which the "+n bulls" badges need;
wrapping determinized search around the 400 MFLOP attention forward
would cost seconds per move for no coach benefit.

The M9 capacity conclusion stands refined: more MLP capacity was not
the lever, but architecture was — and it was only reachable through
distillation, since direct PPO fails on the transformer at every
learning rate tried. The in-class exploitability probe (26.2%) indeed
did not bound the out-of-class gain, exactly as the review cautioned.

**Exploitability of the new champion** (corrected single-learner-seat
pools; champion seats driven by the frozen torch net at argmax via
--opponent-ckpt, since engine-bot attention opponents throttle rollouts
~150x): an MLP best-response warm-started from M10 and trained 1000
iters purely against M11 finishes *below* parity — 20.6% win vs 3 raw
M11 seats (3000 deals, 25% = parity) — the entire MLP class cannot even
draw with the transformer. A same-class attention best-response
warm-started from M11 itself reaches 25.9%, within noise of parity:
M11 is near-unexploitable in its own class. With the champion strictly
dominating the old class and measuring clean on the corrected probe,
the strength program closes again at this scale — one tier higher.

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
