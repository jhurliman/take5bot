# take5bot

A genuinely strong [Take 5 (6 nimmt!)](https://www.amigo.games/wp-content/uploads/2024/08/18415-TakeNumber_Rules.pdf)
bot — and a browser game where you can play against it.

**▶ Play now: <https://jhurliman.github.io/take5bot/>**

Full matches to 66 bulls against three bots, difficulty levels from
random up to the neural champion, and an optional coach that scores
every card in your hand (★ best play, `+n` extra bulls a play risks).

## How it works

- **One Rust rules engine, three targets** (`engine/take5-core`, zero
  dependencies): built natively for training and the arena (PyO3),
  and to WASM for the browser — every surface shares one brain.
- **PPO league self-play** (`training/train_ppo.py`, PyTorch): per-seat
  views handle the hidden information correctly; a league of frozen
  past selves, heuristic exploiters, and frozen champions guards
  against self-play overfitting. An auxiliary belief head predicts who
  holds every unseen card.
- **The champion is a card-token transformer** (d192×4, ~1.8M params)
  trained by distilling the best MLP generation and then fine-tuning
  with PPO — direct PPO fails on this architecture, which is itself one
  of the project's findings. Its raw policy beats the previous
  champion's *search mode* head-to-head with zero search of its own.
- **The arena is the referee** (`arena/`): fixed baselines, rotating
  seats, reproducible games, Elo ladder. A candidate is only "better"
  when it wins with statistical significance.

The design, every experiment, and the full results record live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The original
OpenSpiel + LightZero MuZero implementation this replaced remains in
`take5bot/` as an arena baseline (it loses to the greedy heuristic).

## Quick start

```bash
# toolchain: Rust (stable), Python 3.10+, uv
uv venv --python=python3.10 && uv sync
./scripts/build_engine.sh          # builds py/take5_engine.so

# sanity: move-for-move parity vs the legacy OpenSpiel rules
.venv/bin/python tests/parity_check.py

# arena: play bots against each other
.venv/bin/python arena/run_arena.py --bots greedy,mc:64,random,lowest

# train (PPO league self-play; ~40k samples/s on a single GPU)
.venv/bin/python training/train_ppo.py --iters 2000 --out training/runs/my-run

# web app (Vite + React; WASM engine is committed in web/src/engine/pkg)
cd web && npm ci && npm run dev
```

Rebuild the WASM engine after Rust changes with `./scripts/build_wasm.sh`
(requires [wasm-pack](https://rustwasm.github.io/wasm-pack/); the build
uses WASM SIMD).

## License

MIT License — see LICENSE file for details.
