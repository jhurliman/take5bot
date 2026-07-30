# take5bot web UI

Play [Take 5 (6 nimmt!)](https://en.wikipedia.org/wiki/6_nimmt!) in the
browser against bots, from random-card baselines up to a trained neural
net with belief-guided search.

**Live: https://jhurliman.github.io/take5bot/**

A Vite + React + TypeScript app. Game state and the two easy difficulties
(Random, Greedy) are pure TypeScript; the two strong difficulties run the
real Rust engine compiled to WebAssembly:

- **Search** — determinized Monte Carlo rollouts (`mc:64`)
- **Neural** — trained policy/value/belief net + one-ply expectimax over
  belief-sampled worlds (`neural:16`), plus a coach mode that scores the
  human's hand

## Dev

```sh
cd web
npm ci
npm run dev     # local dev server
npm run lint
npm run build   # tsc + vite build into dist/
```

## The WASM engine and weights

`src/engine/pkg/` is the wasm-pack output of `engine/take5-wasm`
(bindings over `engine/take5-core`). It is committed to git so web
development and CI need no Rust toolchain; rebuild it after engine changes
with `scripts/build_wasm.sh` from the repo root.

The Neural bot's weights live in `public/net.t5n` (~2.9 MB, f16 T5N2
format, exported by `training/export_net.py`) and are fetched lazily via
`import.meta.env.BASE_URL` the first time a neural bot is needed. A neural
move takes ~60–100 ms on the main thread — see the Performance section in
`docs/ARCHITECTURE.md`.

## Deployment

Merges to `main` that touch `web/` auto-deploy to GitHub Pages via
`.github/workflows/deploy-pages.yml` (build `web/dist`, publish with
`actions/deploy-pages`). `base: "/take5bot/"` in `vite.config.ts` matches
the project-pages URL and is also used by the dev server.
