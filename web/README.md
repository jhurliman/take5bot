# take5bot web UI

Play [Take 5 (6 nimmt!)](https://en.wikipedia.org/wiki/6_nimmt!) in the
browser against bots, from random-card baselines up to a trained neural
net with belief-guided search.

**Live: https://jhurliman.github.io/take5bot/**

A Vite + React + TypeScript app. Game state and the two easy difficulties
(Random, Greedy) are pure TypeScript; the two strong difficulties run the
real Rust engine compiled to WebAssembly:

- **Search** - determinized Monte Carlo rollouts (`mc:64`)
- **Neural** - trained policy/value/belief net + one-ply expectimax over
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

`?layout=compact` forces the phone layout in a normal desktop window, and
`?layout=desktop` forces the desktop one. Use these to iterate: Chrome
DevTools' plain window resize does **not** set `pointer: coarse`, so
without the override every touch gate stays inert and the compact layout
never activates. `?rotate=1` (dev builds only) forces the rotate gate.

## Layout

One component tree serves both layouts. `useLayoutMode()` in `src/hooks.ts`
owns the only viewport predicate and publishes it as `data-layout` on
`<html>`; `index.css` keys its `compact:` variant off that same attribute,
so DOM structure and leaf styles cannot disagree.

The predicates are **height and pointer**, never width. A landscape iPhone
is 844-932px wide, so it passes Tailwind's `sm` (640) and `md` (768) while
being only ~330px tall in a Safari tab.

```
compact = (orientation: landscape) and (max-height: 500px) and (pointer: coarse)
rotate  = (orientation: portrait)  and (max-width: 540px)  and (pointer: coarse)
```

In compact mode the four board rows pair into a 2x2 grid with their labels
in a narrow gutter, which frees the vertical room for a full-size hand and
an 80px Play column beside it.

**Card sizing.** `CardView` takes a `w` prop that accepts any CSS length.
Every interior metric is an affine function of the width (`a*W + b`) fitted
through the two tiers the deck shipped with (72px and 92px), so both
reproduce their exact pixel values and any width between or below
interpolates. The compact widths are the `--cw-hand` / `--cw-board` custom
properties in `index.css`, so they track the live viewport and the notch
insets with no ResizeObserver. **The `142` and `158` constants in those
expressions encode the padding/gap/gutter/Play-column arithmetic - if you
change a `gap-*` or the gutter, change them too.**

## The WASM engine and weights

`src/engine/pkg/` is the wasm-pack output of `engine/take5-wasm`
(bindings over `engine/take5-core`). It is committed to git so web
development and CI need no Rust toolchain; rebuild it after engine changes
with `scripts/build_wasm.sh` from the repo root.

Both engine roles run in workers, so the main thread never instantiates
WASM at all - it only fetches the weights and hands them over:

- `src/engine/botWorker.ts` holds one `EngineBot` per opponent seat.
- `src/engine/coachWorker.ts` runs the coach's belief-guided `analyze`.

The Neural bot's weights live in `public/net-attn.t5n` (~3.6 MB, f16 T5N2
format, exported by `training/export_net.py`) and are fetched lazily via
`import.meta.env.BASE_URL` the first time a neural bot is needed.

## Assets

**Icons** are generated from the same bull silhouette the cards use
(`src/bullPaths.js`), so they cannot drift from the art. `sharp` is
deliberately not a dependency - CI runs `npm ci` on every PR and should not
pull ~30 MB of prebuilt binaries for something that regenerates roughly
never. Regenerate and commit the output:

```sh
npm i -D --no-save sharp && node scripts/gen-icons.mjs
```

**The display font** is self-hosted and subset to digits. Bungee is only
ever used for the big card number, whose text is always a card id, so
`U+0030-0039` is all that ships: 14.3 kB -> 1.1 kB, small enough that Vite
inlines it into the CSS, and it removes two render-blocking requests to a
third-party origin. Licence: SIL OFL 1.1 (`src/assets/fonts/Bungee-OFL.txt`).
To regenerate from a newer release:

```sh
python3 -m fontTools.subset bungee-latin.woff2 \
  --unicodes=U+0030-0039 --flavor=woff2 --layout-features='' \
  --no-hinting --desubroutinize --name-IDs='' \
  --output-file=src/assets/fonts/bungee-digits.woff2
```

**The service worker** (`public/sw.js`) cache-firsts the hashed assets, the
WASM and the weights so a cold visit on cellular does not re-download
3.6 MB. It is hand-rolled rather than `vite-plugin-pwa` because that
plugin's default `maximumFileSizeToCacheInBytes` is 2 MB and would silently
skip the very file this exists for. `net-attn.t5n` is served from `public/`
and is therefore not content-hashed, so **bump `CACHE_VERSION` in `sw.js`
whenever the weights change** or clients will keep serving the old net.

## Deployment

Merges to `main` that touch `web/` auto-deploy to GitHub Pages via
`.github/workflows/deploy-pages.yml` (build `web/dist`, publish with
`actions/deploy-pages`). `base: "/take5bot/"` in `vite.config.ts` matches
the project-pages URL and is also used by the dev server.

Icon and manifest `<link href>`s in `index.html` are **document-relative**
(`icon.svg`, not `/icon.svg` or `%BASE_URL%icon.svg`). The page is always
served at the base path, so they resolve correctly in dev and in the build
with the base hardcoded nowhere; `%BASE_URL%` produces a root-relative path
that the dev server then prefixes with the base a second time.
