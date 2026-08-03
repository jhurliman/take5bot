#!/usr/bin/env node
// Guard against the committed WASM pkg and the committed net weights
// drifting out of sync (e.g. a new weights format shipped without
// rebuilding web/src/engine/pkg). Loads exactly what the site serves:
// the web-target pkg via initSync plus web/public/net.t5n, then makes a
// neural bot move. Run from the repo root; used by deploy-pages.yml.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = await import(
  join(root, "web/src/engine/pkg/take5_wasm.js")
);
pkg.initSync({
  module: readFileSync(join(root, "web/src/engine/pkg/take5_wasm_bg.wasm")),
});

const rows = [[4], [15], [20, 25, 30, 31, 35], [80]];
const weights = readFileSync(join(root, "web/public/net-attn.t5n"));
const viewArgs = [
  0,
  4,
  Uint8Array.from([36, 90]),
  Uint8Array.from(rows.flat()),
  Uint8Array.from(rows.map((r) => r.length)),
  Uint16Array.from([0, 0, 0, 0]),
  Uint16Array.from([12, 40, 7, 63]), // carried match totals (standings-aware bot)
  Uint8Array.from(rows.flat()),
  8,
];

// Opponent path: the champion transformer's raw policy.
const bot = new pkg.EngineBot("neural:0", weights, 1n);
const card = bot.choose_card(...viewArgs);
bot.free();
if (card !== 90) {
  console.error(`web engine check FAILED: expected card 90, got ${card}`);
  process.exit(1);
}

// Coach path: same net, belief-guided analyze (worlds > 0) — must score
// every hand card with finite bull-unit values and rank 90 above 36.
const coach = new pkg.EngineBot("neural:4", weights, 1n);
const flat = coach.analyze(...viewArgs);
coach.free();
const scores = new Map();
for (let i = 0; i < flat.length; i += 2) scores.set(flat[i], flat[i + 1]);
if (
  scores.size !== 2 ||
  ![...scores.values()].every(Number.isFinite) ||
  !(scores.get(90) > scores.get(36))
) {
  console.error(`web engine check FAILED: bad coach scores`, [...scores]);
  process.exit(1);
}
// Row-choice coach path: playing the 3 (below every row end) forces a
// row choice; all four rows must get finite scores and the cheap
// single-card rows must beat taking the 5-card row 3.
const rowCoach = new pkg.EngineBot("neural:4", weights, 1n);
const rowsFlat = rowCoach.analyze_rows(
  0,
  4,
  Uint8Array.from([3, 90]),
  Uint8Array.from(rows.flat()),
  Uint8Array.from(rows.map((r) => r.length)),
  Uint16Array.from([0, 0, 0, 0]),
  Uint16Array.from([12, 40, 7, 63]),
  Uint8Array.from(rows.flat()),
  8,
  3,
);
rowCoach.free();
const rowScores = new Map();
for (let i = 0; i < rowsFlat.length; i += 2) rowScores.set(rowsFlat[i], rowsFlat[i + 1]);
if (
  rowScores.size !== 4 ||
  ![...rowScores.values()].every(Number.isFinite) ||
  !(rowScores.get(0) > rowScores.get(2))
) {
  console.error(`web engine check FAILED: bad row scores`, [...rowScores]);
  process.exit(1);
}
console.log("web engine check OK: committed pkg + net-attn.t5n play, analyze, and row-coach correctly");
