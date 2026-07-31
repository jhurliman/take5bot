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

const weights = readFileSync(join(root, "web/public/net.t5n"));
const bot = new pkg.EngineBot("neural:8", weights, 1n);

const rows = [[4], [15], [20, 25, 30, 31, 35], [80]];
const card = bot.choose_card(
  0,
  4,
  Uint8Array.from([36, 90]),
  Uint8Array.from(rows.flat()),
  Uint8Array.from(rows.map((r) => r.length)),
  Uint16Array.from([0, 0, 0, 0]),
  Uint16Array.from([12, 40, 7, 63]), // carried match totals (standings-aware bot)
  Uint8Array.from(rows.flat()),
  8,
);
bot.free();

if (card !== 90) {
  console.error(`web engine check FAILED: expected card 90, got ${card}`);
  process.exit(1);
}
console.log("web engine check OK: committed pkg + net.t5n load and play correctly");
