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
const playMove = (spec, weightsFile) => {
  const weights = readFileSync(join(root, `web/public/${weightsFile}`));
  const bot = new pkg.EngineBot(spec, weights, 1n);
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
  return card;
};

// Coach net (MLP + search) and opponent net (transformer raw policy).
// Both agree card 90 is the right play from this state; a format/pkg
// mismatch throws or returns garbage instead.
for (const [spec, file] of [["neural:8", "net.t5n"], ["neural:0", "net-attn.t5n"]]) {
  const card = playMove(spec, file);
  if (card !== 90) {
    console.error(`web engine check FAILED (${file}): expected card 90, got ${card}`);
    process.exit(1);
  }
}
console.log("web engine check OK: committed pkg + both weight files load and play correctly");
