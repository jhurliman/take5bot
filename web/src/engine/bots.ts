// Bridge to the Rust engine (WASM). The strong bots — determinized MC
// search and the trained neural net with belief-guided search — live in
// engine/take5-core; this module loads the WASM build and the exported
// network weights (public/net.t5n) on demand.

import init, { EngineBot } from "./pkg/take5_wasm";

export type EngineKind = "mc" | "neural";

let wasmReady: Promise<unknown> | null = null;
let netBytes: Uint8Array | null = null;

/** Idempotent; the WASM module and (optionally) the ~6 MB weights blob are
 * fetched once and cached for the session. */
export async function loadEngine(needNet: boolean): Promise<void> {
  wasmReady ??= init();
  await wasmReady;
  if (needNet && !netBytes) {
    const res = await fetch(`${import.meta.env.BASE_URL}net.t5n`);
    if (!res.ok) throw new Error(`failed to fetch net.t5n: ${res.status}`);
    netBytes = new Uint8Array(await res.arrayBuffer());
  }
}

/** Requires a prior successful `loadEngine`. Browser search sizes are
 * tuned for main-thread latency: mc:64 ≈ instant, neural:16 ≈ 100-400 ms
 * per move. */
export function createEngineBot(kind: EngineKind, seed: number): EngineBot {
  if (kind === "neural") {
    if (!netBytes) throw new Error("engine not loaded with weights");
    return new EngineBot("neural:16", netBytes, BigInt(seed >>> 0));
  }
  return new EngineBot("mc:64", undefined, BigInt(seed >>> 0));
}

export interface SeatView {
  player: number;
  numPlayers: number;
  hand: number[]; // card ids
  rows: number[][]; // card ids per row
  penalties: number[]; // bull totals per seat (current deal)
  totals: number[]; // match totals carried from previous deals, per seat
  played: number[]; // all publicly revealed card ids
  turn: number; // 0..9
}

/** Coach mode: per-card scores (higher = better) from a neural bot. */
export function engineAnalyze(bot: EngineBot, view: SeatView): Map<number, number> {
  const rowsFlat: number[] = [];
  const rowLens: number[] = [];
  for (const row of view.rows) {
    rowLens.push(row.length);
    rowsFlat.push(...row);
  }
  const flat = bot.analyze(
    view.player,
    view.numPlayers,
    Uint8Array.from(view.hand),
    Uint8Array.from(rowsFlat),
    Uint8Array.from(rowLens),
    Uint16Array.from(view.penalties),
    Uint16Array.from(view.totals),
    Uint8Array.from(view.played),
    view.turn,
  );
  const scores = new Map<number, number>();
  for (let i = 0; i < flat.length; i += 2) scores.set(flat[i], flat[i + 1]);
  return scores;
}

export function engineChooseCard(bot: EngineBot, view: SeatView): number {
  const rowsFlat: number[] = [];
  const rowLens: number[] = [];
  for (const row of view.rows) {
    rowLens.push(row.length);
    rowsFlat.push(...row);
  }
  return bot.choose_card(
    view.player,
    view.numPlayers,
    Uint8Array.from(view.hand),
    Uint8Array.from(rowsFlat),
    Uint8Array.from(rowLens),
    Uint16Array.from(view.penalties),
    Uint16Array.from(view.totals),
    Uint8Array.from(view.played),
    view.turn,
  );
}
