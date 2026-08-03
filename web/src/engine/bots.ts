// Bridge to the Rust engine (WASM). Every neural role on the site — the
// opponents and the coach — runs the champion transformer net (M11,
// public/net-attn.t5n). Opponents play its raw policy (~15-35 ms/move
// with the SIMD build); the coach runs its belief-guided analyze
// (worlds > 0, bull-unit scores) in a web worker so the ~1-2 s search
// never blocks the UI.

import init, { EngineBot } from "./pkg/take5_wasm";

export type EngineKind = "mc" | "attn";

/** Determinized worlds for the coach's analyze. More worlds = steadier
 * cost estimates, linearly slower (~250 ms/world in the worker). */
const COACH_WORLDS = 4;

let wasmReady: Promise<unknown> | null = null;
let attnBytes: Uint8Array | null = null;

async function fetchWeights(name: string): Promise<Uint8Array> {
  const res = await fetch(`${import.meta.env.BASE_URL}${name}`);
  if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** Idempotent; the WASM module and (if needed) the ~3.6 MB champion
 * weights are fetched once and cached for the session. */
export async function loadEngine(needNet: boolean): Promise<void> {
  wasmReady ??= init();
  await wasmReady;
  if (needNet && !attnBytes) attnBytes = await fetchWeights("net-attn.t5n");
}

/** Requires a prior successful `loadEngine`. */
export function createEngineBot(kind: EngineKind, seed: number): EngineBot {
  if (kind === "attn") {
    if (!attnBytes) throw new Error("engine not loaded with weights");
    return new EngineBot("neural:0", attnBytes, BigInt(seed >>> 0));
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

function flatView(view: SeatView) {
  const rowsFlat: number[] = [];
  const rowLens: number[] = [];
  for (const row of view.rows) {
    rowLens.push(row.length);
    rowsFlat.push(...row);
  }
  return {
    player: view.player,
    numPlayers: view.numPlayers,
    hand: Uint8Array.from(view.hand),
    rowsFlat: Uint8Array.from(rowsFlat),
    rowLens: Uint8Array.from(rowLens),
    penalties: Uint16Array.from(view.penalties),
    totals: Uint16Array.from(view.totals),
    played: Uint8Array.from(view.played),
    turn: view.turn,
  };
}

/** Coach client: the champion net's analyze in a dedicated worker.
 * Replies are matched by request id; stale replies are dropped. */
export class Coach {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (s: Map<number, number>) => void; reject: (e: Error) => void }
  >();

  private constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "ready") return;
      const req = this.pending.get(msg.id);
      if (!req) return;
      this.pending.delete(msg.id);
      if (msg.type === "error") {
        req.reject(new Error(msg.message));
        return;
      }
      const scores = new Map<number, number>();
      const flat: Float32Array = msg.scores;
      for (let i = 0; i < flat.length; i += 2) scores.set(flat[i], flat[i + 1]);
      req.resolve(scores);
    };
  }

  static async create(seed: number): Promise<Coach> {
    await loadEngine(true);
    const worker = new Worker(new URL("./coachWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.postMessage({
      type: "init",
      spec: `neural:${COACH_WORLDS}`,
      weights: attnBytes,
      seed,
    });
    return new Coach(worker);
  }

  analyze(view: SeatView): Promise<Map<number, number>> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "analyze", id, ...flatView(view) });
    });
  }

  dispose(): void {
    this.worker.terminate();
    for (const req of this.pending.values()) {
      req.reject(new Error("coach disposed"));
    }
    this.pending.clear();
  }
}

export function engineChooseCard(bot: EngineBot, view: SeatView): number {
  const v = flatView(view);
  return bot.choose_card(
    v.player,
    v.numPlayers,
    v.hand,
    v.rowsFlat,
    v.rowLens,
    v.penalties,
    v.totals,
    v.played,
    v.turn,
  );
}
