// Bridge to the Rust engine (WASM). Every neural role on the site (the
// opponents and the coach) runs the champion transformer net (M11,
// public/net-attn.t5n).
//
// Both roles now run in dedicated workers: the opponents play the net's
// raw policy (~15-35 ms/move) in botWorker.ts, and the coach runs its
// belief-guided analyze (worlds > 0, bull-unit scores, ~1-2 s) in
// coachWorker.ts. The main thread therefore never instantiates the WASM
// module at all; it only fetches the weights and hands them over.

export type EngineKind = "mc" | "attn";

/** Determinized worlds for the coach's analyze. More worlds = steadier
 * cost estimates, linearly slower (~250 ms/world in the worker). */
const COACH_WORLDS = 4;

let weightsPromise: Promise<Uint8Array> | null = null;

/** Idempotent; the ~3.6 MB champion weights are fetched once and shared
 * by every worker for the session. */
export function loadWeights(): Promise<Uint8Array> {
  weightsPromise ??= (async () => {
    const name = "net-attn.t5n";
    const res = await fetch(`${import.meta.env.BASE_URL}${name}`);
    if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  })();
  return weightsPromise;
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

/** Shared request/reply plumbing: one worker, replies matched by id. */
abstract class WorkerClient<T> {
  protected worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: T) => void; reject: (e: Error) => void }
  >();

  protected constructor(worker: Worker, extract: (msg: unknown) => T) {
    this.worker = worker;
    this.worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "ready") return;
      if (msg.type === "error" && msg.id === -1) {
        console.error("engine worker error:", msg.message);
        return;
      }
      const req = this.pending.get(msg.id);
      if (!req) return;
      this.pending.delete(msg.id);
      if (msg.type === "error") {
        req.reject(new Error(msg.message));
        return;
      }
      req.resolve(extract(msg));
    };
  }

  protected request(payload: Record<string, unknown>): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }

  dispose(): void {
    this.worker.terminate();
    for (const req of this.pending.values()) req.reject(new Error("worker disposed"));
    this.pending.clear();
  }
}

function scoresFrom(msg: { scores: Float32Array }): Map<number, number> {
  const scores = new Map<number, number>();
  const flat = msg.scores;
  for (let i = 0; i < flat.length; i += 2) scores.set(flat[i], flat[i + 1]);
  return scores;
}

/** Coach client: the champion net's analyze in a dedicated worker. */
export class Coach extends WorkerClient<Map<number, number>> {
  private constructor(worker: Worker) {
    super(worker, (msg) => scoresFrom(msg as { scores: Float32Array }));
  }

  static async create(seed: number): Promise<Coach> {
    const weights = await loadWeights();
    const worker = new Worker(new URL("./coachWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.postMessage({
      type: "init",
      spec: `neural:${COACH_WORLDS}`,
      weights,
      seed,
    });
    return new Coach(worker);
  }

  analyze(view: SeatView): Promise<Map<number, number>> {
    return this.request({ type: "analyze", ...flatView(view) });
  }

  /** Score the four rows for a forced row choice (bull units, higher =
   * better). Keys of the returned map are row indices 0-3. */
  analyzeRows(view: SeatView, forced: number): Promise<Map<number, number>> {
    return this.request({ type: "analyzeRows", forced, ...flatView(view) });
  }
}

/** Opponent client: one EngineBot per bot seat, all in one worker. */
export class Bots extends WorkerClient<number> {
  private seats = new Set<number>();

  private constructor(worker: Worker) {
    super(worker, (msg) => (msg as { card: number }).card);
  }

  static async create(needNet: boolean): Promise<Bots> {
    const weights = needNet ? await loadWeights() : null;
    const worker = new Worker(new URL("./botWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.postMessage({ type: "init", weights });
    return new Bots(worker);
  }

  /** Idempotent per seat; safe to call every turn. */
  ensureSeat(pid: number, kind: EngineKind, seed: number): void {
    if (this.seats.has(pid)) return;
    this.seats.add(pid);
    this.worker.postMessage({
      type: "create",
      pid,
      spec: kind === "attn" ? "neural:0" : "mc:64",
      seed,
    });
  }

  chooseCard(pid: number, view: SeatView): Promise<number> {
    return this.request({ type: "choose", pid, ...flatView(view) });
  }

  /** Free the per-seat bots without tearing down the worker (and its
   * compiled module and weights). Used when a new deal is dealt. */
  reset(): void {
    this.seats.clear();
    this.worker.postMessage({ type: "drop" });
  }
}
