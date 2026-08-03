/// <reference lib="webworker" />
// Coach worker: runs the champion net's belief-guided analyze (worlds > 0,
// bull-unit scores) off the main thread. One request in flight at a time;
// the client discards stale replies by id.

import init, { EngineBot } from "./pkg/take5_wasm";

interface InitMsg {
  type: "init";
  spec: string;
  weights: Uint8Array;
  seed: number;
}

interface AnalyzeMsg {
  type: "analyze";
  id: number;
  player: number;
  numPlayers: number;
  hand: Uint8Array;
  rowsFlat: Uint8Array;
  rowLens: Uint8Array;
  penalties: Uint16Array;
  totals: Uint16Array;
  played: Uint8Array;
  turn: number;
}

let bot: EngineBot | null = null;
let ready: Promise<void> | null = null;

self.onmessage = async (e: MessageEvent<InitMsg | AnalyzeMsg>) => {
  const msg = e.data;
  if (msg.type === "init") {
    ready = (async () => {
      await init();
      bot = new EngineBot(msg.spec, msg.weights, BigInt(msg.seed >>> 0));
    })();
    try {
      await ready;
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", id: -1, message: String(err) });
    }
    return;
  }
  try {
    await ready;
    if (!bot) throw new Error("coach worker not initialized");
    const flat = bot.analyze(
      msg.player,
      msg.numPlayers,
      msg.hand,
      msg.rowsFlat,
      msg.rowLens,
      msg.penalties,
      msg.totals,
      msg.played,
      msg.turn,
    );
    self.postMessage({ type: "result", id: msg.id, scores: flat });
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String(err) });
  }
};
