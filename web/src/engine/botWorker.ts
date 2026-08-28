/// <reference lib="webworker" />
// Opponent worker: hosts one EngineBot per bot seat and picks their
// cards off the main thread.
//
// Previously the opponents' choose_card ran synchronously in
// onPlaySelected: ~15-35 ms per move with up to 9 bot seats, so ~315 ms
// of frozen UI every turn on desktop and considerably worse on a phone.
// Same request/reply-by-id shape as coachWorker.ts.

import init, { EngineBot } from "./pkg/take5_wasm";

interface InitMsg {
  type: "init";
  weights: Uint8Array | null;
}

interface CreateMsg {
  type: "create";
  pid: number;
  spec: string;
  seed: number;
}

interface ChooseMsg {
  type: "choose";
  id: number;
  pid: number;
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

interface DropMsg {
  type: "drop";
}

const bots = new Map<number, EngineBot>();
let weights: Uint8Array | null = null;
let ready: Promise<void> | null = null;

function dropAll() {
  for (const bot of bots.values()) bot.free();
  bots.clear();
}

self.onmessage = async (e: MessageEvent<InitMsg | CreateMsg | ChooseMsg | DropMsg>) => {
  const msg = e.data;

  if (msg.type === "init") {
    weights = msg.weights;
    ready = init().then(() => undefined);
    try {
      await ready;
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", id: -1, message: String(err) });
    }
    return;
  }

  if (msg.type === "drop") {
    dropAll();
    return;
  }

  if (msg.type === "create") {
    try {
      await ready;
      if (bots.has(msg.pid)) return;
      const needsNet = msg.spec.startsWith("neural");
      if (needsNet && !weights) throw new Error("bot worker has no weights");
      bots.set(
        msg.pid,
        new EngineBot(msg.spec, needsNet ? weights! : undefined, BigInt(msg.seed >>> 0)),
      );
    } catch (err) {
      self.postMessage({ type: "error", id: -1, message: String(err) });
    }
    return;
  }

  try {
    await ready;
    const bot = bots.get(msg.pid);
    if (!bot) throw new Error(`no bot for seat ${msg.pid}`);
    const card = bot.choose_card(
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
    self.postMessage({ type: "result", id: msg.id, card });
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String(err) });
  }
};
