import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, RefreshCw, Play, Settings2 } from "lucide-react";

/**
 * Take 5 (a.k.a. 6 nimmt!) – Web Frontend
 * Single-file React + TypeScript UI with click/tap input.
 *
 * Features
 * - Visual table with 4 rows, up to 5 cards per row.
 * - Click/tap one of your cards, then press “Play selected”.
 * - Handles the “choose a row to take” interaction when required.
 * - Simple bot opponents (random + greedy).
 * - Mobile-friendly (big tap targets), subtle animations.
 *
 * How to run (Vite + Tailwind)
 *   npm create vite@latest take5-web -- --template react-ts
 *   cd take5-web
 *   npm i framer-motion lucide-react
 *   npm i -D tailwindcss postcss autoprefixer
 *   npx tailwindcss init -p
 *   npm run dev
 */

// ---------- Types ----------
type Card = { id: number; bulls: number };
type Row = Card[];

type PlayerId = number; // 0..N-1

interface PlayerState {
  id: PlayerId;
  name: string;
  isHuman: boolean;
  hand: Card[]; // sorted ascending
  pen: Card[]; // collected penalty
  chosen?: Card | null; // choice for the current turn
  strategy?: BotStrategyId; // for bots
}

interface GameState {
  seed: number;
  turn: number; // 0..9 (10 turns)
  players: PlayerState[];
  rows: [Row, Row, Row, Row];
  phase: "choose" | "reveal" | "resolve" | "needRowChoice" | "gameOver";
  needRowChoiceFor?: PlayerId; // when a played card is < all rows
  pendingPlacements: Array<{ pid: PlayerId; card: Card }>; // cards to place (sorted ascending)
  history: string[]; // simple log (kept for future)
}

// ---------- RNG ----------
function xorShift32(seed: number) {
  let x = seed | 0;
  return () => {
    x ^= x << 13; x |= 0;
    x ^= x >>> 17; x |= 0;
    x ^= x << 5; x |= 0;
    return (x >>> 0) / 0xffffffff;
  };
}

// ---------- Deck & scoring ----------
function bullsFor(n: number): number {
  if (n === 55) return 7;
  if (n % 11 === 0) return 5;
  if (n % 10 === 0) return 3;
  if (n % 5 === 0) return 2;
  return 1;
}

function makeDeck(): Card[] {
  const deck: Card[] = [];
  for (let i = 1; i <= 104; i++) deck.push({ id: i, bulls: bullsFor(i) });
  return deck;
}

function shuffle<T>(arr: T[], rnd: () => number): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- Bots ----------
type BotStrategyId = "random" | "greedy";

function simulateCardPlacementCost(
  rows: [Row, Row, Row, Row],
  card: Card
): { rowIndex: number | null; cost: number } {
  // Find candidate row: highest last < card.id
  let candidate = -1;
  let candidateVal = -Infinity;
  for (let i = 0; i < 4; i++) {
    const last = rows[i][rows[i].length - 1]?.id ?? -Infinity;
    if (last < card.id && last > candidateVal) {
      candidateVal = last;
      candidate = i;
    }
  }
  if (candidate === -1) {
    // Must choose a row to take → least bulls row
    let minIdx = 0, minCost = sumBulls(rows[0]);
    for (let i = 1; i < 4; i++) {
      const c = sumBulls(rows[i]);
      if (c < minCost) { minCost = c; minIdx = i; }
    }
    return { rowIndex: minIdx, cost: minCost };
  }
  // If it would be the 6th card, you take the row
  if (rows[candidate].length >= 5) {
    return { rowIndex: candidate, cost: sumBulls(rows[candidate]) };
  }
  return { rowIndex: candidate, cost: 0 };
}

function botChooseCard(state: GameState, pid: PlayerId, strat: BotStrategyId): Card {
  const p = state.players[pid];
  const hand = p.hand;
  if (hand.length === 1) return hand[0];
  if (strat === "random") return hand[Math.floor(Math.random() * hand.length)];

  // Greedy: pick card with minimal immediate cost this step.
  let best: { card: Card; cost: number } | null = null;
  for (const card of hand) {
    const { cost } = simulateCardPlacementCost(state.rows, card);
    if (!best || cost < best.cost || (cost === best.cost && card.id < best.card.id)) {
      best = { card, cost };
    }
  }
  return best!.card;
}

// ---------- Helpers ----------
function sumBulls(row: Row | Card[]): number {
  return row.reduce((a, c) => a + c.bulls, 0);
}

function cloneRows(rows: [Row, Row, Row, Row]): [Row, Row, Row, Row] {
  return rows.map(r => r.slice()) as [Row, Row, Row, Row];
}

function nameFor(state: GameState, pid: PlayerId) {
  return state.players[pid]?.name ?? `P${pid}`;
}

function pickLeastBullsRow(rows: [Row, Row, Row, Row]) {
  let idx = 0, best = sumBulls(rows[0]);
  for (let i = 1; i < 4; i++) {
    const s = sumBulls(rows[i]);
    if (s < best) { best = s; idx = i; }
  }
  return idx;
}

// Place a single card (synchronous). If a row choice is required and callback returns null, we throw.
function placeCardIntoRows(
  rows: [Row, Row, Row, Row], pid: PlayerId, card: Card,
  chooseRowIfNeeded?: (pid: PlayerId) => number | null
): { rows: [Row, Row, Row, Row]; taken?: Card[]; placedRow?: number } {
  const newRows = cloneRows(rows);

  // pick target row by highest last < card.id
  let target = -1; let best = -Infinity;
  for (let i = 0; i < 4; i++) {
    const last = newRows[i][newRows[i].length - 1]?.id ?? -Infinity;
    if (last < card.id && last > best) { best = last; target = i; }
  }

  if (target === -1) {
    // need a row choice
    let idx: number | null = null;
    if (chooseRowIfNeeded) idx = chooseRowIfNeeded(pid);
    if (idx == null) throw new Error("Row choice required but not provided");
    const taken = newRows[idx].slice();
    newRows[idx] = [card];
    return { rows: newRows, taken, placedRow: idx };
  }

  if (newRows[target].length >= 5) {
    const taken = newRows[target].slice();
    newRows[target] = [card];
    return { rows: newRows, taken, placedRow: target };
  }

  newRows[target] = [...newRows[target], card];
  return { rows: newRows, placedRow: target };
}

// ---------- Game setup ----------
function deal(players: number, seed: number): GameState {
  const rnd = xorShift32(seed);
  const deck = shuffle(makeDeck(), rnd);
  const N = Math.max(2, Math.min(10, players));

  // Deal 10 cards each
  const hands: Card[][] = Array.from({ length: N }, () => []);
  for (let i = 0; i < 10; i++) {
    for (let p = 0; p < N; p++) hands[p].push(deck[i * N + p]);
  }
  for (const h of hands) h.sort((a, b) => a.id - b.id);

  // Next 4 cards start rows
  const rowStarters = deck.slice(N * 10, N * 10 + 4).map(c => [c]);

  const playersState: PlayerState[] = hands.map((h, i) => ({
    id: i,
    name: i === 0 ? "You" : `Bot ${i}`,
    isHuman: i === 0,
    hand: h,
    pen: [],
    strategy: i === 0 ? undefined : (i % 2 ? "greedy" : "random")
  }));

  return {
    seed,
    turn: 0,
    players: playersState,
    rows: rowStarters as [Row, Row, Row, Row],
    phase: "choose",
    pendingPlacements: [],
    history: [`Game start (seed ${seed}). Rows: ${rowStarters.map(r => r[0].id).join(", ")}`],
  };
}

// ---------- UI Root ----------
export default function Take5App() {
  const [playersCount, setPlayersCount] = useState(4);
  const [seed, setSeed] = useState<number>(() => Math.floor(1 + Math.random() * 1e9));
  const [state, setState] = useState<GameState>(() => deal(playersCount, seed));
  const [showSettings, setShowSettings] = useState(false);

  function startNewGame(pCount = playersCount, customSeed?: number) {
    const s = customSeed ?? Math.floor(1 + Math.random() * 1e9);
    setSeed(s);
    setState(deal(pCount, s));
  }

  // --- Actions ---
  function onChooseCard(card: Card) {
    if (state.phase !== "choose") return;
    const you = state.players[0];
    if (!you.hand.find(c => c.id === card.id)) return;
    setState(prev => ({ ...prev, players: prev.players.map(p => p.id === 0 ? { ...p, chosen: card } : p) }));
  }

  function onPlaySelected() {
    if (state.phase !== "choose") return;
    const you = state.players[0];
    if (!you.chosen) return;

    // Bots pick
    const withChoices = state.players.map(p => {
      if (p.isHuman) return p;
      const c = botChooseCard(state, p.id, p.strategy || "greedy");
      return { ...p, chosen: c };
    });

    setState(prev => ({ ...prev, players: withChoices, phase: "reveal" }));

    // After a short reveal, resolve in ascending order
    setTimeout(() => {
      setState(prev => {
        const placements = withChoices.map(p => ({ pid: p.id, card: p.chosen! }))
          .sort((a, b) => a.card.id - b.card.id);
        return { ...prev, pendingPlacements: placements, phase: "resolve" };
      });
    }, 450);
  }

  // Resolve queue one-by-one to animate
  useEffect(() => {
    if (state.phase !== "resolve") return;

    if (state.pendingPlacements.length === 0) {
      const playersDone = state.players.map(p => ({ ...p, chosen: null }));
      const nextTurn = state.turn + 1;
      const done = nextTurn >= 10;
      const nextPhase: GameState["phase"] = done ? "gameOver" : "choose";
      const log = done ? [
        `Game over. Scores: ${state.players.map(p => `${p.name}=${sumBulls(p.pen)}`).join(", ")}`
      ] : [];
      const t = setTimeout(() => {
        setState(prev => ({ ...prev, players: playersDone, turn: nextTurn, phase: nextPhase, history: [...prev.history, ...log] }));
      }, 350);
      return () => clearTimeout(t);
    }

    const step = state.pendingPlacements[0];
    const ps = state.players.map(p => p.id === step.pid ? { ...p, hand: p.hand.filter(c => c.id !== step.card.id) } : p);

    try {
      const result = placeCardIntoRows(state.rows, step.pid, step.card, () => null);
      const rows = result.rows;
      const taken = result.taken ?? [];
      const playersUpdated = ps.map(p => p.id === step.pid ? { ...p, pen: taken.length ? [...p.pen, ...taken] : p.pen } : p);
      const log = taken.length ? [`${nameFor(state, step.pid)} takes ${taken.length} cards (${sumBulls(taken)}⟁)`] : [];
      const t = setTimeout(() => {
        setState(prev => ({
          ...prev,
          players: playersUpdated,
          rows,
          pendingPlacements: prev.pendingPlacements.slice(1),
          history: [...prev.history, ...log],
        }));
      }, 280);
      return () => clearTimeout(t);
    } catch {
      // Row choice required
      if (step.pid === 0) {
        setState(prev => ({ ...prev, needRowChoiceFor: 0, phase: "needRowChoice" }));
      } else {
        const botIdx = pickLeastBullsRow(state.rows);
        const result = placeCardIntoRows(state.rows, step.pid, step.card, () => botIdx);
        const rows = result.rows;
        const taken = result.taken ?? [];
        const playersUpdated = ps.map(p => p.id === step.pid ? { ...p, pen: taken.length ? [...p.pen, ...taken] : p.pen } : p);
        const log = `${nameFor(state, step.pid)} chooses row ${botIdx + 1} and takes ${sumBulls(taken)}⟁`;
        const t = setTimeout(() => {
          setState(prev => ({
            ...prev,
            players: playersUpdated,
            rows,
            pendingPlacements: prev.pendingPlacements.slice(1),
            phase: "resolve",
            history: [...prev.history, log],
          }));
        }, 280);
        return () => clearTimeout(t);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase, state.pendingPlacements]);

  function onChooseRow(idx: number) {
    if (state.phase !== "needRowChoice" || state.needRowChoiceFor !== 0) return;
    const step = state.pendingPlacements[0];
    if (!step || step.pid !== 0) return;
    const result = placeCardIntoRows(state.rows, 0, step.card, () => idx);
    const rows = result.rows;
    const taken = result.taken ?? [];
    const playersUpdated = state.players.map(p => p.id === 0 ? { ...p, pen: taken.length ? [...p.pen, ...taken] : p.pen } : p);

    setState(prev => ({
      ...prev,
      players: playersUpdated,
      rows,
      pendingPlacements: prev.pendingPlacements.slice(1),
      needRowChoiceFor: undefined,
      phase: "resolve",
      history: [...prev.history, `You choose row ${idx + 1} and take ${sumBulls(taken)}⟁`]
    }));
  }

  // --- Derived ---
  const you = state.players[0];
  const score = (p: PlayerState) => sumBulls(p.pen);
  const leaderboard = useMemo(() => [...state.players].sort((a, b) => score(a) - score(b)), [state.players]);

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex flex-col">
      <TopBar
        onNew={() => startNewGame()}
        seed={seed}
        state={state}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className="flex-1 grid grid-rows-[auto_1fr_auto] gap-3 px-3 pb-3">
        {/* Status line */}
        <div className="mx-auto mt-2 text-sm text-slate-300 flex items-center gap-3">
          <span className="opacity-80">Turn {Math.min(state.turn + 1, 10)} / 10</span>
          <span className="opacity-50">•</span>
          <span className="opacity-80 capitalize">{state.phase.replace(/([a-z])([A-Z])/g, "$1 $2")}</span>
        </div>

        {/* Table */}
        <Table rows={state.rows} />

        {/* Controls + Hand */}
        <div className="max-w-6xl w-full mx-auto">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-slate-300">Your score: <b>{score(you)}</b></div>
            <button
              onClick={onPlaySelected}
              className={`px-4 py-2 rounded-2xl shadow-md bg-emerald-600 hover:bg-emerald-500 active:scale-[.98] transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2`}
              disabled={state.phase !== "choose" || !you.chosen}
            >
              <Play className="w-4 h-4" /> Play selected
            </button>
          </div>

          <Hand
            cards={you.hand}
            chosen={you.chosen?.id}
            onChoose={onChooseCard}
            disabled={state.phase !== "choose"}
          />

          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-slate-400">
            {leaderboard.map(p => (
              <div key={p.id} className="flex items-center justify-between bg-slate-900/60 rounded-xl px-3 py-2">
                <span>{p.name}</span>
                <span>{sumBulls(p.pen)}⟁</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row choice modal */}
      <AnimatePresence>
        {state.phase === "needRowChoice" && state.needRowChoiceFor === 0 && (
          <RowChoice rows={state.rows} onPick={onChooseRow} />
        )}
      </AnimatePresence>

      {/* Settings drawer */}
      <AnimatePresence>
        {showSettings && (
          <SettingsDialog
            playersCount={playersCount}
            seed={seed}
            onClose={() => setShowSettings(false)}
            onApply={(pc, sd) => { setPlayersCount(pc); startNewGame(pc, sd); setShowSettings(false); }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------- Pieces ----------
function TopBar({ onNew, seed, state, onOpenSettings }:{
  onNew:()=>void; seed:number; state:GameState; onOpenSettings:()=>void
}) {
  const totalCardsOnTable = state.rows.reduce((a, r) => a + r.length, 0);
  return (
    <div className="w-full border-b border-slate-800 bg-slate-950/80 sticky top-0 z-20">
      <div className="max-w-6xl mx-auto px-3 py-2 flex items-center gap-3">
        <div className="font-semibold tracking-wide">Take 5</div>
        <div className="text-xs text-slate-400">seed {seed}</div>
        <div className="text-xs text-slate-400">table {totalCardsOnTable} cards</div>
        <div className="flex-1" />
        <button onClick={onNew} className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700">
          <RefreshCw className="w-4 h-4"/> New game
        </button>
        <button onClick={onOpenSettings} className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700">
          <Settings2 className="w-4 h-4"/> Settings
        </button>
      </div>
    </div>
  );
}

function Table({ rows }:{ rows:[Row, Row, Row, Row] }) {
  return (
    <div className="max-w-6xl w-full mx-auto grid gap-3">
      {[0,1,2,3].map(i => (
        <div key={i} className="bg-slate-900/60 rounded-2xl p-2 shadow-inner">
          <div className="flex items-center gap-2 mb-1 text-xs text-slate-400">
            <span>Row {i+1}</span>
            <span>•</span>
            <span>{rows[i].length} card{rows[i].length !== 1 ? "s" : ""}</span>
            <span>•</span>
            <span>{sumBulls(rows[i])}⟁</span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {rows[i].map((c) => (
              <motion.div key={c.id} layout initial={{scale:0.9,opacity:0}} animate={{scale:1,opacity:1}} exit={{opacity:0}}>
                <CardView card={c} />
              </motion.div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Hand({ cards, chosen, onChoose, disabled }:{
  cards:Card[]; chosen?:number; onChoose:(c:Card)=>void; disabled?:boolean
}){
  return (
    <div className="bg-slate-900/60 rounded-2xl p-2 shadow-inner">
      <div className="text-xs text-slate-400 mb-1">Your hand ({cards.length})</div>
      <div className="flex gap-2 overflow-x-auto">
        {cards.map(c => (
          <button key={c.id} onClick={() => !disabled && onChoose(c)} disabled={disabled}
            className={`relative ${chosen===c.id?"ring-2 ring-emerald-500":"ring-0"} rounded-xl`}>
            <CardView card={c} small />
            {chosen===c.id && (<div className="absolute -top-1 -right-1 bg-emerald-600 text-[10px] px-1.5 py-0.5 rounded-full">Selected</div>)}
          </button>
        ))}
      </div>
    </div>
  );
}

function CardView({ card, small }: { card: Card; small?: boolean }) {
  const theme = themeForCard(card.id);

  const W = small ? 72 : 92;
  const H = small ? 96 : 124;

  return (
    <div
      className="relative select-none rounded-xl shadow card-surface"
      style={{
        width: W,
        height: H,
        // paper / bevel look
        background:
          "radial-gradient(120% 100% at 50% 0%, #f7fafc 0%, #e6ecf1 60%, #dfe7ee 100%)",
        boxShadow:
          "inset 0 1px 0 rgba(255,255,255,.6), inset 0 -2px 4px rgba(0,0,0,.08), 0 2px 6px rgba(0,0,0,.25)",
        border: "1px solid rgba(0,0,0,.12)",
      }}
    >
      {/* subtle inner border */}
      <div
        className="absolute inset-0 rounded-xl"
        style={{ boxShadow: "inset 0 0 0 2px rgba(255,255,255,.55)" }}
      />

      {/* top band (accent) */}
      <div
        className="absolute left-1 right-1 rounded-md"
        style={{
          top: small ? 29 : 35,
          height: small ? 31 : 37,
          background: theme.band,
          boxShadow: "inset 0 0 0 1px rgba(0,0,0,.08)",
        }}
      />

      {/* corner indices */}
      <div className="absolute text-[10px] leading-none opacity-70" style={{ top: 6, left: 6 }}>
        {card.id}
      </div>
      <div
        className="absolute text-[10px] leading-none opacity-70"
        style={{ bottom: 6, right: 6, transform: "rotate(180deg)" }}
      >
        {card.id}
      </div>

      {/* big number with outline + shadow */}
      <div
        className="absolute font-extrabold tracking-tight number-face"
        style={{
          top: small ? 32 : 38,
          left: 0,
          right: 0,
          textAlign: "center",
          fontSize: small ? 26 : 32,
          color: theme.num,
          WebkitTextStroke: small ? "1px #fff" : "2px #fff",
          textShadow:
            "0 2px 0 rgba(0,0,0,.10), 0 3px 6px rgba(0,0,0,.15)",
          fontFamily:
            "'Bungee', 'Paytone One', system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        }}
      >
        {card.id}
      </div>

      {/* bulls row (penalty pips) */}
      <div
        className="absolute flex items-center justify-center gap-[2px]"
        style={{ bottom: small ? 6 : 8, left: 0, right: 0 }}
      >
        {Array.from({ length: card.bulls }).map((_, i) => (
          <BullIcon key={i} size={small ? 10 : 12} color={theme.bull} />
        ))}
      </div>
    </div>
  );
}

/** Generic bull-shaped icon (simple SVG, easy to swap for a traced official silhouette later) */
function BullIcon({ size = 12, color = "#1e3a8a" }: { size?: number; color?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={color}
      aria-hidden
    >
      {/* stylized horns + head */}
      <path d="M6 5c1 0 2 .4 2.7 1.1l1.5 1.5c.5-.2 1.1-.3 1.8-.3s1.3.1 1.8.3l1.5-1.5C16 5.4 17 5 18 5c.7 0 1.3.2 1.8.6l-1.2 1.8c-.2-.1-.4-.1-.6-.1-.6 0-1.1.2-1.5.6l-.8.8c.7.9 1.1 2 1.1 3.3 0 2.8-2.1 5-4.8 5.2v1.3h-2v-1.3C7.1 17.2 5 15 5 12.2c0-1.2.4-2.4 1.1-3.3l-.8-.8c-.4-.4-.9-.6-1.5-.6-.2 0-.4 0-.6.1L2 5.6C2.7 5.2 3.3 5 4 5c1 0 2 .4 2.7 1.1z" />
    </svg>
  );
}

/** Color / band logic to mimic the physical categories */
function themeForCard(n: number) {
  if (n === 55) {
    return {
      band:
        "repeating-linear-gradient(135deg,#fca5a5 0 6px,#ef4444 6px 12px)",
      num: "#991b1b",
      bull: "#7f1d1d",
    };
  }
  if (n % 11 === 0) {
    return {
      band:
        "linear-gradient(180deg,#ddd6fe 0%,#c4b5fd 100%)", // purple-ish
      num: "#3730a3",
      bull: "#312e81",
    };
  }
  if (n % 10 === 0) {
    return {
      band:
        "linear-gradient(180deg,#fed7aa 0%,#fdba74 100%)", // orange
      num: "#9a3412",
      bull: "#7c2d12",
    };
  }
  if (n % 5 === 0) {
    return {
      band:
        "linear-gradient(180deg,#fde68a 0%,#fbbf24 100%)", // yellow
      num: "#92400e",
      bull: "#78350f",
    };
  }
  // default blue
  return {
    band: "linear-gradient(180deg,#bfdbfe 0%,#93c5fd 100%)",
    num: "#1e3a8a",
    bull: "#1e3a8a",
  };
}

function RowChoice({ rows, onPick }:{ rows:[Row,Row,Row,Row]; onPick:(idx:number)=>void }) {
  return (
    <motion.div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-30 flex items-center justify-center p-4"
      initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
      <motion.div initial={{scale:0.95,opacity:0}} animate={{scale:1,opacity:1}} exit={{scale:0.95,opacity:0}}
        className="bg-slate-900 rounded-2xl shadow-2xl max-w-3xl w-full p-4">
        <div className="text-sm text-slate-300 mb-3">Your card is lower than all rows. Choose a row to take:</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[0,1,2,3].map(i => (
            <button key={i} onClick={() => onPick(i)}
              className="text-left bg-slate-800 hover:bg-slate-700 rounded-xl p-3">
              <div className="flex items-center justify-between text-xs text-slate-300 mb-2">
                <span>Row {i+1}</span>
                <span>{sumBulls(rows[i])}⟁</span>
              </div>
              <div className="flex gap-2 overflow-x-auto">
                {rows[i].map(c => <CardView key={c.id} card={c} small />)}
              </div>
            </button>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

function SettingsDialog({
  playersCount, seed, onClose, onApply
}:{
  playersCount:number;
  seed:number;
  onClose:()=>void;
  onApply:(players:number, seed:number)=>void;
}){
  const [localPlayers, setLocalPlayers] = useState(playersCount);
  const [localSeed, setLocalSeed] = useState(seed);
  return (
    <motion.div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-30 flex items-end md:items-center justify-center"
      initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
      <motion.div initial={{y:40,opacity:0}} animate={{y:0,opacity:1}} exit={{y:40,opacity:0}}
        className="bg-slate-900 w-full md:max-w-md rounded-t-2xl md:rounded-2xl shadow-2xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Settings2 className="w-4 h-4"/>
          <div className="font-semibold">Settings</div>
          <div className="flex-1" />
          <button onClick={onClose} className="p-1 rounded hover:bg-slate-800"><X className="w-4 h-4"/></button>
        </div>

        <div className="space-y-4">
          <label className="block text-sm">
            <div className="text-slate-300 mb-1">Players</div>
            <div className="flex items-center gap-2">
              <input
                type="range" min={2} max={10}
                value={localPlayers}
                onChange={e=>setLocalPlayers(parseInt(e.target.value))}
                className="w-full"
              />
              <span className="w-8 text-right text-slate-200">{localPlayers}</span>
            </div>
          </label>

          <label className="block text-sm">
            <div className="text-slate-300 mb-1">Seed</div>
            <input
              value={localSeed}
              onChange={e=>setLocalSeed(parseInt(e.target.value)||0)}
              className="w-full rounded-lg bg-slate-800 px-3 py-2"
            />
            <div className="mt-1 text-xs text-slate-400">Change for reproducible deals.</div>
          </label>
        </div>

        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            onClick={()=>{ setLocalSeed(Math.floor(1 + Math.random()*1e9)); }}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700"
          >
            Randomize seed
          </button>
          <button
            onClick={()=> onApply(localPlayers, localSeed)}
            className="px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white"
          >
            Start new game
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
