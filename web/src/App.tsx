import { type CSSProperties, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, RefreshCw, Play, Settings2, Lightbulb, CircleHelp, Trophy } from "lucide-react";
import { Bots, Coach } from "./engine/bots";
import { BULL_EARS, BULL_HEAD, BULL_HORN_L, BULL_HORN_R, BULL_VIEWBOX } from "./bullPaths";
import { RotateGate } from "./components/RotateGate";
import { type LayoutMode, useEscape, useLayoutMode, useNeedsRotate } from "./hooks";

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
  dealNumber: number; // 1-based; a match is deals until someone reaches 66
  totals: number[]; // bulls carried from previous deals, by player id
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
type BotStrategyId = "random" | "greedy" | "mc" | "neural";

const DIFFICULTIES: Array<{ id: BotStrategyId; label: string; blurb: string }> = [
  { id: "random", label: "Random", blurb: "Plays any card" },
  { id: "greedy", label: "Greedy", blurb: "Avoids obvious hits" },
  { id: "mc", label: "Search", blurb: "Monte-Carlo search (Rust/WASM)" },
  { id: "neural", label: "Neural", blurb: "Trained transformer policy (strongest)" },
];

function usesEngine(strategy: BotStrategyId): boolean {
  return strategy === "mc" || strategy === "neural";
}

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
function deal(players: number, seed: number, difficulty: BotStrategyId): GameState {
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
    strategy: i === 0 ? undefined : difficulty
  }));

  return {
    seed,
    dealNumber: 1,
    totals: Array(N).fill(0),
    turn: 0,
    players: playersState,
    rows: rowStarters as [Row, Row, Row, Row],
    phase: "choose",
    pendingPlacements: [],
    history: [`Game start (seed ${seed}). Rows: ${rowStarters.map(r => r[0].id).join(", ")}`],
  };
}

/** Take 5 is won by the FEWEST bulls, so the leaderboard head is the
 * winner. Second person needs "win", third person needs "wins". */
function matchResultLine(winner: PlayerState, bulls: number): string {
  return winner.id === 0
    ? `You win with ${bulls} bulls. Congratulations!`
    : `${winner.name} wins with ${bulls} bulls.`;
}

/** A second tap on the selected card commits the play. Taps inside this
 * window are treated as one fumbled tap rather than a commit: a genuine
 * double-tap bounce is <150 ms apart, a deliberate confirm is >250 ms. */
const COMMIT_GUARD_MS = 220;

/** localStorage key: the help sheet auto-opens once on a first visit,
 * which is where the tap-again gesture gets taught. */
const SEEN_HELP_KEY = "take5:seenHelp";

/** Card widths per layout mode.
 *
 * The compact values are CSS custom properties defined in index.css, so
 * they track the live viewport and the notch insets without JS ever
 * measuring anything. The var() fallbacks matter: data-layout is written
 * in a layout effect, and without them the very first paint would have
 * an invalid width. */
function cardWidths(mode: LayoutMode) {
  return mode === "compact"
    ? { board: "var(--cw-board, 54px)", hand: "var(--cw-hand, 60px)" }
    : { board: "92px", hand: "72px" };
}

// ---------- UI Root ----------
export default function Take5App() {
  const mode = useLayoutMode();
  const compact = mode === "compact";
  const needsRotate = useNeedsRotate();
  const cw = useMemo(() => cardWidths(mode), [mode]);

  const [playersCount, setPlayersCount] = useState(4);
  const [difficulty, setDifficulty] = useState<BotStrategyId>("neural");
  const [seed, setSeed] = useState<number>(() => Math.floor(1 + Math.random() * 1e9));
  const [state, setState] = useState<GameState>(() => deal(playersCount, seed, "neural"));
  const [showSettings, setShowSettings] = useState(false);
  const [showScores, setShowScores] = useState(false);
  // Auto-opens once per browser: with the top bar reduced to icons in
  // compact mode, this is where Coach and the tap-again-to-play gesture
  // get explained. localStorage throws in some privacy modes.
  const [showHelp, setShowHelp] = useState(() => {
    try {
      return !localStorage.getItem(SEEN_HELP_KEY);
    } catch {
      return false;
    }
  });
  const [botsLoading, setBotsLoading] = useState(false);
  /** Transient "somebody just took a row" effect. */
  const [takeFx, setTakeFx] = useState<
    { id: number; pid: PlayerId; row: number; bulls: number; name: string } | null
  >(null);
  const [coach, setCoach] = useState(false);
  const [coachHints, setCoachHints] = useState<Map<number, number> | null>(null);
  const [rowHints, setRowHints] = useState<Map<number, number> | null>(null);
  const [coachThinking, setCoachThinking] = useState(false);
  /** Both engine roles now live in workers, so the main thread holds
   *  only the client promises, and never instantiates WASM itself.
   *
   *  Storing the PROMISE rather than the resolved client, and assigning
   *  it synchronously with ??=, is what makes creation race-free:
   *  `x.current ??= await create()` awaits before assigning, so two
   *  concurrent callers can each build a worker and silently orphan one. */
  const botRunner = useRef<{ needNet: boolean; bots: Promise<Bots> } | null>(null);
  const coachBot = useRef<Promise<Coach> | null>(null);
  /** Synchronous re-entrancy lock for the commit path. `botsLoading`
   *  cannot do this job: it is React state, so the closure guarding a
   *  second call still sees the pre-setState value. */
  const playInFlight = useRef(false);
  /** Last hand tap, so a second tap on the selected card can commit. */
  const lastTap = useRef<{ id: number; t: number } | null>(null);
  /** Bumped whenever the deal is replaced. Opponent moves now cross a
   *  worker round-trip, and that async window is long enough for the user
   *  to start a new game or apply settings; without this, choices made
   *  for the old deal get applied to the freshly dealt state and then
   *  scheduled for resolution against the new rows. */
  const gameToken = useRef(0);

  function getBots(needNet: boolean): Promise<Bots> {
    const cur = botRunner.current;
    // Reuse unless we now need weights this worker was never given.
    if (cur && (cur.needNet || !needNet)) return cur.bots;
    cur?.bots.then((b) => b.dispose());
    const entry = { needNet, bots: Bots.create(needNet) };
    botRunner.current = entry;
    return entry.bots;
  }

  function getCoach(): Promise<Coach> {
    return (coachBot.current ??= Coach.create(Math.floor(Math.random() * 1e9)));
  }

  function disposeWorkers() {
    botRunner.current?.bots.then((b) => b.dispose());
    botRunner.current = null;
    coachBot.current?.then((c) => c.dispose());
    coachBot.current = null;
  }

  // Nothing else tears these down, so without this the workers, and the
  // 3.6 MB of weights each holds, outlive the app on unmount.
  useEffect(() => disposeWorkers, []);

  function startNewGame(pCount = playersCount, customSeed?: number, diff = difficulty) {
    gameToken.current++;
    const s = customSeed ?? Math.floor(1 + Math.random() * 1e9);
    // Free the per-seat bots (their seeds derive from the game seed) but
    // keep the worker, its compiled module and its weights.
    botRunner.current?.bots.then((b) => b.reset());
    setSeed(s);
    setState(deal(pCount, s, diff));
  }

  /** Ensure the opponent worker exists and holds a bot for every engine
   * seat. "Neural" opponents play the champion transformer's raw policy
   * It beats the previous net's search mode without any search. */
  async function ensureBots(current: GameState): Promise<Bots> {
    const stratOf = (p: PlayerState) => p.strategy ?? difficulty;
    const needNet = current.players.some((p) => !p.isHuman && stratOf(p) === "neural");
    const bots = await getBots(needNet);
    for (const p of current.players) {
      if (p.isHuman) continue;
      const strat = stratOf(p);
      if (usesEngine(strat)) {
        bots.ensureSeat(p.id, strat === "mc" ? "mc" : "attn", current.seed + p.id);
      }
    }
    return bots;
  }

  function seatView(current: GameState, pid: PlayerId) {
    return {
      player: pid,
      numPlayers: current.players.length,
      hand: current.players[pid].hand.map((c) => c.id),
      rows: current.rows.map((r) => r.map((c) => c.id)),
      penalties: current.players.map((p) => sumBulls(p.pen)),
      totals: current.players.map((p) => current.totals[p.id]),
      played: [
        ...current.rows.flat().map((c) => c.id),
        ...current.players.flatMap((p) => p.pen.map((c) => c.id)),
      ],
      turn: current.turn,
    };
  }

  // Coach mode: score the human's hand with the champion net's
  // belief-guided analyze, the same brain as the strongest opponents,
  // running in a web worker so the search never blocks the UI.
  useEffect(() => {
    if (!coach || state.phase !== "choose") {
      setCoachHints(null);
      return;
    }
    let cancelled = false;
    (async () => {
      setCoachThinking(true);
      try {
        const scores = await (await getCoach()).analyze(seatView(state, 0));
        if (!cancelled) setCoachHints(scores);
      } catch (e) {
        console.error("coach failed", e);
        if (!cancelled) setCoach(false);
      } finally {
        if (!cancelled) setCoachThinking(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coach, state.phase, state.turn]);

  // Coach mode, forced row choice: score the four rows with the same
  // brain (immediate bulls + value of the position after the row
  // restarts with your card).
  useEffect(() => {
    if (!coach || state.phase !== "needRowChoice" || state.needRowChoiceFor !== 0) {
      setRowHints(null);
      return;
    }
    const step = state.pendingPlacements[0];
    if (!step || step.pid !== 0) return;
    let cancelled = false;
    (async () => {
      setCoachThinking(true);
      try {
        const scores = await (await getCoach()).analyzeRows(seatView(state, 0), step.card.id);
        if (!cancelled) setRowHints(scores);
      } catch (e) {
        console.error("row coach failed", e);
      } finally {
        if (!cancelled) setCoachThinking(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coach, state.phase]);

  // --- Actions ---
  /** Tap once to select; tap the SAME, already-selected card again to
   * play it. The commit tap must land on the card that is already
   * selected, so a mis-tap can only ever re-select. That is the
   * confirmation, and why there is no confirm dialog. */
  function onChooseCard(card: Card) {
    if (state.phase !== "choose" || playInFlight.current || botsLoading) return;
    const you = state.players[0];
    if (!you.hand.find(c => c.id === card.id)) return;

    const now = performance.now();
    const prev = lastTap.current;
    const isSelected = you.chosen?.id === card.id;

    if (isSelected && prev?.id === card.id && now - prev.t >= COMMIT_GUARD_MS) {
      lastTap.current = null;
      void onPlaySelected(card);
      return;
    }

    // Otherwise (re)select. A sub-guard bounce lands here and is a no-op,
    // because the card is already the selected one.
    lastTap.current = { id: card.id, t: now };
    if (!isSelected) {
      setState(prev => ({ ...prev, players: prev.players.map(p => p.id === 0 ? { ...p, chosen: card } : p) }));
    }
  }

  async function onPlaySelected(explicit?: Card) {
    if (playInFlight.current || state.phase !== "choose") return;
    const chosen = explicit ?? state.players[0].chosen;
    if (!chosen) return;

    // Snapshot the deal identity before any await; see gameToken.
    const token = gameToken.current;

    playInFlight.current = true;
    try {
      // Bots pick. Engine-backed strategies run in a worker, so the ~15-35 ms
      // per move (up to 9 seats) no longer blocks the UI; first use fetches
      // the module and, for "neural", the net weights.
      const engineSeats = state.players.filter(
        p => !p.isHuman && usesEngine(p.strategy ?? difficulty),
      );
      const picks = new Map<PlayerId, number>();
      if (engineSeats.length) {
        setBotsLoading(true);
        try {
          const bots = await ensureBots(state);
          const chosenIds = await Promise.all(
            engineSeats.map(p => bots.chooseCard(p.id, seatView(state, p.id))),
          );
          engineSeats.forEach((p, i) => picks.set(p.id, chosenIds[i]));
        } finally {
          setBotsLoading(false);
        }
      }

      // The deal was replaced while the worker was thinking: these picks
      // belong to a game that no longer exists.
      if (token !== gameToken.current) return;

      const withChoices = state.players.map(p => {
        // Use the local `chosen`, not p.chosen: it is authoritative for
        // the tap-again path and removes a stale-state dependency.
        if (p.isHuman) return { ...p, chosen };
        const strat = p.strategy ?? difficulty;
        const cardId = picks.get(p.id);
        if (cardId !== undefined) {
          const card = p.hand.find(c => c.id === cardId);
          if (card) return { ...p, chosen: card };
        }
        const c = botChooseCard(state, p.id, strat === "mc" || strat === "neural" ? "greedy" : strat);
        return { ...p, chosen: c };
      });

      setState(prev =>
        token === gameToken.current
          ? { ...prev, players: withChoices, phase: "reveal" }
          : prev,
      );

      // After a short reveal, resolve in ascending order
      setTimeout(() => {
        if (token !== gameToken.current) return;
        setState(prev => {
          if (prev.phase !== "reveal") return prev;
          const placements = withChoices.map(p => ({ pid: p.id, card: p.chosen! }))
            .sort((a, b) => a.card.id - b.card.id);
          return { ...prev, pendingPlacements: placements, phase: "resolve" };
        });
      }, 450);
    } finally {
      playInFlight.current = false;
    }
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
      const log = taken.length ? [`${nameFor(state, step.pid)} takes ${taken.length} cards (${sumBulls(taken)} bulls)`] : [];
      const t = setTimeout(() => {
        // Fire with the state change so the flash lands on the frame the
        // row actually empties.
        if (taken.length) {
          flashTake(step.pid, result.placedRow ?? 0, sumBulls(taken), nameFor(state, step.pid));
        }
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
        const log = `${nameFor(state, step.pid)} chooses row ${botIdx + 1} and takes ${sumBulls(taken)} bulls`;
        const t = setTimeout(() => {
          if (taken.length) {
            flashTake(step.pid, botIdx, sumBulls(taken), nameFor(state, step.pid));
          }
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
    // Remove the played card from the hand here too, because every other
    // resolution path does it before placing (a card that stayed in hand
    // after taking a row could be played twice).
    const playersUpdated = state.players.map(p => p.id === 0
      ? {
          ...p,
          hand: p.hand.filter(c => c.id !== step.card.id),
          pen: taken.length ? [...p.pen, ...taken] : p.pen,
        }
      : p);

    if (taken.length) flashTake(0, idx, sumBulls(taken), "You");

    setState(prev => ({
      ...prev,
      players: playersUpdated,
      rows,
      pendingPlacements: prev.pendingPlacements.slice(1),
      needRowChoiceFor: undefined,
      phase: "resolve",
      history: [...prev.history, `You choose row ${idx + 1} and take ${sumBulls(taken)} bulls`]
    }));
  }

  // Deal fresh hands, carrying match totals forward (first to 66 ends it).
  function nextDeal() {
    gameToken.current++;
    const s = Math.floor(1 + Math.random() * 1e9);
    setSeed(s);
    setState(prev => {
      const fresh = deal(prev.players.length, s, difficulty);
      return {
        ...fresh,
        dealNumber: prev.dealNumber + 1,
        totals: prev.players.map(p => prev.totals[p.id] + sumBulls(p.pen)),
      };
    });
  }

  const takeFxId = useRef(0);
  const takeFxTimer = useRef<number | null>(null);

  /** Announce a row being taken. Previously the six cards just vanished
   * with only a line in the (invisible) history log, so there was no way
   * to tell what had happened or what it cost. */
  function flashTake(pid: PlayerId, row: number, bulls: number, name: string) {
    if (bulls <= 0) return;
    const id = ++takeFxId.current;
    setTakeFx({ id, pid, row, bulls, name });
    if (takeFxTimer.current !== null) clearTimeout(takeFxTimer.current);
    takeFxTimer.current = window.setTimeout(
      () => setTakeFx(f => (f && f.id === id ? null : f)),
      1700,
    );
  }
  useEffect(() => () => {
    if (takeFxTimer.current !== null) clearTimeout(takeFxTimer.current);
  }, []);

  function closeHelp() {
    setShowHelp(false);
    try {
      localStorage.setItem(SEEN_HELP_KEY, "1");
    } catch {
      /* private mode: just don't remember */
    }
  }

  // The compact layout has no room for a persistent scoreboard, so the
  // end of a deal opens it instead of rendering a banner inline.
  useEffect(() => {
    if (compact && state.phase === "gameOver") setShowScores(true);
  }, [compact, state.phase]);

  // --- Derived ---
  const you = state.players[0];
  const score = (p: PlayerState) => sumBulls(p.pen);
  const matchTotal = (p: PlayerState) => state.totals[p.id] + sumBulls(p.pen);
  const matchOver = state.phase === "gameOver" && state.players.some(p => matchTotal(p) >= 66);
  const leaderboard = useMemo(
    () => [...state.players].sort((a, b) => (state.totals[a.id] + score(a)) - (state.totals[b.id] + score(b))),
    [state.players, state.totals]
  );

  const rowChoiceOpen = state.phase === "needRowChoice" && state.needRowChoiceFor === 0;

  // Overlays shared by both layouts. RotateGate is deliberately an
  // overlay rather than an early return: unmounting the tree would
  // replay every framer-motion enter animation on rotate-back and
  // re-fire the coach effects, spawning a worker analyze on every flip.
  const overlays = (
    <>
      <AnimatePresence>
        {rowChoiceOpen && (
          <RowChoice
            rows={state.rows}
            onPick={onChooseRow}
            hints={coach ? rowHints : null}
            thinking={coachThinking}
            compact={compact}
            cardW={compact ? cw.board : "72px"}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showSettings && (
          <SettingsDialog
            playersCount={playersCount}
            seed={seed}
            difficulty={difficulty}
            onClose={() => setShowSettings(false)}
            onApply={(pc, sd, diff) => {
              setPlayersCount(pc);
              setDifficulty(diff);
              startNewGame(pc, sd, diff);
              setShowSettings(false);
            }}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showHelp && <HelpSheet onClose={closeHelp} />}
      </AnimatePresence>

      {/* Says who took what. Positioned over the board and click-through
          so it never intercepts a tap on a card. */}
      <AnimatePresence>
        {takeFx && (
          <motion.div
            key={takeFx.id}
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.22 }}
            className="pointer-events-none fixed left-1/2 top-20 compact:top-11 z-40 -translate-x-1/2"
          >
            <div
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-semibold shadow-xl ring-1 ${
                takeFx.pid === 0
                  ? "bg-red-600 text-white ring-red-400/60"
                  : "bg-slate-800 text-slate-100 ring-slate-600"
              }`}
            >
              {/* Icon leads, so it decorates rather than reading as a
                  duplicate of the word ("takes 1 bull bull"). */}
              <BullIcon size={14} color="currentColor" />
              <span className="tabular-nums">
                {takeFx.pid === 0 ? "You take" : `${takeFx.name} takes`} {takeFx.bulls}{" "}
                {takeFx.bulls === 1 ? "bull" : "bulls"}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {needsRotate && <RotateGate />}
    </>
  );

  if (compact) {
    return (
      <div className="compact-shell w-full bg-slate-950 text-slate-100 flex flex-col">
        <TopBar
          compact
          onNew={() => startNewGame()}
          seed={seed}
          state={state}
          onOpenSettings={() => setShowSettings(true)}
          onOpenScores={() => setShowScores(true)}
          onOpenHelp={() => setShowHelp(true)}
          coach={coach}
          coachThinking={coachThinking}
          onToggleCoach={() => setCoach(c => !c)}
        />

        <div className="flex-1 min-h-0 flex flex-col gap-1.5 mt-1.5">
          {/* key={mode}: without a remount, rotating makes framer-motion
              animate every board card from the stacked positions to the
              2x2 ones across a simultaneously resizing viewport. */}
          <Table key={mode} rows={state.rows} compact cardW={cw.board} take={takeFx} />

          <div className="flex gap-2 items-stretch shrink-0">
            <Hand
              compact
              cardW={cw.hand}
              cards={you.hand}
              chosen={you.chosen?.id}
              onChoose={onChooseCard}
              disabled={state.phase !== "choose"}
              hints={coach ? coachHints : null}
            />
            {/* An 80px column beside the hand rather than a row above it:
                it costs zero vertical pixels, and the bottom-right corner
                is the most reachable point for a landscape thumb grip. */}
            <button
              onClick={() => onPlaySelected()}
              disabled={state.phase !== "choose" || !you.chosen || botsLoading}
              className="w-20 shrink-0 rounded-xl bg-emerald-600 active:bg-emerald-500
                         disabled:opacity-30 disabled:active:bg-emerald-600
                         flex flex-col items-center justify-center gap-1 text-xs font-semibold"
            >
              <Play className="w-6 h-6" aria-hidden />
              {botsLoading ? "Loading…" : "Play"}
            </button>
          </div>
        </div>

        <AnimatePresence>
          {showScores && (
            <ScoresSheet
              state={state}
              leaderboard={leaderboard}
              matchOver={matchOver}
              onNextDeal={() => { setShowScores(false); nextDeal(); }}
              onNewMatch={() => { setShowScores(false); startNewGame(); }}
              onClose={() => setShowScores(false)}
            />
          )}
        </AnimatePresence>

        {overlays}
      </div>
    );
  }

  return (
    <div className="min-h-svh w-full bg-slate-950 text-slate-100 flex flex-col">
      {/* display:none while the rotate gate is up, rather than unmounting.
          On a phone in portrait the desktop layout does not fit and
          overflows to ~820px, and a mobile browser grows the LAYOUT
          viewport to the content width, which resizes `fixed inset-0`
          overlays with it and zooms the page out. Hiding removes the
          overflow; `contents` keeps the flex layout identical when
          visible, and the subtree stays mounted so framer-motion does not
          replay its enter animations on rotate-back. */}
      <div className={needsRotate ? "hidden" : "contents"}>
      <TopBar
        compact={false}
        onNew={() => startNewGame()}
        seed={seed}
        state={state}
        onOpenSettings={() => setShowSettings(true)}
        onOpenScores={() => setShowScores(true)}
        onOpenHelp={() => setShowHelp(true)}
        coach={coach}
        coachThinking={coachThinking}
        onToggleCoach={() => setCoach(c => !c)}
      />

      <div className="flex-1 grid grid-rows-[auto_1fr_auto] gap-3 px-gutter pb-gutter">
        {/* Status line */}
        <div className="mx-auto mt-2 text-sm text-slate-300 flex items-center gap-3">
          <span className="opacity-80">Deal {state.dealNumber}</span>
          <span className="opacity-50">•</span>
          <span className="opacity-80">Turn {Math.min(state.turn + 1, 10)} / 10</span>
          <span className="opacity-50">•</span>
          <span className="opacity-60">match ends at 66</span>
          <span className="opacity-50">•</span>
          <span className="opacity-80 capitalize">{state.phase.replace(/([a-z])([A-Z])/g, "$1 $2")}</span>
        </div>

        {/* Table */}
        <Table rows={state.rows} compact={false} cardW={cw.board} take={takeFx} />

        {/* Controls + Hand */}
        <div className="max-w-6xl w-full mx-auto">
          {state.phase === "gameOver" && (
            <div className="mb-3 rounded-2xl bg-slate-900/80 border border-slate-700 p-3 flex items-center justify-between">
              <div className="text-sm">
                {matchOver ? (
                  <>
                    <span className="font-semibold text-amber-400">Match over!</span>{" "}
                    {matchResultLine(leaderboard[0], matchTotal(leaderboard[0]))}
                  </>
                ) : (
                  <>Deal {state.dealNumber} finished. You took {score(you)} bulls (total {matchTotal(you)}).</>
                )}
              </div>
              {matchOver ? (
                <button onClick={() => startNewGame()} className="px-4 py-2 rounded-2xl bg-emerald-600 hover:bg-emerald-500">
                  New match
                </button>
              ) : (
                <button onClick={nextDeal} className="px-4 py-2 rounded-2xl bg-emerald-600 hover:bg-emerald-500">
                  Next deal
                </button>
              )}
            </div>
          )}
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-slate-300">
              This deal: <b>{score(you)}</b> · Match: <b>{matchTotal(you)}</b> / 66
            </div>
            <button
              onClick={() => onPlaySelected()}
              className={`px-4 py-2 rounded-2xl shadow-md bg-emerald-600 hover:bg-emerald-500 active:scale-[.98] transition disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2`}
              disabled={state.phase !== "choose" || !you.chosen || botsLoading}
            >
              <Play className="w-4 h-4" /> {botsLoading ? "Loading bot…" : "Play selected"}
            </button>
          </div>

          <Hand
            compact={false}
            cardW={cw.hand}
            cards={you.hand}
            chosen={you.chosen?.id}
            onChoose={onChooseCard}
            disabled={state.phase !== "choose"}
            hints={coach ? coachHints : null}
          />

          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-slate-400">
            {leaderboard.map(p => (
              <div key={p.id} className="flex items-center justify-between bg-slate-900/60 rounded-xl px-3 py-2">
                <span>{p.name}</span>
                <span className="inline-flex items-center gap-1">
                  <Bulls n={sumBulls(p.pen)} />
                  <span className="ml-1 opacity-60">match {state.totals[p.id] + sumBulls(p.pen)}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      </div>

      {overlays}
    </div>
  );
}

// ---------- Pieces ----------
function TopBar({
  onNew, seed, state, onOpenSettings, onOpenScores, onOpenHelp,
  coach, coachThinking, onToggleCoach, compact,
}:{
  onNew:()=>void; seed:number; state:GameState; onOpenSettings:()=>void;
  onOpenScores:()=>void; onOpenHelp:()=>void;
  coach:boolean; coachThinking:boolean; onToggleCoach:()=>void; compact:boolean;
}) {
  const totalCardsOnTable = state.rows.reduce((a, r) => a + r.length, 0);
  const dealBulls = sumBulls(state.players[0].pen);
  const matchBulls = state.totals[0] + dealBulls;

  if (compact) {
    // 36px tall. The buttons paint at ~28px but each carries a centred
    // 44x44 hit area via .tap-target, so the touch guideline is met
    // without the bar costing 44px of the ~330px viewport.
    return (
      <div className="h-9 shrink-0 flex items-center gap-2 px-1 border-b border-slate-800 text-[11px] text-slate-400">
        <span className="font-semibold text-slate-200 text-xs">Take 5</span>
        <span className="tabular-nums">D{state.dealNumber} · T{Math.min(state.turn + 1, 10)}/10</span>
        <span className="capitalize opacity-70 truncate">
          {state.phase.replace(/([a-z])([A-Z])/g, "$1 $2")}
        </span>
        <div className="flex-1" />

        <button
          onClick={onOpenScores}
          aria-label="Scores"
          className="tap-target px-2 py-1 rounded-lg bg-slate-800 active:bg-slate-700 text-slate-200 inline-flex items-center gap-1 tabular-nums"
        >
          <Bulls n={dealBulls} />
          <span className="opacity-70">/{matchBulls}</span>
        </button>
        <button
          onClick={onToggleCoach}
          aria-label="Coach: the trained bot scores your options"
          aria-pressed={coach}
          className={`tap-target p-1.5 rounded-lg ${coach ? "bg-amber-600" : "bg-slate-800"}`}
        >
          <Lightbulb className={`w-4 h-4 ${coachThinking ? "animate-pulse text-amber-200" : ""}`} aria-hidden />
        </button>
        <button onClick={onNew} aria-label="New game" className="tap-target p-1.5 rounded-lg bg-slate-800">
          <RefreshCw className="w-4 h-4" aria-hidden />
        </button>
        <button onClick={onOpenHelp} aria-label="How to play" className="tap-target p-1.5 rounded-lg bg-slate-800">
          <CircleHelp className="w-4 h-4" aria-hidden />
        </button>
        <button onClick={onOpenSettings} aria-label="Settings" className="tap-target p-1.5 rounded-lg bg-slate-800">
          <Settings2 className="w-4 h-4" aria-hidden />
        </button>
      </div>
    );
  }

  return (
    <div className="w-full border-b border-slate-800 bg-slate-950/80 sticky top-0 z-20">
      {/* pb-2 + an explicit pt rather than py-2, so the safe-area max()
          cannot collide with a padding-block shorthand. */}
      <div className="max-w-6xl mx-auto px-gutter pb-2 pt-[max(0.5rem,var(--sa-t))] flex items-center gap-3">
        <div className="font-semibold tracking-wide">Take 5</div>
        <div className="text-xs text-slate-400">seed {seed}</div>
        <div className="text-xs text-slate-400">table {totalCardsOnTable} cards</div>
        <div className="flex-1" />
        <button
          onClick={onToggleCoach}
          aria-label="Coach: the trained bot scores your options"
          aria-pressed={coach}
          className={`tap-target flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl ${coach ? "bg-amber-600 hover:bg-amber-500" : "bg-slate-800 hover:bg-slate-700"}`}
        >
          <Lightbulb className="w-4 h-4" aria-hidden/> {coachThinking ? "Thinking…" : "Coach"}
        </button>
        <button onClick={onOpenHelp} aria-label="How to play" className="tap-target flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700">
          <CircleHelp className="w-4 h-4" aria-hidden/> Help
        </button>
        <button onClick={onNew} className="tap-target flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700">
          <RefreshCw className="w-4 h-4" aria-hidden/> New game
        </button>
        <button onClick={onOpenSettings} className="tap-target flex items-center gap-2 text-sm px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700">
          <Settings2 className="w-4 h-4" aria-hidden/> Settings
        </button>
      </div>
    </div>
  );
}

function Table({ rows, compact, cardW, take }: {
  rows:[Row, Row, Row, Row]; compact: boolean; cardW: string;
  take?: { id: number; row: number; bulls: number; pid: PlayerId } | null;
}) {
  // Keyed on take.id so a second take on the same row restarts the CSS
  // animation; remounting this tiny overlay is cheap, whereas keying the
  // panel itself would remount the cards and kill their layout animation.
  const takeFxFor = (i: number) =>
    take && take.row === i ? (
      <div key={take.id} className="pointer-events-none absolute inset-0 z-10 rounded-xl animate-take-flash" />
    ) : null;

  const takeBadgeFor = (i: number) =>
    take && take.row === i ? (
      <motion.div
        key={`badge-${take.id}`}
        initial={{ opacity: 0, y: 8, scale: 0.8 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="pointer-events-none absolute left-1/2 top-0.5 z-20 -translate-x-1/2 inline-flex items-center gap-1
                   rounded-full bg-red-600 px-2 py-0.5 text-[11px] font-bold text-white shadow-lg"
      >
        +{take.bulls}
        <BullIcon size={10} color="#fff" />
      </motion.div>
    ) : null;

  const cardsIn = (i: number) =>
    rows[i].map((c) => (
      <motion.div key={c.id} layout initial={{scale:0.9,opacity:0}} animate={{scale:1,opacity:1}} exit={{opacity:0}}>
        <CardView card={c} w={cardW} />
      </motion.div>
    ));

  if (compact) {
    return (
      // min-h-0 is what lets this grid shrink inside the flex shell
      // instead of overflowing it.
      <div className="flex-1 min-h-0 grid grid-cols-2 grid-rows-2 gap-2">
        {[0,1,2,3].map(i => (
          <div
            key={i}
            className="relative bg-slate-900/60 rounded-xl p-1 shadow-inner flex items-center gap-1 min-w-0 min-h-0 overflow-hidden"
          >
            {takeFxFor(i)}
            {takeBadgeFor(i)}
            {/* The label lives in a narrow gutter rather than a header
                line above the cards; that is what buys the vertical room
                the 2x2 board needs. */}
            <div className="w-[34px] shrink-0 flex flex-col items-center justify-center gap-px text-[10px] leading-none text-slate-400">
              <span className="font-semibold text-slate-300">R{i+1}</span>
              <Bulls n={sumBulls(rows[i])} />
            </div>
            <div className="flex gap-1 min-w-0 flex-1 items-center hscroll">
              {cardsIn(i)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="max-w-6xl w-full mx-auto grid gap-3">
      {[0,1,2,3].map(i => (
        <div key={i} className="relative bg-slate-900/60 rounded-2xl p-2 shadow-inner">
          {takeFxFor(i)}
          {takeBadgeFor(i)}
          <div className="flex items-center gap-2 mb-1 text-xs text-slate-400">
            <span>Row {i+1}</span>
            <span>•</span>
            <span>{rows[i].length} card{rows[i].length !== 1 ? "s" : ""}</span>
            <span>•</span>
            <Bulls n={sumBulls(rows[i])} />
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {cardsIn(i)}
          </div>
        </div>
      ))}
    </div>
  );
}

function Hand({ cards, chosen, onChoose, disabled, hints, compact, cardW }:{
  cards:Card[]; chosen?:number; onChoose:(c:Card)=>void; disabled?:boolean;
  hints?:Map<number, number> | null; compact: boolean; cardW: string;
}){
  // Coach badges: ★ marks the bot's pick; every other card shows +n, the
  // extra bulls that play is expected to cost compared with the ★ play.
  const best = hints && hints.size ? Math.max(...hints.values()) : null;

  const badgeFor = (c: Card) => {
    const score = hints?.get(c.id);
    // Positive cost in bulls vs the bot's pick (0 for the pick itself).
    const cost = best !== null && score !== undefined ? best - score : null;
    const isBest = cost !== null && cost < 1e-6;
    const costClass =
      cost === null || isBest
        ? "bg-amber-500 text-slate-950"
        : cost < 1
          ? "bg-emerald-700 text-emerald-100"
          : cost < 3
            ? "bg-amber-700 text-amber-100"
            : "bg-red-700 text-red-100";
    return { cost, isBest, costClass };
  };

  if (compact) {
    return (
      // pt-3 gives 12px of headroom: 8 for the coach badge at -top-2 and
      // 4 for the selected card's lift. That 12 is the hand term in the
      // --cw-board budget; keep them in step.
      <div className="flex-1 min-w-0 bg-slate-900/60 rounded-xl px-1 pt-3 pb-1 shadow-inner">
        {/* No overflow-x-auto: overflow-x:auto forces overflow-y:auto,
            which would clip the coach badge. The card width is derived so
            that ten of them always fit. */}
        <div className="flex gap-1.5 justify-center items-end">
          {cards.map(c => {
            const { cost, isBest, costClass } = badgeFor(c);
            return (
              <button
                key={c.id}
                onClick={() => !disabled && onChoose(c)}
                disabled={disabled}
                aria-label={`Card ${c.id}${chosen === c.id ? ", selected, tap again to play" : ""}`}
                className={`relative rounded-lg shrink-0 transition-transform duration-100 ${
                  chosen === c.id
                    ? "ring-2 ring-emerald-400 -translate-y-1"
                    : isBest
                      ? "ring-2 ring-amber-500"
                      : ""
                }`}
              >
                <CardView card={c} w={cardW} />
                {cost !== null && (
                  <div className={`absolute -top-2 left-1/2 -translate-x-1/2 z-10 text-[9px] leading-none px-1 py-0.5 rounded-full ${costClass}`}>
                    {isBest ? "★" : `+${cost.toFixed(1)}`}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/60 rounded-2xl p-2 shadow-inner">
      <div className="text-xs text-slate-400 mb-1">
        Your hand ({cards.length}){hints && best !== null && (
          <span className="ml-2 text-amber-400">
            coach: ★ = best play · +n = extra bulls that play risks
          </span>
        )}
      </div>
      <div className="flex gap-2 overflow-x-auto pt-2">
        {cards.map(c => {
          const { cost, isBest, costClass } = badgeFor(c);
          return (
            <button key={c.id} onClick={() => !disabled && onChoose(c)} disabled={disabled}
              className={`relative ${chosen===c.id?"ring-2 ring-emerald-500":isBest?"ring-2 ring-amber-500":"ring-0"} rounded-xl`}>
              <CardView card={c} w={cardW} />
              {chosen===c.id && (
                <div className="absolute -top-1 -right-1 bg-emerald-600 text-[10px] px-1.5 py-0.5 rounded-full z-10 whitespace-nowrap">
                  {/* On a touch pointer the second tap plays the card, so
                      say so; with a mouse the Play button is the path. */}
                  <span className="touch:hidden">Selected</span>
                  <span className="hidden touch:inline">Tap to play</span>
                </div>
              )}
              {cost !== null && (
                <div className={`absolute -top-2 left-1/2 -translate-x-1/2 text-[10px] px-1.5 py-0.5 rounded-full ${costClass}`}>
                  {isBest ? "★" : `+${cost.toFixed(1)}`}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Card geometry.
 *
 * Every metric is an affine function of the card width W (a*W + b),
 * fitted through the two tiers the deck shipped with (W=72 and W=92) so
 * both reproduce their exact pixel values, and any other width, including
 * a fluid min()/clamp() against the viewport, interpolates smoothly.
 *
 * A single proportional scale factor cannot do this: the shipped tiers are
 * not proportional. The corner index sits 5px from the edge in BOTH tiers,
 * and its font is 9 vs 10 rather than the 7.8 vs 10 a ratio would give, so
 * scaling would visibly shrink desktop hand-card text.
 *
 * Emitted as CSS calc() against --cw so the width may itself be a CSS
 * expression, which is what lets the compact layout size cards from the
 * live viewport with no ResizeObserver. See --cw-hand / --cw-board in
 * index.css.
 */
const px = (a: number, b: number) =>
  `calc(${a} * var(--cw) ${b < 0 ? "-" : "+"} ${Math.abs(b)}px)`;

const CARD = {
  h: px(1.4, -4.8),
  pipTop: px(0.05, 0.4),
  pipSize: px(0.1, 0.8),
  // Not cosmetic: card 55 carries 7 pips, which at W=54 would need
  // 7*6.2 + 6*2 = 55.4px inside a 54px card and clip. Exactly 2 at both
  // anchors, tapering below W=72.
  pipGap: `min(2px, ${px(0.06, -2.32)})`,
  cornerTop: px(0.15, 3.2),
  cornerBot: px(0.05, 1.4),
  // The two floors engage strictly below W=72, so neither can perturb a
  // shipped tier. They keep the small text legible and keep the number's
  // outline visible against the busy starburst behind it.
  cornerFont: `max(8px, ${px(0.05, 5.4)})`,
  numFont: px(0.4, -3.8),
  numStroke: `max(0.7px, ${px(0.03, -0.96)})`,
} as const;

/** @param w any CSS length for the card width (default: the table tier) */
function CardView({ card, w = "92px" }: { card: Card; w?: string }) {
  const t = themeForCard(card.id);

  return (
    <div
      className="relative select-none rounded-xl card-surface overflow-hidden"
      style={{
        ...({ "--cw": w } as CSSProperties),
        width: "var(--cw)",
        height: CARD.h,
        background: t.face,
        border: "1px solid rgba(0,0,0,.2)",
        boxShadow:
          "inset 0 1px 0 rgba(255,255,255,.5), inset 0 -2px 4px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.3)",
      }}
    >
      {/* silver starburst behind the bull, like the printed cards */}
      <div
        className="absolute inset-0"
        style={{
          background: `repeating-conic-gradient(from 0deg at 50% 54%, ${t.burstA} 0deg 5deg, ${t.burstB} 5deg 10deg)`,
          opacity: 0.5,
        }}
      />
      {/* inner frame */}
      <div
        className="absolute inset-[3px] rounded-lg"
        style={{ boxShadow: `inset 0 0 0 1.5px ${t.frame}` }}
      />

      {/* bull-count pips along the top edge */}
      <div
        className="absolute flex items-center justify-center"
        style={{ top: CARD.pipTop, left: 0, right: 0, gap: CARD.pipGap }}
      >
        {Array.from({ length: card.bulls }).map((_, i) => (
          <BullIcon key={i} size={CARD.pipSize} color={t.pip} />
        ))}
      </div>

      {/* corner indices */}
      <div
        className="absolute font-bold leading-none"
        style={{ top: CARD.cornerTop, left: 5, fontSize: CARD.cornerFont, color: t.corner }}
      >
        {card.id}
      </div>
      <div
        className="absolute font-bold leading-none"
        style={{
          bottom: CARD.cornerBot,
          right: 5,
          fontSize: CARD.cornerFont,
          color: t.corner,
          transform: "rotate(180deg)",
        }}
      >
        {card.id}
      </div>

      {/* the bull */}
      <div
        className="absolute left-1/2 -translate-x-1/2"
        style={{ top: "17%", width: "80%", height: "70%" }}
      >
        <BullIcon color={t.bull} />
      </div>

      {/* big number over the bull's forehead */}
      <div
        className="absolute font-extrabold tracking-tight number-face"
        style={{
          top: "50%",
          left: 0,
          right: 0,
          transform: "translateY(-52%)",
          textAlign: "center",
          fontSize: CARD.numFont,
          color: t.num,
          // Longhands, not the WebkitTextStroke shorthand: shorthand
          // parsing of `calc(...) #hex` is fragile across engines.
          WebkitTextStrokeWidth: CARD.numStroke,
          WebkitTextStrokeColor: t.numStroke,
          textShadow: "0 2px 3px rgba(0,0,0,.35)",
          fontFamily:
            "'Bungee', system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        }}
      >
        {card.id}
      </div>
    </div>
  );
}

/** Bull-head silhouette (front view, official-card style: broad lyre
 * horns, side ears, tapering face). One fill color; scales from 8 px
 * pips to the full card face. Pass no size for 100% width/height.
 *
 * `size` takes a CSS length as well as a number, and is applied via
 * `style` because SVG width/height ATTRIBUTES do not accept calc().
 * The geometry lives in bullPaths.js, shared with scripts/gen-icons.mjs
 * so the app icon can never drift from the card art. */
function BullIcon({ size, color = "#1e3a8a" }: { size?: number | string; color?: string }) {
  return (
    <svg
      viewBox={BULL_VIEWBOX}
      fill={color}
      aria-hidden
      style={{ width: size ?? "100%", height: size ?? "100%", flex: "none" }}
    >
      <path d={BULL_HORN_L} />
      <path d={BULL_HORN_R} />
      {BULL_EARS.map((e, i) => (
        <ellipse
          key={i}
          cx={e.cx}
          cy={e.cy}
          rx={e.rx}
          ry={e.ry}
          transform={`rotate(${e.rotate} ${e.cx} ${e.cy})`}
        />
      ))}
      <path d={BULL_HEAD} />
    </svg>
  );
}

/** Inline bull count: the number plus a small bull icon (replaces the
 * old, cryptic triangle glyph). Inherits the surrounding text color. */
function Bulls({ n }: { n: number }) {
  return (
    <span className="inline-flex items-center gap-[3px] whitespace-nowrap align-middle">
      {n}
      <BullIcon size={11} color="currentColor" />
    </span>
  );
}

/** Face theming that follows the official deck: silver for 1 bull,
 * blue for 2 (ends in 5), amber for 3 (ends in 0), red for 5
 * (multiples of 11), purple for the 7-bull 55. */
function themeForCard(n: number) {
  if (n === 55) {
    return {
      face: "linear-gradient(180deg,#8b5cf6,#6d28d9)",
      burstA: "rgba(255,255,255,.28)", burstB: "rgba(76,29,149,.35)",
      frame: "rgba(255,255,255,.35)",
      bull: "#2e1065", num: "#ffffff", numStroke: "#2e1065",
      pip: "#fbbf24", corner: "#ede9fe",
    };
  }
  if (n % 11 === 0) {
    return {
      face: "linear-gradient(180deg,#ef4444,#b91c1c)",
      burstA: "rgba(255,255,255,.28)", burstB: "rgba(127,29,29,.35)",
      frame: "rgba(255,255,255,.3)",
      bull: "#450a0a", num: "#ffffff", numStroke: "#450a0a",
      pip: "#4ade80", corner: "#fecaca",
    };
  }
  if (n % 10 === 0) {
    return {
      face: "linear-gradient(180deg,#fcd34d,#f59e0b)",
      burstA: "rgba(255,255,255,.45)", burstB: "rgba(180,83,9,.25)",
      frame: "rgba(120,53,15,.35)",
      bull: "#1e40af", num: "#ffffff", numStroke: "#92400e",
      pip: "#92400e", corner: "#78350f",
    };
  }
  if (n % 5 === 0) {
    return {
      face: "linear-gradient(180deg,#3b82f6,#1d4ed8)",
      burstA: "rgba(255,255,255,.25)", burstB: "rgba(30,58,138,.3)",
      frame: "rgba(255,255,255,.35)",
      bull: "#172554", num: "#fde047", numStroke: "#172554",
      pip: "#fde047", corner: "#dbeafe",
    };
  }
  return {
    face: "linear-gradient(180deg,#f8fafc,#dbe3ea)",
    burstA: "rgba(255,255,255,.9)", burstB: "rgba(148,163,184,.25)",
    frame: "rgba(30,64,175,.25)",
    bull: "#1e40af", num: "#ffffff", numStroke: "#1e3a8a",
    pip: "#1e40af", corner: "#334155",
  };
}

function RowChoice({ rows, onPick, hints, thinking, compact, cardW }:{
  rows:[Row,Row,Row,Row]; onPick:(idx:number)=>void;
  hints?:Map<number, number> | null; thinking?:boolean;
  compact:boolean; cardW:string;
}) {
  // Same coach semantics as the hand: the star marks the pick, +n the
  // extra bulls a row is expected to cost compared with it.
  const best = hints && hints.size ? Math.max(...hints.values()) : null;

  // Deliberately no dismiss path: the rules require a choice here.
  return (
    <motion.div
      className={`fixed inset-0 z-30 flex items-center justify-center ${
        compact ? "bg-black/80 p-2" : "bg-black/70 backdrop-blur-sm p-4"
      }`}
      style={compact ? {
        // A fixed inset-0 element spans UNDER the notch, so the backdrop
        // has to carry the insets itself.
        paddingLeft: "calc(8px + var(--sa-l))",
        paddingRight: "calc(8px + var(--sa-r))",
        paddingTop: "calc(8px + var(--sa-t))",
        paddingBottom: "calc(8px + var(--sa-b))",
      } : undefined}
      initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
      <motion.div initial={{scale:0.95,opacity:0}} animate={{scale:1,opacity:1}} exit={{scale:0.95,opacity:0}}
        className={`bg-slate-900 rounded-2xl shadow-2xl w-full overflow-y-auto overscroll-contain ${
          compact ? "max-w-none max-h-full p-2" : "max-w-3xl p-4"
        }`}>
        <div className={`text-slate-300 ${compact ? "text-[11px] mb-1.5" : "text-sm mb-3"}`}>
          {compact
            ? "Card is below every row. Take one:"
            : "Your card is lower than all rows. Choose a row to take:"}
          {thinking && best === null && (<span className="ml-2 text-amber-400">coach thinking…</span>)}
          {best !== null && !compact && (<span className="ml-2 text-amber-400">coach: ★ = best row · +n = extra bulls it risks</span>)}
        </div>
        <div className={`grid gap-2 ${compact ? "grid-cols-2" : "grid-cols-1 sm:grid-cols-2 gap-3"}`}>
          {[0,1,2,3].map(i => {
            const score = hints?.get(i);
            const cost = best !== null && score !== undefined ? best - score : null;
            const isBest = cost !== null && cost < 1e-6;
            const costClass =
              cost === null || isBest
                ? "bg-amber-500 text-slate-950"
                : cost < 1
                  ? "bg-emerald-700 text-emerald-100"
                  : cost < 3
                    ? "bg-amber-700 text-amber-100"
                    : "bg-red-700 text-red-100";
            return (
              <button key={i} onClick={() => onPick(i)}
                className={`text-left bg-slate-800 active:bg-slate-700 rounded-xl ${
                  compact ? "p-2" : "hover:bg-slate-700 p-3"
                } ${isBest ? "ring-2 ring-amber-500" : ""}`}>
                <div className={`flex items-center justify-between text-slate-300 ${
                  compact ? "text-[10px] mb-1" : "text-xs mb-2"
                }`}>
                  <span>Row {i+1}{cost !== null && (
                    <span className={`ml-1.5 px-1.5 py-0.5 rounded-full text-[10px] ${costClass}`}>
                      {isBest ? "★" : `+${cost.toFixed(1)}`}
                    </span>
                  )}</span>
                  <Bulls n={sumBulls(rows[i])} />
                </div>
                <div className={`flex hscroll ${compact ? "gap-1" : "gap-2"}`}>
                  {rows[i].map(c => <CardView key={c.id} card={c} w={cardW} />)}
                </div>
              </button>
            );
          })}
        </div>
      </motion.div>
    </motion.div>
  );
}

/** Shared bottom-sheet shell: anchored to the bottom on small screens,
 * centred from md up, always scrollable inside a short viewport, and
 * always clear of the home indicator. */
function Sheet({ title, icon, onClose, children }:{
  title: string; icon: ReactNode; onClose: () => void; children: ReactNode;
}) {
  useEscape(onClose);
  return (
    <motion.div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm compact:backdrop-blur-none z-30 flex items-end md:items-center justify-center"
      onClick={onClose}
      initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
      <motion.div
        onClick={(e) => e.stopPropagation()}
        initial={{y:40,opacity:0}} animate={{y:0,opacity:1}} exit={{y:40,opacity:0}}
        className="bg-slate-900 w-full md:max-w-md rounded-t-2xl md:rounded-2xl shadow-2xl
                   p-4 pb-gutter max-h-[calc(100svh-1rem)] overflow-y-auto overscroll-contain">
        <div className="flex items-center gap-2 mb-3">
          {icon}
          <div className="font-semibold">{title}</div>
          <div className="flex-1" />
          <button onClick={onClose} aria-label="Close" className="tap-target p-1 rounded hover:bg-slate-800">
            <X className="w-4 h-4" aria-hidden />
          </button>
        </div>
        {children}
      </motion.div>
    </motion.div>
  );
}

/** The compact layout has no room for a persistent leaderboard, so the
 * standings and the end-of-deal actions live here. */
function ScoresSheet({ state, leaderboard, matchOver, onNextDeal, onNewMatch, onClose }:{
  state: GameState; leaderboard: PlayerState[]; matchOver: boolean;
  onNextDeal: () => void; onNewMatch: () => void; onClose: () => void;
}) {
  const total = (p: PlayerState) => state.totals[p.id] + sumBulls(p.pen);
  const gameOver = state.phase === "gameOver";
  return (
    <Sheet title="Scores" icon={<Trophy className="w-4 h-4" aria-hidden />} onClose={onClose}>
      {gameOver && (
        <div className="mb-3 rounded-xl bg-slate-800/80 border border-slate-700 p-3 text-sm">
          {matchOver
            ? matchResultLine(leaderboard[0], total(leaderboard[0]))
            : `Deal ${state.dealNumber} finished.`}
        </div>
      )}

      <div className="text-xs text-slate-400 mb-1">Fewest bulls wins. Match ends at 66.</div>
      <div className="space-y-1">
        {leaderboard.map((p, i) => (
          <div
            key={p.id}
            className={`flex items-center justify-between rounded-xl px-3 py-2 text-sm ${
              p.id === 0 ? "bg-emerald-900/40 border border-emerald-800" : "bg-slate-800/60"
            }`}
          >
            <span className="flex items-center gap-2">
              <span className="w-4 text-slate-500 tabular-nums">{i + 1}</span>
              <span>{p.name}</span>
            </span>
            <span className="inline-flex items-center gap-3 text-slate-300">
              <span className="inline-flex items-center gap-1">
                <Bulls n={sumBulls(p.pen)} />
                <span className="opacity-60 text-xs">this deal</span>
              </span>
              <span className="tabular-nums font-semibold">{total(p)}/66</span>
            </span>
          </div>
        ))}
      </div>

      {gameOver && (
        <div className="mt-4 flex justify-end">
          {matchOver ? (
            <button onClick={onNewMatch} className="px-4 py-2 rounded-xl bg-emerald-600 active:bg-emerald-500 hover:bg-emerald-500">
              New match
            </button>
          ) : (
            <button onClick={onNextDeal} className="px-4 py-2 rounded-xl bg-emerald-600 active:bg-emerald-500 hover:bg-emerald-500">
              Next deal
            </button>
          )}
        </div>
      )}
    </Sheet>
  );
}

/** Rules and controls. Auto-opens once per browser: the compact top bar
 * is icon-only, so this is the only place Coach and the tap-again
 * gesture are explained. */
function HelpSheet({ onClose }:{ onClose: () => void }) {
  return (
    <Sheet title="How to play" icon={<CircleHelp className="w-4 h-4" aria-hidden />} onClose={onClose}>
      <div className="space-y-3 text-sm text-slate-300">
        <p>
          Everyone plays one card at a time. Cards resolve lowest first, each joining the
          row that ends with the highest card below it. Take the sixth card in a row and
          you collect that row instead. <b>Fewest bulls wins</b>; the match ends when
          someone reaches 66.
        </p>
        <div className="rounded-xl bg-slate-800/60 p-3">
          <div className="font-semibold text-slate-100 mb-1">Playing a card</div>
          Tap a card to select it, then press <b>Play</b>. On a touch screen you can also
          just tap the selected card a second time.
        </div>
        <div className="rounded-xl bg-slate-800/60 p-3">
          <div className="font-semibold text-amber-400 mb-1 inline-flex items-center gap-2">
            <Lightbulb className="w-4 h-4" aria-hidden /> Coach
          </div>
          The trained neural bot scores every card in your hand: ★ is its pick, and +n is
          the extra bulls another card is expected to risk. It runs in the background, so
          badges appear a moment after your turn starts.
        </div>
        <p className="text-slate-400">
          Playing on a phone? Add this to your home screen for a full-screen, landscape
          game with no browser chrome in the way.
        </p>
      </div>
    </Sheet>
  );
}

function SettingsDialog({
  playersCount, seed, difficulty, onClose, onApply
}:{
  playersCount:number;
  seed:number;
  difficulty:BotStrategyId;
  onClose:()=>void;
  onApply:(players:number, seed:number, difficulty:BotStrategyId)=>void;
}){
  const [localPlayers, setLocalPlayers] = useState(playersCount);
  const [localSeed, setLocalSeed] = useState(seed);
  const [localDifficulty, setLocalDifficulty] = useState<BotStrategyId>(difficulty);
  return (
    <Sheet title="Settings" icon={<Settings2 className="w-4 h-4" aria-hidden />} onClose={onClose}>
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

        <div className="block text-sm">
          <div className="text-slate-300 mb-1">Bot difficulty</div>
          <div className="grid grid-cols-2 gap-2">
            {DIFFICULTIES.map(d => (
              <button
                key={d.id}
                onClick={() => setLocalDifficulty(d.id)}
                className={`text-left rounded-xl px-3 py-2 border ${
                  localDifficulty === d.id
                    ? "border-emerald-500 bg-emerald-600/20"
                    : "border-slate-700 bg-slate-800 hover:bg-slate-700"
                }`}
              >
                <div className="text-slate-100">{d.label}</div>
                <div className="text-xs text-slate-400">{d.blurb}</div>
              </button>
            ))}
          </div>
        </div>

        <label className="block text-sm">
          <div className="text-slate-300 mb-1">Seed</div>
          {/* text-base, i.e. 16px: iOS zooms the whole page when a focused
              input is smaller than that. This, not the viewport tag, is
              the usual cause of "the page zoomed when I tapped a field". */}
          <input
            value={localSeed}
            onChange={e=>setLocalSeed(parseInt(e.target.value)||0)}
            inputMode="numeric"
            className="w-full rounded-lg bg-slate-800 px-3 py-2 text-base"
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
          onClick={()=> onApply(localPlayers, localSeed, localDifficulty)}
          className="px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white"
        >
          Start new game
        </button>
      </div>
    </Sheet>
  );
}
