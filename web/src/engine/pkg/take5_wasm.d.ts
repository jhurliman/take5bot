/* tslint:disable */
/* eslint-disable */

/**
 * A bot the browser can consult. Specs: "random" | "lowest" | "greedy" |
 * "mc:<worlds>" | "neural:<worlds>" (neural requires the weights blob).
 */
export class EngineBot {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Coach mode: score every legal card (higher = better). Returns a flat
     * Float32Array of (card_id, score) pairs. Neural bots only.
     */
    analyze(player: number, num_players: number, hand: Uint8Array, rows_flat: Uint8Array, row_lens: Uint8Array, penalties: Uint16Array, played: Uint8Array, turn: number): Float32Array;
    /**
     * Pick a card to play from `hand`, given everything this seat can see.
     * `played` must contain every publicly revealed card (all cards
     * currently in rows plus every card in any penalty pile).
     */
    choose_card(player: number, num_players: number, hand: Uint8Array, rows_flat: Uint8Array, row_lens: Uint8Array, penalties: Uint16Array, played: Uint8Array, turn: number): number;
    /**
     * Pick which row to take when `forced` card is below every row end.
     */
    choose_row(player: number, num_players: number, hand: Uint8Array, rows_flat: Uint8Array, row_lens: Uint8Array, penalties: Uint16Array, played: Uint8Array, turn: number, forced: number): number;
    constructor(spec: string, weights: Uint8Array | null | undefined, seed: bigint);
}

/**
 * Bullhead value of a card (parity helper for the TS side).
 */
export function bullheads(card: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_enginebot_free: (a: number, b: number) => void;
    readonly bullheads: (a: number) => number;
    readonly enginebot_analyze: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number, number];
    readonly enginebot_choose_card: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number];
    readonly enginebot_choose_row: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number, number];
    readonly enginebot_new: (a: number, b: number, c: number, d: number, e: bigint) => [number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
