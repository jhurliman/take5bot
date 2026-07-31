//! Observation encoding v2. This is the single definition used by training,
//! arena, and (via WASM) the browser — bit-identical everywhere.
//!
//! Layout (f32, all values in [0, 1]):
//!   [0..104)    own hand, binary mask over card ids
//!   [104..208)  played mask: every publicly revealed card so far (row
//!               starters + all reveals, including cards buried in penalty
//!               piles). The card-counting signal.
//!   [208..228)  rows, per-slot normalized card ids (4 rows x 5 slots, 0=empty)
//!   [228..240)  per row: last card /104, length /5, bull sum /35
//!   [240..250)  penalty totals /66, seat-relative (self first, then other
//!               seats in play order), zero-padded to 10
//!   [250..260)  seat-present mask (supports 2-10 players), self first
//!   [260]       turn /10
//!   [261]       cards left in own hand /10
//!   [262]       1 if we are resolving our own forced row choice
//!   [263]       the forced card /104 (0 otherwise)
//!   [264..274)  match totals carried from previous deals /66, seat-relative
//!               (self first), zero-padded to 10 — zeros in single-deal play
//!   [274]       highest carried total at the table /66 (match urgency)
//!   [275]       own headroom to 66: (66 - own carried total) /66, floor 0
//!
//! Total: 276 (v3; [0..264) is exactly the v2 layout, so nets exported
//! against v2 observations read the prefix unchanged). Bullhead values are intentionally not encoded: they are a
//! deterministic function of card id, learned trivially by an embedding.

use crate::cards::{set_contains, set_len, Card, NUM_CARDS};
use crate::game::{Game, Phase, View, MAX_PLAYERS, MAX_ROW_LEN, ROWS};
use crate::{bullheads, HAND_SIZE};

pub const OBS_LEN: usize = 276;

const ROW_SLOTS_OFF: usize = 2 * NUM_CARDS;
const ROW_SUMMARY_OFF: usize = ROW_SLOTS_OFF + ROWS * MAX_ROW_LEN;
const PENALTY_OFF: usize = ROW_SUMMARY_OFF + ROWS * 3;
const SEAT_MASK_OFF: usize = PENALTY_OFF + MAX_PLAYERS;
const SCALAR_OFF: usize = SEAT_MASK_OFF + MAX_PLAYERS;
const TOTALS_OFF: usize = SCALAR_OFF + 4;
const MATCH_SCALAR_OFF: usize = TOTALS_OFF + MAX_PLAYERS;

/// Encode `player`'s observation of `game` into `out` (length `OBS_LEN`).
pub fn encode_observation(game: &Game, player: usize, out: &mut [f32]) {
    let forced = match game.phase() {
        Phase::ChooseRow { player: p, card } if p as usize == player => Some(card),
        _ => None,
    };
    encode_view(&game.view(player), forced, out);
}

/// Encode an observation directly from a seat's `View`. `forced` is the
/// card awaiting this seat's row choice, if any. This is the entry point
/// bots use — they never hold a full `Game`.
pub fn encode_view(view: &View, forced: Option<Card>, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_LEN);
    out.fill(0.0);
    let player = view.player as usize;
    for c in 1..=NUM_CARDS as u8 {
        if set_contains(view.hand, c) {
            out[c as usize - 1] = 1.0;
        }
        if set_contains(view.played, c) {
            out[NUM_CARDS + c as usize - 1] = 1.0;
        }
    }

    for (r, row) in view.rows.iter().enumerate() {
        let mut bulls = 0u16;
        for (i, &c) in row.iter().enumerate() {
            out[ROW_SLOTS_OFF + r * MAX_ROW_LEN + i] = c as f32 / NUM_CARDS as f32;
            bulls += bullheads(c) as u16;
        }
        let last = *row.last().expect("rows are never empty");
        out[ROW_SUMMARY_OFF + r * 3] = last as f32 / NUM_CARDS as f32;
        out[ROW_SUMMARY_OFF + r * 3 + 1] = row.len() as f32 / MAX_ROW_LEN as f32;
        out[ROW_SUMMARY_OFF + r * 3 + 2] = bulls as f32 / 35.0;
    }

    let n = view.num_players as usize;
    for i in 0..n {
        let seat = (player + i) % n;
        out[PENALTY_OFF + i] = (view.penalties[seat] as f32 / 66.0).min(1.0);
        out[SEAT_MASK_OFF + i] = 1.0;
    }

    out[SCALAR_OFF] = view.turn as f32 / HAND_SIZE as f32;
    out[SCALAR_OFF + 1] = set_len(view.hand) as f32 / HAND_SIZE as f32;
    if let Some(card) = forced {
        out[SCALAR_OFF + 2] = 1.0;
        out[SCALAR_OFF + 3] = card as f32 / NUM_CARDS as f32;
    }

    let mut max_total = 0u16;
    for i in 0..n {
        let seat = (player + i) % n;
        let t = view.totals[seat];
        out[TOTALS_OFF + i] = (t as f32 / 66.0).min(1.5);
        max_total = max_total.max(t);
    }
    out[MATCH_SCALAR_OFF] = (max_total as f32 / 66.0).min(1.5);
    out[MATCH_SCALAR_OFF + 1] = (66.0 - view.totals[player] as f32).max(0.0) / 66.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs_shape_and_ranges() {
        let g = Game::deal(4, 99).unwrap();
        let mut out = vec![0.0f32; OBS_LEN];
        encode_observation(&g, 0, &mut out);
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        // 10 hand bits, 4 played bits (row starters).
        let hand_bits: f32 = out[..NUM_CARDS].iter().sum();
        let played_bits: f32 = out[NUM_CARDS..2 * NUM_CARDS].iter().sum();
        assert_eq!(hand_bits, 10.0);
        assert_eq!(played_bits, 4.0);
        // 4 seats present.
        let seats: f32 = out[SEAT_MASK_OFF..SEAT_MASK_OFF + MAX_PLAYERS].iter().sum();
        assert_eq!(seats, 4.0);
    }

    #[test]
    fn obs_is_player_relative() {
        let g = Game::deal(4, 7).unwrap();
        let mut a = vec![0.0f32; OBS_LEN];
        let mut b = vec![0.0f32; OBS_LEN];
        encode_observation(&g, 0, &mut a);
        encode_observation(&g, 1, &mut b);
        // Different hands -> different observations.
        assert_ne!(a[..NUM_CARDS], b[..NUM_CARDS]);
        // Same public info.
        assert_eq!(a[NUM_CARDS..2 * NUM_CARDS], b[NUM_CARDS..2 * NUM_CARDS]);
    }
}
