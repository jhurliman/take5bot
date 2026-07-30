//! WASM bridge for the web UI. The browser keeps its own game state (the
//! existing TypeScript implementation) and asks an `EngineBot` for decisions
//! by handing over exactly what a seat may see — the same `View` the native
//! bots consume, so browser play and arena play share one brain.

use take5_core::bots::{Bot, BotSpec, NeuralSearchBot};
use take5_core::cards::{set_insert, CardSet};
use take5_core::game::{View, ROWS};
use take5_core::{NeuralNet, SplitMix64};
use wasm_bindgen::prelude::*;

fn to_set(cards: &[u8]) -> Result<CardSet, JsError> {
    let mut set: CardSet = 0;
    for &c in cards {
        if !(1..=104).contains(&c) {
            return Err(JsError::new(&format!("invalid card id {c}")));
        }
        set_insert(&mut set, c);
    }
    Ok(set)
}

fn to_rows(rows_flat: &[u8], row_lens: &[u8]) -> Result<[Vec<u8>; ROWS], JsError> {
    if row_lens.len() != ROWS {
        return Err(JsError::new("expected 4 row lengths"));
    }
    let total: usize = row_lens.iter().map(|&l| l as usize).sum();
    if rows_flat.len() != total {
        return Err(JsError::new("rows_flat length mismatch"));
    }
    let mut rows: [Vec<u8>; ROWS] = Default::default();
    let mut off = 0;
    for (r, &len) in row_lens.iter().enumerate() {
        let len = len as usize;
        if len == 0 || len > 5 {
            return Err(JsError::new("row length must be 1..=5"));
        }
        rows[r] = rows_flat[off..off + len].to_vec();
        off += len;
    }
    Ok(rows)
}

enum Inner {
    Generic(Box<dyn Bot + Send + Sync>),
    Neural(NeuralSearchBot),
}

impl Inner {
    fn as_bot(&mut self) -> &mut dyn Bot {
        match self {
            Inner::Generic(b) => b.as_mut(),
            Inner::Neural(b) => b,
        }
    }
}

/// A bot the browser can consult. Specs: "random" | "lowest" | "greedy" |
/// "mc:<worlds>" | "neural:<worlds>" (neural requires the weights blob).
#[wasm_bindgen]
pub struct EngineBot {
    inner: Inner,
    rng: SplitMix64,
}

#[wasm_bindgen]
impl EngineBot {
    #[wasm_bindgen(constructor)]
    pub fn new(spec: &str, weights: Option<Box<[u8]>>, seed: u64) -> Result<EngineBot, JsError> {
        let inner: Inner = if let Some(rest) = spec.strip_prefix("neural:") {
            let worlds: u32 = rest
                .parse()
                .map_err(|_| JsError::new("neural spec must be neural:<worlds>"))?;
            let bytes = weights.ok_or_else(|| JsError::new("neural bot requires weights"))?;
            let net = NeuralNet::from_bytes(&bytes)
                .map_err(|e| JsError::new(&format!("bad weights: {e:?}")))?;
            Inner::Neural(NeuralSearchBot {
                net: std::sync::Arc::new(net),
                worlds,
            })
        } else {
            let parsed = BotSpec::parse(spec)
                .ok_or_else(|| JsError::new(&format!("unknown bot spec: {spec}")))?;
            if matches!(parsed, BotSpec::Neural { .. }) {
                return Err(JsError::new("use neural:<worlds> with weights in WASM"));
            }
            Inner::Generic(parsed.build())
        };
        Ok(EngineBot {
            inner,
            rng: SplitMix64::new(seed),
        })
    }

    /// Pick a card to play from `hand`, given everything this seat can see.
    /// `played` must contain every publicly revealed card (all cards
    /// currently in rows plus every card in any penalty pile).
    #[allow(clippy::too_many_arguments)]
    pub fn choose_card(
        &mut self,
        player: u8,
        num_players: u8,
        hand: &[u8],
        rows_flat: &[u8],
        row_lens: &[u8],
        penalties: &[u16],
        played: &[u8],
        turn: u8,
    ) -> Result<u8, JsError> {
        if hand.is_empty() {
            return Err(JsError::new("hand is empty"));
        }
        if penalties.len() != num_players as usize || player >= num_players {
            return Err(JsError::new("bad player/penalties"));
        }
        let rows = to_rows(rows_flat, row_lens)?;
        let view = View {
            player,
            num_players,
            hand: to_set(hand)?,
            played: to_set(played)?,
            rows: &rows,
            penalties,
            turn,
        };
        Ok(self.inner.as_bot().choose_card(&view, &mut self.rng))
    }

    /// Pick which row to take when `forced` card is below every row end.
    #[allow(clippy::too_many_arguments)]
    pub fn choose_row(
        &mut self,
        player: u8,
        num_players: u8,
        hand: &[u8],
        rows_flat: &[u8],
        row_lens: &[u8],
        penalties: &[u16],
        played: &[u8],
        turn: u8,
        forced: u8,
    ) -> Result<u8, JsError> {
        let rows = to_rows(rows_flat, row_lens)?;
        let view = View {
            player,
            num_players,
            hand: to_set(hand)?,
            played: to_set(played)?,
            rows: &rows,
            penalties,
            turn,
        };
        Ok(self.inner.as_bot().choose_row(&view, forced, &mut self.rng) as u8)
    }
}

#[wasm_bindgen]
impl EngineBot {
    /// Coach mode: score every legal card (higher = better). Returns a flat
    /// Float32Array of (card_id, score) pairs. Neural bots only.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze(
        &mut self,
        player: u8,
        num_players: u8,
        hand: &[u8],
        rows_flat: &[u8],
        row_lens: &[u8],
        penalties: &[u16],
        played: &[u8],
        turn: u8,
    ) -> Result<Vec<f32>, JsError> {
        let Inner::Neural(bot) = &mut self.inner else {
            return Err(JsError::new("analyze requires a neural bot"));
        };
        if hand.is_empty() {
            return Err(JsError::new("hand is empty"));
        }
        let rows = to_rows(rows_flat, row_lens)?;
        let view = View {
            player,
            num_players,
            hand: to_set(hand)?,
            played: to_set(played)?,
            rows: &rows,
            penalties,
            turn,
        };
        let scored = bot.analyze(&view, &mut self.rng);
        let mut out = Vec::with_capacity(scored.len() * 2);
        for (card, score) in scored {
            out.push(card as f32);
            out.push(score as f32);
        }
        Ok(out)
    }
}

/// Bullhead value of a card (parity helper for the TS side).
#[wasm_bindgen]
pub fn bullheads(card: u8) -> u8 {
    take5_core::bullheads(card)
}
