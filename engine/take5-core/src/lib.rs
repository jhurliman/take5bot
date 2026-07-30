//! Take 5 (6 nimmt!) engine: rules, observation encoding, baseline bots, arena.
//!
//! This crate is the single source of truth for game rules. It is
//! dependency-free so the same code compiles to native (training/arena via
//! PyO3) and WASM (browser play). Randomness comes from an internal SplitMix64
//! generator so results are reproducible across platforms.

pub mod arena;
pub mod bots;
pub mod cards;
pub mod game;
pub mod neural;
pub mod obs;
pub mod rng;

pub use cards::{bullheads, Card, NUM_CARDS};
pub use game::{Game, GameError, Phase, View, HAND_SIZE, MAX_PLAYERS, MAX_ROW_LEN, ROWS};
pub use neural::{NeuralNet, NeuralOutput};
pub use obs::{encode_observation, encode_view, OBS_LEN};
pub use rng::SplitMix64;
