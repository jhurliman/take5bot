//! Python bindings: thin wrappers over take5-core plus a multithreaded arena
//! entry point. Build with `scripts/build_engine.sh`; import as `take5_engine`.

mod vec_games;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use take5_core::bots::BotSpec;
use take5_core::cards::{set_contains, set_insert, CardSet, NUM_CARDS};
use take5_core::{arena, encode_observation, Phase, OBS_LEN};
use vec_games::VecGames;

fn to_set(cards: &[u8]) -> PyResult<CardSet> {
    let mut set: CardSet = 0;
    for &c in cards {
        if !(1..=NUM_CARDS as u8).contains(&c) {
            return Err(PyValueError::new_err(format!("invalid card id {c}")));
        }
        if set_contains(set, c) {
            return Err(PyValueError::new_err(format!("duplicate card id {c}")));
        }
        set_insert(&mut set, c);
    }
    Ok(set)
}

fn game_err(e: take5_core::GameError) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

/// A single deal of Take 5.
#[pyclass]
struct Game {
    inner: take5_core::Game,
}

#[pymethods]
impl Game {
    /// Shuffle and deal a fresh round for `num_players` seats.
    #[staticmethod]
    fn deal(num_players: usize, seed: u64) -> PyResult<Game> {
        Ok(Game {
            inner: take5_core::Game::deal(num_players, seed).map_err(game_err)?,
        })
    }

    /// Mirror an externally dealt round (used by the parity tests).
    #[staticmethod]
    fn from_state(
        hands: Vec<Vec<u8>>,
        row_starters: Vec<u8>,
        penalties: Vec<u16>,
        turn: u8,
    ) -> PyResult<Game> {
        let hands = hands
            .iter()
            .map(|h| to_set(h))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Game {
            inner: take5_core::Game::from_state(hands, row_starters, penalties, turn)
                .map_err(game_err)?,
        })
    }

    /// "select" | "choose_row" | "terminal"
    fn phase(&self) -> &'static str {
        match self.inner.phase() {
            Phase::Select => "select",
            Phase::ChooseRow { .. } => "choose_row",
            Phase::Terminal => "terminal",
        }
    }

    /// (player, forced_card) when in the choose_row phase, else None.
    fn choose_row_context(&self) -> Option<(u8, u16)> {
        match self.inner.phase() {
            Phase::ChooseRow { player, card } => Some((player, card as u16)),
            _ => None,
        }
    }

    fn legal_cards(&self, player: usize) -> Vec<u16> {
        self.inner
            .hand_cards(player)
            .iter()
            .map(|&c| c as u16)
            .collect()
    }

    /// Simultaneous reveal: one card per seat, in seat order.
    fn play_cards(&mut self, cards: Vec<u8>) -> PyResult<()> {
        self.inner.play_cards(&cards).map_err(game_err)
    }

    fn choose_row(&mut self, row: usize) -> PyResult<()> {
        self.inner.choose_row(row).map_err(game_err)
    }

    fn rows(&self) -> Vec<Vec<u16>> {
        self.inner
            .rows()
            .iter()
            .map(|r| r.iter().map(|&c| c as u16).collect())
            .collect()
    }

    fn penalties(&self) -> Vec<u16> {
        self.inner.penalties().to_vec()
    }

    fn hands(&self) -> Vec<Vec<u16>> {
        (0..self.inner.num_players())
            .map(|p| self.inner.hand_cards(p).iter().map(|&c| c as u16).collect())
            .collect()
    }

    fn played(&self) -> Vec<u16> {
        take5_core::cards::set_iter(self.inner.played())
            .map(|c| c as u16)
            .collect()
    }

    fn turn(&self) -> u8 {
        self.inner.turn()
    }

    fn num_players(&self) -> usize {
        self.inner.num_players()
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Final scores as negative penalties (higher is better).
    fn returns(&self) -> Vec<f32> {
        self.inner.returns()
    }

    /// Flat f32 observation for `player` (length `obs_len()`).
    fn observe(&self, player: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; OBS_LEN];
        encode_observation(&self.inner, player, &mut out);
        out
    }
}

#[pyfunction]
fn obs_len() -> usize {
    OBS_LEN
}

/// Test helper: run the Rust forward pass on one observation.
/// Returns (policy_logits[104], value, belief_logits[416]).
#[pyfunction]
fn debug_neural_eval(path: &str, obs: Vec<f32>) -> PyResult<(Vec<f32>, f32, Vec<f32>)> {
    if obs.len() != OBS_LEN {
        return Err(PyValueError::new_err(format!(
            "expected {OBS_LEN} obs values, got {}",
            obs.len()
        )));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("failed to read {path}: {e}")))?;
    let net = take5_core::NeuralNet::from_bytes(&bytes)
        .map_err(|e| PyValueError::new_err(format!("failed to parse {path}: {e:?}")))?;
    let out = net.forward(&obs);
    Ok((out.policy_logits.to_vec(), out.value, out.belief_logits))
}

#[pyfunction]
fn bullheads(card: u8) -> PyResult<u8> {
    if !(1..=NUM_CARDS as u8).contains(&card) {
        return Err(PyValueError::new_err(format!("invalid card id {card}")));
    }
    Ok(take5_core::bullheads(card))
}

/// Run a round-robin arena: one bot spec per seat (seats rotate each game).
/// Specs: "random" | "lowest" | "greedy" | "mc" | "mc:<worlds>".
/// Returns [(seat_bots, penalties)] per game; deterministic in (seed, games).
#[pyfunction]
#[pyo3(signature = (specs, games, seed=0, threads=0))]
fn run_arena(
    py: Python<'_>,
    specs: Vec<String>,
    games: u64,
    seed: u64,
    threads: usize,
) -> PyResult<Vec<(Vec<usize>, Vec<u16>)>> {
    let parsed: Vec<BotSpec> = specs
        .iter()
        .map(|s| {
            BotSpec::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown bot spec: {s}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    if parsed.len() < 2 || parsed.len() > take5_core::MAX_PLAYERS {
        return Err(PyValueError::new_err(
            "provide 2..=10 bot specs (one per seat)",
        ));
    }

    let threads = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };

    let results = py.allow_threads(move || {
        let threads = threads.min(games.max(1) as usize);
        let chunk = games.div_ceil(threads as u64);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for t in 0..threads as u64 {
                let start = t * chunk;
                let end = (start + chunk).min(games);
                if start >= end {
                    break;
                }
                let parsed = &parsed;
                handles.push(scope.spawn(move || arena::run_match_range(parsed, start, end, seed)));
            }
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("arena thread panicked"))
                .collect::<Vec<_>>()
        })
    });

    Ok(results
        .into_iter()
        .map(|r| (r.seat_bots, r.penalties))
        .collect())
}

#[pymodule]
fn take5_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<VecGames>()?;
    m.add_function(wrap_pyfunction!(obs_len, m)?)?;
    m.add_function(wrap_pyfunction!(debug_neural_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bullheads, m)?)?;
    m.add_function(wrap_pyfunction!(run_arena, m)?)?;
    Ok(())
}
