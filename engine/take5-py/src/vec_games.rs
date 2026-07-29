//! Vectorized self-play environment for training.
//!
//! A `VecGames` holds N concurrent deals. Each seat is either *policy*
//! (actions supplied from Python each step) or a built-in bot (acts inside
//! `step`). Every deal lasts exactly `HAND_SIZE` (10) simultaneous turns, so
//! all games finish together and auto-reset — rollouts stay rectangular.
//!
//! v1 simplification: when a *policy* seat is forced to pick a row to take
//! (its card was lower than every row end), the cheapest row is taken
//! automatically. Bots use their own row logic. Revisit with a row-choice
//! head in M4 if the arena shows it mattering.
//!
//! Rewards are relative penalty deltas, attributed per seat at resolution
//! time: `mean(other seats' bull delta) - own bull delta`. Summed over a
//! deal (gamma=1) this equals the seat's final relative score; summed over
//! all seats each step it is exactly zero.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use take5_core::bots::{min_row_bulls, Bot, BotSpec};
use take5_core::cards::{set_contains, Card, NUM_CARDS};
use take5_core::{encode_observation, Game, Phase, SplitMix64, HAND_SIZE, MAX_PLAYERS, OBS_LEN};

type SeatBots = Vec<Option<Box<dyn Bot + Send + Sync>>>;

#[pyclass]
pub struct VecGames {
    games: Vec<Game>,
    rngs: Vec<SplitMix64>,
    bots: Vec<SeatBots>,
    policy_seats: Vec<usize>,
    num_players: usize,
    seed_stream: SplitMix64,
}

impl VecGames {
    /// Drain any pending forced row choices so the game lands back in
    /// `Select` (or `Terminal`).
    fn settle(game: &mut Game, bots: &mut SeatBots, rng: &mut SplitMix64) {
        while let Phase::ChooseRow { player, card } = game.phase() {
            let p = player as usize;
            let row = match &mut bots[p] {
                Some(bot) => bot.choose_row(&game.view(p), card, rng),
                None => min_row_bulls(game.rows()).0,
            };
            game.choose_row(row).expect("row choice is legal");
        }
    }
}

#[pymethods]
impl VecGames {
    /// `specs[seat]` is None for a policy seat or a bot spec string
    /// ("random" | "lowest" | "greedy" | "mc" | "mc:<worlds>").
    #[new]
    #[pyo3(signature = (num_games, specs, seed=0))]
    fn new(num_games: usize, specs: Vec<Option<String>>, seed: u64) -> PyResult<Self> {
        let num_players = specs.len();
        if !(2..=MAX_PLAYERS).contains(&num_players) {
            return Err(PyValueError::new_err("provide 2..=10 seat specs"));
        }
        if num_games == 0 {
            return Err(PyValueError::new_err("num_games must be >= 1"));
        }
        let parsed: Vec<Option<BotSpec>> = specs
            .iter()
            .map(|s| match s {
                None => Ok(None),
                Some(s) => BotSpec::parse(s)
                    .map(Some)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown bot spec: {s}"))),
            })
            .collect::<PyResult<_>>()?;
        let policy_seats: Vec<usize> = (0..num_players).filter(|&i| parsed[i].is_none()).collect();
        if policy_seats.is_empty() {
            return Err(PyValueError::new_err(
                "at least one seat must be None (policy)",
            ));
        }

        let mut seed_stream = SplitMix64::new(seed);
        let mut games = Vec::with_capacity(num_games);
        let mut rngs = Vec::with_capacity(num_games);
        let mut bots = Vec::with_capacity(num_games);
        for _ in 0..num_games {
            games.push(Game::deal(num_players, seed_stream.next_u64()).expect("valid count"));
            rngs.push(SplitMix64::new(seed_stream.next_u64()));
            bots.push(
                parsed
                    .iter()
                    .map(|s| s.as_ref().map(|spec| spec.build()))
                    .collect::<SeatBots>(),
            );
        }
        Ok(VecGames {
            games,
            rngs,
            bots,
            policy_seats,
            num_players,
            seed_stream,
        })
    }

    fn num_games(&self) -> usize {
        self.games.len()
    }

    fn num_players(&self) -> usize {
        self.num_players
    }

    /// Seat indices controlled by the policy, in the order actions are taken.
    fn policy_seats(&self) -> Vec<usize> {
        self.policy_seats.clone()
    }

    /// Observations and legal-card masks for every policy seat.
    /// obs: flat (num_games * num_policy_seats * OBS_LEN) f32,
    /// mask: flat (num_games * num_policy_seats * 104) f32 (1 = playable).
    fn observe<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
        let k = self.policy_seats.len();
        let mut obs = vec![0.0f32; self.games.len() * k * OBS_LEN];
        let mut mask = vec![0.0f32; self.games.len() * k * NUM_CARDS];
        for (g, game) in self.games.iter().enumerate() {
            for (j, &seat) in self.policy_seats.iter().enumerate() {
                let o = (g * k + j) * OBS_LEN;
                encode_observation(game, seat, &mut obs[o..o + OBS_LEN]);
                let m = (g * k + j) * NUM_CARDS;
                for c in 1..=NUM_CARDS as u8 {
                    if set_contains(game.hand(seat), c) {
                        mask[m + c as usize - 1] = 1.0;
                    }
                }
            }
        }
        (obs.into_pyarray(py), mask.into_pyarray(py))
    }

    /// Advance every game one simultaneous turn.
    ///
    /// `actions`: flat (num_games * num_policy_seats) card ids (1..=104).
    /// Returns (rewards, dones, final_penalties):
    ///   rewards: flat (num_games * num_policy_seats) f32 relative deltas,
    ///   dones: (num_games,) u8 — 1 when that game just finished (and was
    ///     auto-reset to a fresh deal),
    ///   final_penalties: flat (num_games * num_players) f32 — final bull
    ///     totals for finished games, zeros otherwise.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let k = self.policy_seats.len();
        let n = self.num_players;
        let acts = actions.as_slice()?;
        if acts.len() != self.games.len() * k {
            return Err(PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.games.len() * k,
                acts.len()
            )));
        }

        let mut rewards = vec![0.0f32; self.games.len() * k];
        let mut dones = vec![0u8; self.games.len()];
        let mut finals = vec![0.0f32; self.games.len() * n];

        for g in 0..self.games.len() {
            let game = &mut self.games[g];
            let pen_before: Vec<u16> = game.penalties().to_vec();

            let mut cards: Vec<Card> = vec![0; n];
            for (j, &seat) in self.policy_seats.iter().enumerate() {
                let a = acts[g * k + j];
                if !(1..=NUM_CARDS as i64).contains(&a) || !set_contains(game.hand(seat), a as Card)
                {
                    return Err(PyValueError::new_err(format!(
                        "game {g} seat {seat}: card {a} is not playable"
                    )));
                }
                cards[seat] = a as Card;
            }
            for (seat, bot) in self.bots[g].iter_mut().enumerate() {
                if let Some(bot) = bot {
                    cards[seat] = bot.choose_card(&game.view(seat), &mut self.rngs[g]);
                }
            }

            game.play_cards(&cards).expect("all cards validated");
            Self::settle(game, &mut self.bots[g], &mut self.rngs[g]);

            let deltas: Vec<f32> = (0..n)
                .map(|p| (game.penalties()[p] - pen_before[p]) as f32)
                .collect();
            let total: f32 = deltas.iter().sum();
            for (j, &seat) in self.policy_seats.iter().enumerate() {
                let others = (total - deltas[seat]) / (n - 1) as f32;
                rewards[g * k + j] = others - deltas[seat];
            }

            if game.is_terminal() {
                dones[g] = 1;
                for p in 0..n {
                    finals[g * n + p] = game.penalties()[p] as f32;
                }
                self.games[g] = Game::deal(n, self.seed_stream.next_u64()).expect("valid count");
            }
        }

        Ok((
            rewards.into_pyarray(py),
            dones.into_pyarray(py),
            finals.into_pyarray(py),
        ))
    }

    /// Turns remaining in the current deals (all games stay in lockstep).
    fn turns_per_deal(&self) -> usize {
        HAND_SIZE
    }
}
