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
    /// Match mode: deals accumulate bull totals until someone reaches this
    /// (then the match ends, a win/loss bonus lands on the final step's
    /// rewards, and totals reset). 0 = independent single deals.
    match_to: u16,
    totals: Vec<Vec<u16>>,
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
    #[pyo3(signature = (num_games, specs, seed=0, match_to=0))]
    fn new(
        num_games: usize,
        specs: Vec<Option<String>>,
        seed: u64,
        match_to: u16,
    ) -> PyResult<Self> {
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
            match_to,
            totals: vec![vec![0; num_players]; num_games],
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

    /// Ground-truth belief targets for auxiliary supervision. Training-only:
    /// this reads hidden opponent hands, so it must never feed the policy's
    /// observation path.
    ///
    /// Flat (num_games * num_policy_seats * 104) i64. For each card, from the
    /// policy seat's perspective:
    ///   -100        not predicted (in own hand or publicly seen),
    ///   0..n-2      held by the opponent `label + 1` seats after us,
    ///   n-1         in the undealt stock.
    fn belief_targets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let k = self.policy_seats.len();
        let n = self.num_players;
        let mut out = vec![0i64; self.games.len() * k * NUM_CARDS];
        for (g, game) in self.games.iter().enumerate() {
            for (j, &seat) in self.policy_seats.iter().enumerate() {
                let base = (g * k + j) * NUM_CARDS;
                for c in 1..=NUM_CARDS as u8 {
                    let idx = base + c as usize - 1;
                    if set_contains(game.hand(seat), c) || set_contains(game.played(), c) {
                        out[idx] = -100;
                        continue;
                    }
                    let mut label = (n - 1) as i64; // stock unless an opponent has it
                    for d in 1..n {
                        if set_contains(game.hand((seat + d) % n), c) {
                            label = (d - 1) as i64;
                            break;
                        }
                    }
                    out[idx] = label;
                }
            }
        }
        out.into_pyarray(py)
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
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let k = self.policy_seats.len();
        let n = self.num_players;
        let num_games = self.games.len();
        let acts: Vec<i64> = actions.as_slice()?.to_vec();
        if acts.len() != num_games * k {
            return Err(PyValueError::new_err(format!(
                "expected {} actions, got {}",
                num_games * k,
                acts.len()
            )));
        }

        // Validate every action before mutating anything.
        for (g, game) in self.games.iter().enumerate() {
            for (j, &seat) in self.policy_seats.iter().enumerate() {
                let a = acts[g * k + j];
                if !(1..=NUM_CARDS as i64).contains(&a) || !set_contains(game.hand(seat), a as Card)
                {
                    return Err(PyValueError::new_err(format!(
                        "game {g} seat {seat}: card {a} is not playable"
                    )));
                }
            }
        }

        // Deterministic reset seeds drawn up front (one per game per step)
        // so the parallel section never touches the shared stream.
        let reset_seeds: Vec<u64> = (0..num_games)
            .map(|_| self.seed_stream.next_u64())
            .collect();

        struct Work<'a> {
            game: &'a mut Game,
            bots: &'a mut SeatBots,
            rng: &'a mut SplitMix64,
            acts: &'a [i64],
            reset_seed: u64,
            rewards: &'a mut [f32],
            done: &'a mut u8,
            finals: &'a mut [f32],
            totals: &'a mut Vec<u16>,
            match_done: &'a mut u8,
            match_finals: &'a mut [f32],
        }

        let policy_seats = &self.policy_seats;
        let match_to = self.match_to;
        let mut rewards = vec![0.0f32; num_games * k];
        let mut dones = vec![0u8; num_games];
        let mut finals = vec![0.0f32; num_games * n];
        let mut match_dones = vec![0u8; num_games];
        let mut match_finals = vec![0.0f32; num_games * n];

        {
            let mut items: Vec<Work> = self
                .games
                .iter_mut()
                .zip(self.bots.iter_mut())
                .zip(self.rngs.iter_mut())
                .zip(acts.chunks(k))
                .zip(reset_seeds.iter())
                .zip(rewards.chunks_mut(k))
                .zip(dones.iter_mut())
                .zip(finals.chunks_mut(n))
                .zip(self.totals.iter_mut())
                .zip(match_dones.iter_mut())
                .zip(match_finals.chunks_mut(n))
                .map(
                    |(
                        (
                            (
                                (((((((game, bots), rng), acts), seed), rewards), done), finals),
                                totals,
                            ),
                            match_done,
                        ),
                        match_finals,
                    )| Work {
                        game,
                        bots,
                        rng,
                        acts,
                        reset_seed: *seed,
                        rewards,
                        done,
                        finals,
                        totals,
                        match_done,
                        match_finals,
                    },
                )
                .collect();

            let run_one = move |w: &mut Work| {
                let mut cards: Vec<Card> = vec![0; n];
                for (j, &seat) in policy_seats.iter().enumerate() {
                    cards[seat] = w.acts[j] as Card;
                }
                for (seat, bot) in w.bots.iter_mut().enumerate() {
                    if let Some(bot) = bot {
                        cards[seat] = bot.choose_card(&w.game.view(seat), w.rng);
                    }
                }
                let pen_before: Vec<u16> = w.game.penalties().to_vec();
                w.game.play_cards(&cards).expect("all cards validated");
                Self::settle(w.game, w.bots, w.rng);

                let deltas: Vec<f32> = (0..n)
                    .map(|p| (w.game.penalties()[p] - pen_before[p]) as f32)
                    .collect();
                let total: f32 = deltas.iter().sum();
                for (j, &seat) in policy_seats.iter().enumerate() {
                    let others = (total - deltas[seat]) / (n - 1) as f32;
                    w.rewards[j] = others - deltas[seat];
                }

                if w.game.is_terminal() {
                    *w.done = 1;
                    for p in 0..n {
                        w.finals[p] = w.game.penalties()[p] as f32;
                    }
                    if match_to > 0 {
                        for p in 0..n {
                            w.totals[p] += w.game.penalties()[p];
                        }
                        if w.totals.iter().any(|&t| t >= match_to) {
                            // Match over: zero-sum outcome bonus on this
                            // step's rewards (lowest total wins, ties split).
                            *w.match_done = 1;
                            let best = *w.totals.iter().min().expect("non-empty");
                            let winners = w.totals.iter().filter(|&&t| t == best).count() as f32;
                            for (j, &seat) in policy_seats.iter().enumerate() {
                                let share = if w.totals[seat] == best {
                                    1.0 / winners
                                } else {
                                    0.0
                                };
                                w.rewards[j] += 10.0 * (share - 1.0 / n as f32);
                            }
                            for p in 0..n {
                                w.match_finals[p] = w.totals[p] as f32;
                                w.totals[p] = 0;
                            }
                        }
                    }
                    *w.game = Game::deal(n, w.reset_seed).expect("valid count");
                    if match_to > 0 {
                        w.game.set_totals(w.totals);
                    }
                }
            };

            py.allow_threads(|| {
                let threads = std::thread::available_parallelism()
                    .map(|t| t.get())
                    .unwrap_or(1)
                    .min(items.len());
                if threads <= 1 {
                    for w in items.iter_mut() {
                        run_one(w);
                    }
                } else {
                    let chunk_size = items.len().div_ceil(threads);
                    std::thread::scope(|scope| {
                        for chunk in items.chunks_mut(chunk_size) {
                            scope.spawn(|| {
                                for w in chunk.iter_mut() {
                                    run_one(w);
                                }
                            });
                        }
                    });
                }
            });
        }

        Ok((
            rewards.into_pyarray(py),
            dones.into_pyarray(py),
            finals.into_pyarray(py),
            match_dones.into_pyarray(py),
            match_finals.into_pyarray(py),
        ))
    }

    /// Turns remaining in the current deals (all games stay in lockstep).
    fn turns_per_deal(&self) -> usize {
        HAND_SIZE
    }

    /// Test/debug introspection: every seat's hand, per game. Reads hidden
    /// state — never use on the observation path.
    fn debug_hands(&self) -> Vec<Vec<Vec<u16>>> {
        self.games
            .iter()
            .map(|g| {
                (0..self.num_players)
                    .map(|p| g.hand_cards(p).iter().map(|&c| c as u16).collect())
                    .collect()
            })
            .collect()
    }

    /// Test/debug introspection: publicly revealed cards per game.
    fn debug_played(&self) -> Vec<Vec<u16>> {
        self.games
            .iter()
            .map(|g| {
                take5_core::cards::set_iter(g.played())
                    .map(|c| c as u16)
                    .collect()
            })
            .collect()
    }
}
