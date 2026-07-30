//! Baseline bots. These are the arena's fixed measuring sticks — the neural
//! bot must beat `mc:N` convincingly before it earns "genuinely strong".

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::cards::{bullheads, set_insert, set_iter, set_len, Card, CardSet};
use crate::game::{Game, Phase, View, MAX_ROW_LEN, ROWS};
use crate::neural::{NeuralNet, BELIEF_CLASSES};
use crate::obs::{encode_view, OBS_LEN};
use crate::rng::SplitMix64;

pub trait Bot {
    fn choose_card(&mut self, view: &View, rng: &mut SplitMix64) -> Card;
    fn choose_row(&mut self, view: &View, forced: Card, rng: &mut SplitMix64) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BotSpec {
    Random,
    Lowest,
    Greedy,
    McRollout {
        worlds: u32,
    },
    /// Trained net + belief-guided determinized search. `worlds == 0` plays
    /// the raw policy argmax (no search).
    Neural {
        path: String,
        worlds: u32,
    },
}

impl BotSpec {
    /// Parse "random" | "lowest" | "greedy" | "mc" | "mc:<worlds>" |
    /// "neural:<weights-path>[:<worlds>]".
    pub fn parse(s: &str) -> Option<BotSpec> {
        match s {
            "random" => Some(BotSpec::Random),
            "lowest" => Some(BotSpec::Lowest),
            "greedy" => Some(BotSpec::Greedy),
            "mc" => Some(BotSpec::McRollout { worlds: 64 }),
            _ => {
                if let Some(rest) = s.strip_prefix("neural:") {
                    // Optional trailing ":<worlds>"; the path may not
                    // contain further colons in that case.
                    return Some(match rest.rsplit_once(':') {
                        Some((path, w)) => match w.parse() {
                            Ok(worlds) => BotSpec::Neural {
                                path: path.to_string(),
                                worlds,
                            },
                            Err(_) => BotSpec::Neural {
                                path: rest.to_string(),
                                worlds: 32,
                            },
                        },
                        None => BotSpec::Neural {
                            path: rest.to_string(),
                            worlds: 32,
                        },
                    });
                }
                let worlds = s.strip_prefix("mc:")?.parse().ok()?;
                Some(BotSpec::McRollout { worlds })
            }
        }
    }

    pub fn build(&self) -> Box<dyn Bot + Send + Sync> {
        match self {
            BotSpec::Random => Box::new(RandomBot),
            BotSpec::Lowest => Box::new(LowestBot),
            BotSpec::Greedy => Box::new(GreedyBot),
            BotSpec::McRollout { worlds } => Box::new(McRolloutBot { worlds: *worlds }),
            BotSpec::Neural { path, worlds } => Box::new(NeuralSearchBot {
                net: load_cached_net(path),
                worlds: *worlds,
            }),
        }
    }
}

/// Process-wide weight cache: the arena builds bots per game, and reloading
/// a multi-megabyte net for each would dwarf the games themselves.
fn load_cached_net(path: &str) -> Arc<NeuralNet> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<NeuralNet>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("net cache lock");
    if let Some(net) = map.get(path) {
        return net.clone();
    }
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read net weights {path}: {e}"));
    let net = Arc::new(
        NeuralNet::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("failed to parse net weights {path}: {e:?}")),
    );
    map.insert(path.to_string(), net.clone());
    net
}

// ------------------------------------------------------------ shared logic

/// Immediate bull cost of revealing `card` against the current rows, ignoring
/// other players' simultaneous cards.
pub fn immediate_cost(rows: &[Vec<Card>; ROWS], card: Card) -> u16 {
    match Game::target_row(rows, card) {
        None => min_row_bulls(rows).1,
        Some(r) => {
            if rows[r].len() == MAX_ROW_LEN {
                row_bulls(&rows[r])
            } else {
                0
            }
        }
    }
}

pub fn row_bulls(row: &[Card]) -> u16 {
    row.iter().map(|&c| bullheads(c) as u16).sum()
}

/// (index, bulls) of the cheapest row to take.
pub fn min_row_bulls(rows: &[Vec<Card>; ROWS]) -> (usize, u16) {
    let mut best = (0, row_bulls(&rows[0]));
    for (i, row) in rows.iter().enumerate().skip(1) {
        let b = row_bulls(row);
        if b < best.1 {
            best = (i, b);
        }
    }
    best
}

/// Greedy card choice: minimize immediate cost, tie-break on lower card.
pub fn greedy_card(hand: CardSet, rows: &[Vec<Card>; ROWS]) -> Card {
    let mut best: Option<(u16, Card)> = None;
    for card in set_iter(hand) {
        let cost = immediate_cost(rows, card);
        if best.is_none() || (cost, card) < best.unwrap() {
            best = Some((cost, card));
        }
    }
    best.expect("hand is non-empty").1
}

// ------------------------------------------------------------------- bots

pub struct RandomBot;

impl Bot for RandomBot {
    fn choose_card(&mut self, view: &View, rng: &mut SplitMix64) -> Card {
        let cards: Vec<Card> = set_iter(view.hand).collect();
        cards[rng.choose_index(cards.len())]
    }

    fn choose_row(&mut self, _view: &View, _forced: Card, rng: &mut SplitMix64) -> usize {
        rng.choose_index(ROWS)
    }
}

pub struct LowestBot;

impl Bot for LowestBot {
    fn choose_card(&mut self, view: &View, _rng: &mut SplitMix64) -> Card {
        set_iter(view.hand).next().expect("hand is non-empty")
    }

    fn choose_row(&mut self, view: &View, _forced: Card, _rng: &mut SplitMix64) -> usize {
        min_row_bulls(view.rows).0
    }
}

pub struct GreedyBot;

impl Bot for GreedyBot {
    fn choose_card(&mut self, view: &View, _rng: &mut SplitMix64) -> Card {
        greedy_card(view.hand, view.rows)
    }

    fn choose_row(&mut self, view: &View, _forced: Card, _rng: &mut SplitMix64) -> usize {
        min_row_bulls(view.rows).0
    }
}

/// Determinized Monte-Carlo rollout bot: sample `worlds` completions of the
/// hidden opponent hands, play each own-card candidate against greedy
/// opponents to the end of the deal, and pick the card with the best mean
/// relative score (own penalty delta minus mean opponent delta).
///
/// No neural net involved — this is the search-only strength baseline.
pub struct McRolloutBot {
    pub worlds: u32,
}

impl McRolloutBot {
    /// Sample a full determinized game consistent with `view`.
    fn sample_world(view: &View, rng: &mut SplitMix64) -> Game {
        let n = view.num_players as usize;
        let hand_count = set_len(view.hand) as usize;
        let mut pool: Vec<Card> = set_iter(view.unseen()).collect();
        rng.shuffle(&mut pool);

        let mut hands = vec![0 as CardSet; n];
        hands[view.player as usize] = view.hand;
        let mut next = 0;
        for (p, hand) in hands.iter_mut().enumerate() {
            if p == view.player as usize {
                continue;
            }
            for _ in 0..hand_count {
                crate::cards::set_insert(hand, pool[next]);
                next += 1;
            }
        }

        Game::from_position(
            hands,
            view.rows.clone(),
            view.penalties.to_vec(),
            view.played,
            view.turn,
        )
        .expect("view state is always valid")
    }

    /// Play `game` to the end of the deal with all seats greedy.
    fn rollout(game: &mut Game) {
        loop {
            match game.phase() {
                Phase::Terminal => return,
                Phase::ChooseRow { .. } => {
                    let row = min_row_bulls(game.rows()).0;
                    game.choose_row(row).expect("row choice is legal");
                }
                Phase::Select => {
                    let cards: Vec<Card> = (0..game.num_players())
                        .map(|p| greedy_card(game.hand(p), game.rows()))
                        .collect();
                    game.play_cards(&cards).expect("greedy cards are legal");
                }
            }
        }
    }

    /// Mean relative penalty delta for `me` across a finished world.
    fn relative_score(game: &Game, me: usize, start: &[u16]) -> f32 {
        let n = game.num_players();
        let my_delta = (game.penalties()[me] - start[me]) as f32;
        let opp_delta: f32 = (0..n)
            .filter(|&p| p != me)
            .map(|p| (game.penalties()[p] - start[p]) as f32)
            .sum();
        my_delta - opp_delta / (n - 1) as f32
    }
}

impl Bot for McRolloutBot {
    fn choose_card(&mut self, view: &View, rng: &mut SplitMix64) -> Card {
        let candidates: Vec<Card> = set_iter(view.hand).collect();
        if candidates.len() == 1 {
            return candidates[0];
        }
        let me = view.player as usize;
        let start = view.penalties.to_vec();
        let mut totals = vec![0.0f32; candidates.len()];

        for _ in 0..self.worlds {
            let world = Self::sample_world(view, rng);
            for (i, &candidate) in candidates.iter().enumerate() {
                let mut g = world.clone();
                let cards: Vec<Card> = (0..g.num_players())
                    .map(|p| {
                        if p == me {
                            candidate
                        } else {
                            greedy_card(g.hand(p), g.rows())
                        }
                    })
                    .collect();
                g.play_cards(&cards).expect("candidate is legal");
                while let Phase::ChooseRow { .. } = g.phase() {
                    let row = min_row_bulls(g.rows()).0;
                    g.choose_row(row).expect("row choice is legal");
                }
                Self::rollout(&mut g);
                totals[i] += Self::relative_score(&g, me, &start);
            }
        }

        let best = totals
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("scores are finite"))
            .expect("candidates is non-empty")
            .0;
        candidates[best]
    }

    fn choose_row(&mut self, view: &View, _forced: Card, _rng: &mut SplitMix64) -> usize {
        // Taking the cheapest row is near-optimal; the interesting decision
        // (which card to play) is handled above. Revisit if the arena ever
        // shows this mattering.
        min_row_bulls(view.rows).0
    }
}

/// Neural bot: raw policy argmax when `worlds == 0`, else one-ply
/// expectimax over belief-sampled determinizations with value bootstrap.
///
/// Per world: sample opponent hands from the belief head (capacity-
/// constrained), let opponents pick cards by sampling their own policy,
/// then score every own candidate by (this turn's relative bull delta +
/// value head of the resulting state). Averages over worlds; plays argmax.
pub struct NeuralSearchBot {
    pub net: Arc<NeuralNet>,
    pub worlds: u32,
}

impl NeuralSearchBot {
    fn policy_argmax(out_logits: &[f32], hand: CardSet) -> Card {
        set_iter(hand)
            .max_by(|&a, &b| {
                out_logits[a as usize - 1]
                    .partial_cmp(&out_logits[b as usize - 1])
                    .expect("logits are finite")
            })
            .expect("hand is non-empty")
    }

    fn sample_masked_policy(logits: &[f32], hand: CardSet, rng: &mut SplitMix64) -> Card {
        let cards: Vec<Card> = set_iter(hand).collect();
        let max = cards
            .iter()
            .map(|&c| logits[c as usize - 1])
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = cards
            .iter()
            .map(|&c| ((logits[c as usize - 1] - max).exp()) as f64)
            .collect();
        cards[weighted_choice(&weights, rng)]
    }

    /// Sample a full deal consistent with `view`, assigning each unseen card
    /// to an opponent or the stock according to the belief head, under the
    /// hard constraint that every opponent holds exactly `hand_count` cards.
    fn sample_world(
        &self,
        view: &View,
        belief: &crate::neural::NeuralOutput,
        rng: &mut SplitMix64,
    ) -> Game {
        let n = view.num_players as usize;
        let me = view.player as usize;
        let hand_count = set_len(view.hand) as usize;
        let mut order: Vec<Card> = set_iter(view.unseen()).collect();
        rng.shuffle(&mut order);

        let mut caps = [0usize; BELIEF_CLASSES];
        caps[..n - 1].fill(hand_count);
        caps[BELIEF_CLASSES - 1] = order.len() - (n - 1) * hand_count;

        let mut hands = vec![0 as CardSet; n];
        hands[me] = view.hand;
        for &card in &order {
            let probs = NeuralNet::belief_probs(belief, card as usize - 1);
            let weights: Vec<f64> = (0..BELIEF_CLASSES)
                .map(|k| {
                    if caps[k] > 0 {
                        probs[k] as f64 + 1e-6
                    } else {
                        0.0
                    }
                })
                .collect();
            let class = weighted_choice(&weights, rng);
            caps[class] -= 1;
            if class < BELIEF_CLASSES - 1 {
                set_insert(&mut hands[(me + class + 1) % n], card);
            }
        }

        Game::from_position(
            hands,
            view.rows.clone(),
            view.penalties.to_vec(),
            view.played,
            view.turn,
        )
        .expect("view state is always valid")
    }
}

/// Index sampled proportionally to `weights` (at least one must be > 0).
fn weighted_choice(weights: &[f64], rng: &mut SplitMix64) -> usize {
    let total: f64 = weights.iter().sum();
    debug_assert!(total > 0.0);
    let mut r = (rng.next_u64() as f64 / u64::MAX as f64) * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 && w > 0.0 {
            return i;
        }
    }
    weights
        .iter()
        .rposition(|&w| w > 0.0)
        .expect("positive weight")
}

impl Bot for NeuralSearchBot {
    fn choose_card(&mut self, view: &View, rng: &mut SplitMix64) -> Card {
        let candidates: Vec<Card> = set_iter(view.hand).collect();
        if candidates.len() == 1 {
            return candidates[0];
        }
        let mut obs = vec![0.0f32; OBS_LEN];
        encode_view(view, None, &mut obs);
        let root = self.net.forward(&obs);

        let n = view.num_players as usize;
        // The belief head is trained for 4 players; elsewhere play raw policy.
        if self.worlds == 0 || n != 4 {
            return Self::policy_argmax(&root.policy_logits, view.hand);
        }

        let me = view.player as usize;
        let start: Vec<u16> = view.penalties.to_vec();
        let mut totals = vec![0.0f64; candidates.len()];
        let mut scratch = vec![0.0f32; OBS_LEN];

        for _ in 0..self.worlds {
            let world = self.sample_world(view, &root, rng);
            // Opponents choose simultaneously: their cards cannot depend on
            // ours, so sample once per world and reuse for every candidate.
            let mut opp_cards: Vec<Card> = vec![0; n];
            for (seat, card) in opp_cards.iter_mut().enumerate() {
                if seat == me {
                    continue;
                }
                encode_view(&world.view(seat), None, &mut scratch);
                let out = self.net.forward(&scratch);
                *card = Self::sample_masked_policy(&out.policy_logits, world.hand(seat), rng);
            }

            for (i, &candidate) in candidates.iter().enumerate() {
                let mut g = world.clone();
                let mut cards = opp_cards.clone();
                cards[me] = candidate;
                g.play_cards(&cards).expect("candidate is legal");
                while let Phase::ChooseRow { .. } = g.phase() {
                    let row = min_row_bulls(g.rows()).0;
                    g.choose_row(row).expect("row choice is legal");
                }
                let my_delta = (g.penalties()[me] - start[me]) as f64;
                let opp_delta: f64 = (0..n)
                    .filter(|&p| p != me)
                    .map(|p| (g.penalties()[p] - start[p]) as f64)
                    .sum();
                let reward = opp_delta / (n - 1) as f64 - my_delta;
                let future = if g.is_terminal() {
                    0.0
                } else {
                    encode_view(&g.view(me), None, &mut scratch);
                    self.net.forward(&scratch).value as f64
                };
                totals[i] += reward + future;
            }
        }

        let best = totals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("scores are finite"))
            .expect("candidates is non-empty")
            .0;
        candidates[best]
    }

    fn choose_row(&mut self, view: &View, _forced: Card, _rng: &mut SplitMix64) -> usize {
        min_row_bulls(view.rows).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::set_insert;

    fn hand_of(cards: &[Card]) -> CardSet {
        let mut s: CardSet = 0;
        for &c in cards {
            set_insert(&mut s, c);
        }
        s
    }

    #[test]
    fn greedy_prefers_free_placement() {
        // Row 2 is full: playing 35 would take it (9 bulls); 11 places free.
        let rows = [vec![10], vec![20], vec![30, 31, 32, 33, 34], vec![100]];
        let hand = hand_of(&[35, 11]);
        assert_eq!(greedy_card(hand, &rows), 11);
    }

    #[test]
    fn mc_bot_avoids_obvious_blunder() {
        // Row 2 is one card from full; playing 36 takes it. Playing 90 on
        // row 3 is free. Any sensible search must prefer 90.
        let rows = [vec![4], vec![15], vec![20, 25, 30, 31, 35], vec![80]];
        let mut penalties = vec![0u16; 4];
        penalties[0] = 0;
        let view = View {
            player: 0,
            num_players: 4,
            hand: hand_of(&[36, 90]),
            played: {
                let mut p: CardSet = 0;
                for &c in &[4, 15, 20, 25, 30, 31, 35, 80] {
                    set_insert(&mut p, c);
                }
                p
            },
            rows: &rows,
            penalties: &penalties,
            turn: 8,
        };
        let mut bot = McRolloutBot { worlds: 32 };
        let mut rng = SplitMix64::new(1);
        assert_eq!(bot.choose_card(&view, &mut rng), 90);
    }
}
