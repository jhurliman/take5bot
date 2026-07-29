//! Head-to-head match runner. Seats rotate every game so no bot benefits from
//! a fixed position, and every game is reproducible from (base_seed, index).

use crate::bots::BotSpec;
use crate::game::{Game, Phase};
use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct MatchResult {
    /// `seat_bots[seat]` = index into the spec list of the bot in that seat.
    pub seat_bots: Vec<usize>,
    /// Final penalty bullheads per seat (lower is better).
    pub penalties: Vec<u16>,
}

/// Run games `start..end` of a tournament. Splitting the range across threads
/// (done by callers on native targets) yields identical results to a single
/// sequential run because each game is seeded independently.
pub fn run_match_range(
    specs: &[BotSpec],
    start: u64,
    end: u64,
    base_seed: u64,
) -> Vec<MatchResult> {
    let n = specs.len();
    assert!(n >= 2, "need one bot spec per seat");
    let mut results = Vec::with_capacity((end - start) as usize);

    for g in start..end {
        // Decorrelate deal and bot randomness from the game index.
        let mut seeder = SplitMix64::new(base_seed ^ g.wrapping_mul(0x9E3779B97F4A7C15));
        let deal_seed = seeder.next_u64();
        let mut rng = SplitMix64::new(seeder.next_u64());

        // Rotate seats: seat i hosts spec (i + g) % n.
        let seat_bots: Vec<usize> = (0..n).map(|i| (i + g as usize) % n).collect();
        let mut bots: Vec<Box<dyn crate::bots::Bot + Send + Sync>> =
            seat_bots.iter().map(|&b| specs[b].build()).collect();

        let mut game = Game::deal(n, deal_seed).expect("valid player count");
        loop {
            match game.phase() {
                Phase::Terminal => break,
                Phase::Select => {
                    let cards: Vec<u8> = (0..n)
                        .map(|p| bots[p].choose_card(&game.view(p), &mut rng))
                        .collect();
                    game.play_cards(&cards).expect("bots return legal cards");
                }
                Phase::ChooseRow { player, card } => {
                    let p = player as usize;
                    let row = bots[p].choose_row(&game.view(p), card, &mut rng);
                    game.choose_row(row).expect("bots return legal rows");
                }
            }
        }

        results.push(MatchResult {
            seat_bots,
            penalties: game.penalties().to_vec(),
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seat_rotating() {
        let specs = [
            BotSpec::Random,
            BotSpec::Lowest,
            BotSpec::Greedy,
            BotSpec::Random,
        ];
        let a = run_match_range(&specs, 0, 20, 42);
        let b = run_match_range(&specs, 0, 20, 42);
        assert_eq!(a.len(), 20);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.penalties, y.penalties);
            assert_eq!(x.seat_bots, y.seat_bots);
        }
        // Rotation: game 1 shifts assignments by one seat.
        assert_eq!(a[0].seat_bots, vec![0, 1, 2, 3]);
        assert_eq!(a[1].seat_bots, vec![1, 2, 3, 0]);
        // Splitting the range reproduces the sequential run.
        let tail = run_match_range(&specs, 10, 20, 42);
        assert_eq!(a[10].penalties, tail[0].penalties);
    }

    #[test]
    fn greedy_beats_random_over_many_games() {
        let specs = [
            BotSpec::Greedy,
            BotSpec::Random,
            BotSpec::Random,
            BotSpec::Random,
        ];
        let results = run_match_range(&specs, 0, 300, 7);
        let mut totals = [0f64; 4]; // per bot index
        for r in &results {
            for (seat, &bot) in r.seat_bots.iter().enumerate() {
                totals[bot] += r.penalties[seat] as f64;
            }
        }
        // Greedy should collect clearly fewer bulls than the random bots.
        assert!(
            totals[0] < totals[1] * 0.8,
            "greedy={} random={}",
            totals[0],
            totals[1]
        );
    }
}
