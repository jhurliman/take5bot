use crate::cards::{
    bullheads, set_contains, set_insert, set_iter, set_len, set_remove, Card, CardSet, FULL_DECK,
    NUM_CARDS,
};
use crate::rng::SplitMix64;

pub const ROWS: usize = 4;
pub const MAX_ROW_LEN: usize = 5;
pub const HAND_SIZE: usize = 10;
pub const MAX_PLAYERS: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// All players simultaneously pick a card from hand.
    Select,
    /// `player`'s revealed `card` is lower than every row end; they must pick
    /// a row to take before resolution continues.
    ChooseRow {
        player: u8,
        card: Card,
    },
    Terminal,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameError {
    WrongPhase,
    CardNotInHand { player: u8, card: Card },
    InvalidRow,
    InvalidArgument,
}

/// What one seat is allowed to see. Bots and observation encoding only ever
/// receive a `View`, which makes information leaks a type-level impossibility.
pub struct View<'a> {
    pub player: u8,
    pub num_players: u8,
    pub hand: CardSet,
    /// Every card that has been publicly revealed: initial row starters plus
    /// every card played in a reveal (including cards now buried in penalty
    /// piles). This is the card-counting signal.
    pub played: CardSet,
    pub rows: &'a [Vec<Card>; ROWS],
    pub penalties: &'a [u16],
    pub turn: u8,
}

impl View<'_> {
    /// Cards this seat cannot see: not public and not in own hand. Opponent
    /// hands are a uniform unknown subset of these (plus undealt stock).
    pub fn unseen(&self) -> CardSet {
        FULL_DECK & !(self.played | self.hand)
    }
}

/// A single deal (round) of Take 5. Scores are penalty bullheads collected;
/// lower is better.
#[derive(Clone, Debug)]
pub struct Game {
    num_players: u8,
    hands: Vec<CardSet>,
    rows: [Vec<Card>; ROWS],
    penalties: Vec<u16>,
    played: CardSet,
    turn: u8,
    phase: Phase,
    /// Revealed cards awaiting placement, ascending; index 0 is next.
    pending: Vec<(Card, u8)>,
}

impl Game {
    /// Shuffle and deal a fresh round.
    pub fn deal(num_players: usize, seed: u64) -> Result<Game, GameError> {
        if !(2..=MAX_PLAYERS).contains(&num_players) {
            return Err(GameError::InvalidArgument);
        }
        let mut deck: Vec<Card> = (1..=NUM_CARDS as Card).collect();
        let mut rng = SplitMix64::new(seed);
        rng.shuffle(&mut deck);

        let mut hands = vec![0 as CardSet; num_players];
        for (p, hand) in hands.iter_mut().enumerate() {
            for i in 0..HAND_SIZE {
                set_insert(hand, deck[p * HAND_SIZE + i]);
            }
        }
        let start = num_players * HAND_SIZE;
        let starters: Vec<Card> = deck[start..start + ROWS].to_vec();
        Self::from_state(hands, starters, vec![0; num_players], 0)
    }

    /// Construct from explicit state. Used by parity tests (mirroring a deal
    /// made elsewhere) and by determinization (sampling opponent hands).
    pub fn from_state(
        hands: Vec<CardSet>,
        row_starters: Vec<Card>,
        penalties: Vec<u16>,
        turn: u8,
    ) -> Result<Game, GameError> {
        let num_players = hands.len();
        if !(2..=MAX_PLAYERS).contains(&num_players)
            || row_starters.len() != ROWS
            || penalties.len() != num_players
            || hands.iter().any(|h| h & !FULL_DECK != 0)
        {
            return Err(GameError::InvalidArgument);
        }
        let mut played: CardSet = 0;
        for &c in &row_starters {
            // Reject out-of-range ids, duplicate starters, and starters that
            // are simultaneously claimed as hand cards.
            if !(1..=NUM_CARDS as Card).contains(&c)
                || set_contains(played, c)
                || hands.iter().any(|h| set_contains(*h, c))
            {
                return Err(GameError::InvalidArgument);
            }
            set_insert(&mut played, c);
        }
        let rows = [
            vec![row_starters[0]],
            vec![row_starters[1]],
            vec![row_starters[2]],
            vec![row_starters[3]],
        ];
        Ok(Game {
            num_players: num_players as u8,
            hands,
            rows,
            penalties,
            played,
            turn,
            phase: Phase::Select,
            pending: Vec::new(),
        })
    }

    /// Construct mid-deal with arbitrary row contents (for determinization
    /// from an observed position).
    pub fn from_position(
        hands: Vec<CardSet>,
        rows: [Vec<Card>; ROWS],
        penalties: Vec<u16>,
        played: CardSet,
        turn: u8,
    ) -> Result<Game, GameError> {
        let num_players = hands.len();
        if !(2..=MAX_PLAYERS).contains(&num_players)
            || penalties.len() != num_players
            || hands.iter().any(|h| h & !FULL_DECK != 0)
        {
            return Err(GameError::InvalidArgument);
        }
        if rows.iter().any(|r| {
            r.is_empty()
                || r.len() > MAX_ROW_LEN
                || r.iter().any(|&c| !(1..=NUM_CARDS as Card).contains(&c))
        }) {
            return Err(GameError::InvalidArgument);
        }
        Ok(Game {
            num_players: num_players as u8,
            hands,
            rows,
            penalties,
            played,
            turn,
            phase: Phase::Select,
            pending: Vec::new(),
        })
    }

    // ---------------------------------------------------------- accessors

    pub fn num_players(&self) -> usize {
        self.num_players as usize
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn is_terminal(&self) -> bool {
        self.phase == Phase::Terminal
    }

    pub fn turn(&self) -> u8 {
        self.turn
    }

    pub fn rows(&self) -> &[Vec<Card>; ROWS] {
        &self.rows
    }

    pub fn penalties(&self) -> &[u16] {
        &self.penalties
    }

    pub fn hand(&self, player: usize) -> CardSet {
        self.hands[player]
    }

    pub fn hand_cards(&self, player: usize) -> Vec<Card> {
        set_iter(self.hands[player]).collect()
    }

    pub fn played(&self) -> CardSet {
        self.played
    }

    pub fn view(&self, player: usize) -> View<'_> {
        View {
            player: player as u8,
            num_players: self.num_players,
            hand: self.hands[player],
            played: self.played,
            rows: &self.rows,
            penalties: &self.penalties,
            turn: self.turn,
        }
    }

    /// Final scores as negative penalties (higher is better), matching the
    /// OpenSpiel convention of maximizing utility.
    pub fn returns(&self) -> Vec<f32> {
        self.penalties.iter().map(|&p| -(p as f32)).collect()
    }

    // ------------------------------------------------------------ actions

    /// Simultaneous reveal: one card per player, in seat order. Resolution
    /// runs ascending and may pause in `ChooseRow`.
    pub fn play_cards(&mut self, cards: &[Card]) -> Result<(), GameError> {
        if self.phase != Phase::Select {
            return Err(GameError::WrongPhase);
        }
        if cards.len() != self.num_players as usize {
            return Err(GameError::InvalidArgument);
        }
        for (p, &card) in cards.iter().enumerate() {
            if !set_contains(self.hands[p], card) {
                return Err(GameError::CardNotInHand {
                    player: p as u8,
                    card,
                });
            }
        }
        for (p, &card) in cards.iter().enumerate() {
            set_remove(&mut self.hands[p], card);
            set_insert(&mut self.played, card);
            self.pending.push((card, p as u8));
        }
        self.pending.sort_unstable();
        self.resolve();
        Ok(())
    }

    /// Resolve a pending `ChooseRow`: the player takes `row` and their card
    /// starts it fresh.
    pub fn choose_row(&mut self, row: usize) -> Result<(), GameError> {
        let Phase::ChooseRow { player, card } = self.phase else {
            return Err(GameError::WrongPhase);
        };
        if row >= ROWS {
            return Err(GameError::InvalidRow);
        }
        self.take_row(player as usize, row, card);
        self.phase = Phase::Select;
        self.resolve();
        Ok(())
    }

    // ------------------------------------------------------------ internal

    fn take_row(&mut self, player: usize, row: usize, replacement: Card) {
        let bulls: u16 = self.rows[row].iter().map(|&c| bullheads(c) as u16).sum();
        self.penalties[player] += bulls;
        self.rows[row].clear();
        self.rows[row].push(replacement);
    }

    /// Row the card must go to: highest row-end lower than the card, or None.
    pub fn target_row(rows: &[Vec<Card>; ROWS], card: Card) -> Option<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, r)| *r.last().expect("rows are never empty") < card)
            .max_by_key(|(_, r)| *r.last().expect("rows are never empty"))
            .map(|(idx, _)| idx)
    }

    fn resolve(&mut self) {
        while !self.pending.is_empty() {
            let (card, player) = self.pending[0];
            match Self::target_row(&self.rows, card) {
                None => {
                    self.pending.remove(0);
                    self.phase = Phase::ChooseRow { player, card };
                    return;
                }
                Some(row) => {
                    self.pending.remove(0);
                    if self.rows[row].len() == MAX_ROW_LEN {
                        self.take_row(player as usize, row, card);
                    } else {
                        self.rows[row].push(card);
                    }
                }
            }
        }
        if self.hands.iter().all(|h| set_len(*h) == 0) {
            self.phase = Phase::Terminal;
        } else {
            self.phase = Phase::Select;
            self.turn += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand_of(cards: &[Card]) -> CardSet {
        let mut s: CardSet = 0;
        for &c in cards {
            set_insert(&mut s, c);
        }
        s
    }

    #[test]
    fn deal_shapes() {
        let g = Game::deal(4, 123).unwrap();
        assert_eq!(g.num_players(), 4);
        for p in 0..4 {
            assert_eq!(g.hand_cards(p).len(), HAND_SIZE);
        }
        for row in g.rows() {
            assert_eq!(row.len(), 1);
        }
        // 44 dealt cards are distinct.
        let mut all: Vec<Card> = (0..4).flat_map(|p| g.hand_cards(p)).collect();
        all.extend(g.rows().iter().map(|r| r[0]));
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 44);
        assert_eq!(set_len(g.played()), 4);
    }

    #[test]
    fn placement_and_sixth_card_takes_row() {
        // Rows start 10/20/30/40. Fill row 0 to five cards; the sixth takes it.
        let hands = vec![hand_of(&[11, 17, 19]), hand_of(&[16, 18, 21])];
        let mut g = Game::from_state(hands, vec![10, 20, 30, 40], vec![0, 0], 7).unwrap();

        g.play_cards(&[11, 16]).unwrap();
        assert_eq!(g.rows()[0], vec![10, 11, 16]);

        g.play_cards(&[17, 18]).unwrap();
        assert_eq!(g.rows()[0], vec![10, 11, 16, 17, 18]);

        // 19 resolves first (ascending), row 0 is full: player 0 takes
        // 3+5+1+1+1 = 11 bulls and 19 starts the row fresh. 21 then goes to
        // row 1 (end 20 is the highest end below 21, beating the fresh 19).
        g.play_cards(&[19, 21]).unwrap();
        assert_eq!(g.rows()[0], vec![19]);
        assert_eq!(g.rows()[1], vec![20, 21]);
        assert_eq!(g.penalties(), &[11, 0]);
        assert!(g.is_terminal());
        assert_eq!(g.returns(), vec![-11.0, 0.0]);
    }

    #[test]
    fn low_card_forces_row_choice_in_ascending_order() {
        // Seat order differs from resolution order: player 1's 5 resolves
        // before player 0's 71.
        let hands = vec![hand_of(&[71]), hand_of(&[5])];
        let mut g = Game::from_state(hands, vec![50, 60, 70, 80], vec![0, 0], 9).unwrap();
        g.play_cards(&[71, 5]).unwrap();
        assert_eq!(g.phase(), Phase::ChooseRow { player: 1, card: 5 });

        // Player 1 takes row 2 (just the 70, worth 3 bulls); 5 starts it.
        g.choose_row(2).unwrap();
        assert_eq!(g.rows()[2], vec![5]);
        assert_eq!(g.penalties(), &[0, 3]);
        // 71 then lands after the highest lower row end, 60.
        assert_eq!(g.rows()[1], vec![60, 71]);
        assert!(g.is_terminal());
    }

    #[test]
    fn view_hides_opponent_hands() {
        let hands = vec![hand_of(&[11, 17]), hand_of(&[16, 18])];
        let mut g = Game::from_state(hands, vec![10, 20, 30, 40], vec![0, 0], 0).unwrap();
        g.play_cards(&[11, 16]).unwrap();

        let v = g.view(0);
        // Played = 4 starters + 2 revealed cards.
        assert_eq!(set_len(v.played), 6);
        assert!(set_contains(v.played, 11) && set_contains(v.played, 16));
        // Unseen excludes own hand and public cards; opponent's 18 is unseen.
        assert!(set_contains(v.unseen(), 18));
        assert!(!set_contains(v.unseen(), 17));
        assert!(!set_contains(v.unseen(), 10));
        assert_eq!(set_len(v.unseen()), 104 - 6 - 1);
    }

    #[test]
    fn target_row_picks_highest_lower_end() {
        let rows = [vec![10], vec![30], vec![50], vec![70]];
        assert_eq!(Game::target_row(&rows, 55), Some(2));
        assert_eq!(Game::target_row(&rows, 71), Some(3));
        assert_eq!(Game::target_row(&rows, 11), Some(0));
        assert_eq!(Game::target_row(&rows, 9), None);
    }

    #[test]
    fn from_state_rejects_malformed_input() {
        let hands = || vec![hand_of(&[11]), hand_of(&[16])];
        let ok = |starters: Vec<Card>| Game::from_state(hands(), starters, vec![0, 0], 0);
        // Out-of-range starters (0 would wrap to bit 127; 105 is off-deck).
        assert_eq!(
            ok(vec![0, 20, 30, 40]).err(),
            Some(GameError::InvalidArgument)
        );
        assert_eq!(
            ok(vec![105, 20, 30, 40]).err(),
            Some(GameError::InvalidArgument)
        );
        // Duplicate starters and starters already claimed as hand cards.
        assert_eq!(
            ok(vec![20, 20, 30, 40]).err(),
            Some(GameError::InvalidArgument)
        );
        assert_eq!(
            ok(vec![11, 20, 30, 40]).err(),
            Some(GameError::InvalidArgument)
        );
        // Hand masks with bits beyond card 104.
        assert_eq!(
            Game::from_state(
                vec![1u128 << 120, hand_of(&[16])],
                vec![10, 20, 30, 40],
                vec![0, 0],
                0
            )
            .err(),
            Some(GameError::InvalidArgument)
        );
        assert!(ok(vec![10, 20, 30, 40]).is_ok());
    }

    #[test]
    fn illegal_actions_rejected() {
        let hands = vec![hand_of(&[11]), hand_of(&[16])];
        let mut g = Game::from_state(hands, vec![10, 20, 30, 40], vec![0, 0], 0).unwrap();
        assert_eq!(g.choose_row(0), Err(GameError::WrongPhase));
        assert_eq!(
            g.play_cards(&[12, 16]),
            Err(GameError::CardNotInHand {
                player: 0,
                card: 12
            })
        );
        // Failed joint action must not mutate state.
        assert_eq!(g.hand_cards(0), vec![11]);
        assert_eq!(g.hand_cards(1), vec![16]);
    }
}
