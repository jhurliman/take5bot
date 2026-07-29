/// Card identifier, 1..=104.
pub type Card = u8;

pub const NUM_CARDS: usize = 104;

/// Penalty ("bullhead") value printed on a card.
pub fn bullheads(card: Card) -> u8 {
    debug_assert!((1..=NUM_CARDS as u8).contains(&card));
    if card == 55 {
        7
    } else if card.is_multiple_of(11) {
        5
    } else if card.is_multiple_of(10) {
        3
    } else if card.is_multiple_of(5) {
        2
    } else {
        1
    }
}

/// Bitmask over cards: bit `i` set means card `i + 1` is present.
pub type CardSet = u128;

pub const FULL_DECK: CardSet = (1u128 << NUM_CARDS) - 1;

pub fn set_contains(set: CardSet, card: Card) -> bool {
    set & (1u128 << (card - 1)) != 0
}

pub fn set_insert(set: &mut CardSet, card: Card) {
    *set |= 1u128 << (card - 1);
}

pub fn set_remove(set: &mut CardSet, card: Card) {
    *set &= !(1u128 << (card - 1));
}

pub fn set_iter(set: CardSet) -> impl Iterator<Item = Card> {
    (1..=NUM_CARDS as u8).filter(move |c| set_contains(set, *c))
}

pub fn set_len(set: CardSet) -> u32 {
    set.count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bullhead_values() {
        assert_eq!(bullheads(55), 7);
        assert_eq!(bullheads(11), 5);
        assert_eq!(bullheads(22), 5);
        assert_eq!(bullheads(10), 3);
        assert_eq!(bullheads(20), 3);
        assert_eq!(bullheads(5), 2);
        assert_eq!(bullheads(15), 2);
        assert_eq!(bullheads(1), 1);
        assert_eq!(bullheads(104), 1);
        // Total bullheads in the deck is 171 per the rulebook.
        let total: u32 = (1..=NUM_CARDS as u8).map(|c| bullheads(c) as u32).sum();
        assert_eq!(total, 171);
    }
}
