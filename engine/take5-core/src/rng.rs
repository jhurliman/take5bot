/// SplitMix64: tiny, fast, deterministic PRNG. Good enough for dealing and
/// rollouts; zero dependencies so it works identically on native and WASM.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (bound > 0), via rejection sampling to
    /// avoid modulo bias.
    pub fn next_below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_below(i as u64 + 1) as usize;
            slice.swap(i, j);
        }
    }

    /// Pick a uniformly random element index for a non-empty slice.
    pub fn choose_index(&mut self, len: usize) -> usize {
        self.next_below(len as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn bounded() {
        let mut r = SplitMix64::new(7);
        for _ in 0..1000 {
            assert!(r.next_below(10) < 10);
        }
    }
}
