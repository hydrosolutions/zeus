//! Precipitation occurrence states for the three-state Markov chain.

/// Three-state precipitation classification.
///
/// Precipitation is classified into one of three mutually exclusive states
/// based on wet and extreme thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrecipState {
    /// Precipitation at or below the wet threshold.
    Dry = 0,
    /// Precipitation above the wet threshold but at or below the extreme threshold.
    Wet = 1,
    /// Precipitation above the extreme threshold.
    Extreme = 2,
}

impl PrecipState {
    /// All three states in index order.
    pub const ALL: [PrecipState; 3] = [Self::Dry, Self::Wet, Self::Extreme];

    /// Returns the zero-based index of this state (matches the `#[repr(u8)]` discriminant).
    pub fn as_index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_index_values() {
        assert_eq!(PrecipState::Dry.as_index(), 0);
        assert_eq!(PrecipState::Wet.as_index(), 1);
        assert_eq!(PrecipState::Extreme.as_index(), 2);
    }

    #[test]
    fn all_ordering() {
        assert_eq!(
            PrecipState::ALL,
            [PrecipState::Dry, PrecipState::Wet, PrecipState::Extreme]
        );
    }

    #[test]
    fn trait_assertions() {
        fn assert_copy<T: Copy>() {}
        fn assert_eq<T: Eq>() {}
        fn assert_hash<T: std::hash::Hash>() {}
        assert_copy::<PrecipState>();
        assert_eq::<PrecipState>();
        assert_hash::<PrecipState>();
    }
}
