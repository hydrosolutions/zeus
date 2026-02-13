//! Output type for resampling operations.

use zeus_markov::PrecipState;

/// Result of a single year's resampling: 365 observation indices.
#[derive(Debug, Clone)]
pub struct ResampleResult {
    indices: Vec<usize>,
    final_state: PrecipState,
    last_obs_idx: usize,
}

impl ResampleResult {
    /// Creates a new `ResampleResult`.
    #[allow(dead_code)]
    pub(crate) fn new(indices: Vec<usize>, final_state: PrecipState, last_obs_idx: usize) -> Self {
        Self {
            indices,
            final_state,
            last_obs_idx,
        }
    }

    /// Returns the resampled observation indices (length 365).
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Consumes the result and returns the indices.
    pub fn into_indices(self) -> Vec<usize> {
        self.indices
    }

    /// Returns the Markov state of the last day.
    pub fn final_state(&self) -> PrecipState {
        self.final_state
    }

    /// Returns the observation index of the last selected day.
    pub fn last_obs_idx(&self) -> usize {
        self.last_obs_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors() {
        let result = ResampleResult::new(vec![0, 1, 2], PrecipState::Wet, 42);
        assert_eq!(result.indices(), &[0, 1, 2]);
        assert_eq!(result.final_state(), PrecipState::Wet);
        assert_eq!(result.last_obs_idx(), 42);
    }

    #[test]
    fn into_indices() {
        let result = ResampleResult::new(vec![5, 10, 15], PrecipState::Dry, 15);
        let indices = result.into_indices();
        assert_eq!(indices, vec![5, 10, 15]);
    }

    #[test]
    fn clone() {
        let result = ResampleResult::new(vec![1], PrecipState::Extreme, 1);
        let cloned = result.clone();
        assert_eq!(cloned.indices(), result.indices());
        assert_eq!(cloned.final_state(), result.final_state());
        assert_eq!(cloned.last_obs_idx(), result.last_obs_idx());
    }
}
