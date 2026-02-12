//! Markov chain state simulation.

use crate::error::MarkovError;
use crate::state::PrecipState;
use crate::transition::MonthlyTransitions;

/// Simulates a sequence of precipitation states.
///
/// # Arguments
///
/// * `transitions` - Monthly transition matrices.
/// * `sim_months` - 1-indexed month for each simulated day.
/// * `initial` - The state of the day before the first simulated day.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A vector of [`PrecipState`] values with the same length as `sim_months`.
pub fn simulate_states(
    transitions: &MonthlyTransitions,
    sim_months: &[u8],
    initial: PrecipState,
    rng: &mut impl rand::Rng,
) -> Vec<PrecipState> {
    let mut out = vec![PrecipState::Dry; sim_months.len()];
    // Delegate to _into; unwrap is safe because we sized the buffer correctly.
    simulate_states_into(transitions, sim_months, initial, rng, &mut out)
        .expect("buffer length matches sim_months length");
    out
}

/// Simulates precipitation states into a pre-allocated buffer.
///
/// # Arguments
///
/// * `transitions` - Monthly transition matrices.
/// * `sim_months` - 1-indexed month for each simulated day.
/// * `initial` - The state of the day before the first simulated day.
/// * `rng` - Random number generator.
/// * `out` - Pre-allocated output buffer; must have the same length as `sim_months`.
///
/// # Errors
///
/// Returns [`MarkovError::BufferLengthMismatch`] if `out.len() != sim_months.len()`.
pub fn simulate_states_into(
    transitions: &MonthlyTransitions,
    sim_months: &[u8],
    initial: PrecipState,
    rng: &mut impl rand::Rng,
    out: &mut [PrecipState],
) -> Result<(), MarkovError> {
    if out.len() != sim_months.len() {
        return Err(MarkovError::BufferLengthMismatch {
            expected: sim_months.len(),
            got: out.len(),
        });
    }
    let mut prev = initial;
    for (i, &m) in sim_months.iter().enumerate() {
        let next = transitions.for_month(m).sample(prev, rng);
        out[i] = next;
        prev = next;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transition::TransitionMatrix;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Creates a `MonthlyTransitions` where every month uses the same matrix.
    fn uniform_monthly(matrix: TransitionMatrix) -> MonthlyTransitions {
        MonthlyTransitions::from_matrices([matrix; 12])
    }

    // 1. length_correctness
    #[test]
    fn length_correctness() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        let mt = uniform_monthly(tm);
        let months = vec![1u8; 100];
        let mut rng = StdRng::seed_from_u64(42);

        let result = simulate_states(&mt, &months, PrecipState::Dry, &mut rng);
        assert_eq!(result.len(), 100);
    }

    // 2. empty_sim_months
    #[test]
    fn empty_sim_months() {
        let tm = TransitionMatrix::from_probs([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mt = uniform_monthly(tm);
        let mut rng = StdRng::seed_from_u64(42);

        let result = simulate_states(&mt, &[], PrecipState::Dry, &mut rng);
        assert!(result.is_empty());
    }

    // 3. deterministic_with_seed
    #[test]
    fn deterministic_with_seed() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        let mt = uniform_monthly(tm);
        let months: Vec<u8> = (0..50).map(|i| (i % 12) as u8 + 1).collect();

        let mut rng1 = StdRng::seed_from_u64(123);
        let result1 = simulate_states(&mt, &months, PrecipState::Wet, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(123);
        let result2 = simulate_states(&mt, &months, PrecipState::Wet, &mut rng2);

        assert_eq!(result1, result2);
    }

    // 4. identity_preserves_state
    #[test]
    fn identity_preserves_state() {
        let tm = TransitionMatrix::from_probs([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mt = uniform_monthly(tm);
        let months = vec![1u8; 50];
        let mut rng = StdRng::seed_from_u64(42);

        let result = simulate_states(&mt, &months, PrecipState::Dry, &mut rng);
        assert!(
            result.iter().all(|&s| s == PrecipState::Dry),
            "identity matrix from Dry should produce all Dry"
        );
    }

    // 5. into_matches_allocating
    #[test]
    fn into_matches_allocating() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        let mt = uniform_monthly(tm);
        let months: Vec<u8> = (0..30).map(|i| (i % 12) as u8 + 1).collect();

        let mut rng1 = StdRng::seed_from_u64(999);
        let alloc_result = simulate_states(&mt, &months, PrecipState::Extreme, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(999);
        let mut buf = vec![PrecipState::Dry; months.len()];
        simulate_states_into(&mt, &months, PrecipState::Extreme, &mut rng2, &mut buf).unwrap();

        assert_eq!(alloc_result, buf);
    }

    // 6. buffer_mismatch_error
    #[test]
    fn buffer_mismatch_error() {
        let tm = TransitionMatrix::from_probs([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mt = uniform_monthly(tm);
        let months = vec![1u8; 10];
        let mut rng = StdRng::seed_from_u64(42);
        let mut buf = vec![PrecipState::Dry; 5]; // wrong size

        let result = simulate_states_into(&mt, &months, PrecipState::Dry, &mut rng, &mut buf);
        assert!(matches!(
            result,
            Err(MarkovError::BufferLengthMismatch {
                expected: 10,
                got: 5
            })
        ));
    }

    // 7. distribution_test
    #[test]
    fn distribution_test() {
        // Matrix that always transitions Dry -> Wet, Wet -> Extreme, Extreme -> Dry.
        // So cycling: D, W, E, D, W, E, ...
        // With a stochastic matrix, check frequency is plausible.
        let tm = TransitionMatrix::from_probs([[0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.5, 0.3, 0.2]]);
        let mt = uniform_monthly(tm);
        let n = 10_000;
        let months = vec![1u8; n];
        let mut rng = StdRng::seed_from_u64(42);

        let result = simulate_states(&mt, &months, PrecipState::Dry, &mut rng);

        let mut counts = [0usize; 3];
        for s in &result {
            counts[s.as_index()] += 1;
        }

        // With this symmetric-ish matrix, frequencies should be roughly equal (~1/3 each).
        let f0 = counts[0] as f64 / n as f64;
        let f1 = counts[1] as f64 / n as f64;
        let f2 = counts[2] as f64 / n as f64;

        assert!(
            (f0 - 1.0 / 3.0).abs() < 0.05,
            "Dry frequency: {f0}, expected ~0.33"
        );
        assert!(
            (f1 - 1.0 / 3.0).abs() < 0.05,
            "Wet frequency: {f1}, expected ~0.33"
        );
        assert!(
            (f2 - 1.0 / 3.0).abs() < 0.05,
            "Extreme frequency: {f2}, expected ~0.33"
        );
    }
}
