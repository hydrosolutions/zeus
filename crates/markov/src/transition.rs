//! Transition matrix estimation for the three-state Markov chain.

use crate::config::MarkovConfig;
use crate::error::MarkovError;
use crate::state::PrecipState;
use crate::threshold::StateThresholds;

/// A 3x3 row-stochastic transition matrix.
///
/// Each row `i` contains the probabilities of transitioning from state `i`
/// to states 0, 1, and 2 respectively. Row sums are expected to be 1.0.
#[derive(Debug, Clone, Copy)]
pub struct TransitionMatrix {
    probs: [[f64; 3]; 3],
}

impl TransitionMatrix {
    /// Constructs a transition matrix directly from a 3x3 array.
    pub(crate) fn from_probs(probs: [[f64; 3]; 3]) -> Self {
        Self { probs }
    }

    /// Returns the transition probabilities from a given state.
    pub fn row(&self, from: PrecipState) -> &[f64; 3] {
        &self.probs[from.as_index()]
    }

    /// Returns the probability of transitioning from one state to another.
    pub fn prob(&self, from: PrecipState, to: PrecipState) -> f64 {
        self.probs[from.as_index()][to.as_index()]
    }

    /// Returns the full 3x3 probability matrix.
    pub fn probs(&self) -> &[[f64; 3]; 3] {
        &self.probs
    }

    /// Validates that the matrix is row-stochastic.
    ///
    /// Checks that all values are finite, in `[0, 1]`, and that each row
    /// sums to approximately 1.0 (tolerance: 1e-6).
    pub fn validate(&self) -> Result<(), MarkovError> {
        for (i, row) in self.probs.iter().enumerate() {
            let mut sum = 0.0;
            for (j, &p) in row.iter().enumerate() {
                if !p.is_finite() {
                    return Err(MarkovError::InvalidThreshold {
                        reason: format!("probs[{i}][{j}] is not finite: {p}"),
                    });
                }
                if !(0.0..=1.0).contains(&p) {
                    return Err(MarkovError::InvalidThreshold {
                        reason: format!("probs[{i}][{j}] = {p} is outside [0, 1]"),
                    });
                }
                sum += p;
            }
            if (sum - 1.0).abs() > 1e-6 {
                return Err(MarkovError::InvalidThreshold {
                    reason: format!("row {i} sums to {sum}, expected ~1.0"),
                });
            }
        }
        Ok(())
    }

    /// Samples the next state given the current state, using cumulative CDF.
    ///
    /// Draws a uniform random number and walks through the row's cumulative
    /// distribution, returning the first state whose cumulative probability
    /// meets or exceeds the draw. Falls back to the last state if rounding
    /// prevents a match.
    pub fn sample(&self, from: PrecipState, rng: &mut impl rand::Rng) -> PrecipState {
        let u: f64 = rng.random();
        let row = &self.probs[from.as_index()];
        let mut cumulative = 0.0;
        for &state in &PrecipState::ALL {
            cumulative += row[state.as_index()];
            if cumulative >= u {
                return state;
            }
        }
        // Fallback to last state (should only be reached due to floating-point rounding).
        PrecipState::Extreme
    }
}

/// Twelve month-specific transition matrices (1-indexed months).
///
/// Index 0 corresponds to January, index 11 to December.
#[derive(Debug, Clone)]
pub struct MonthlyTransitions {
    matrices: [TransitionMatrix; 12],
}

impl MonthlyTransitions {
    /// Constructs monthly transitions from an array of 12 matrices.
    pub(crate) fn from_matrices(matrices: [TransitionMatrix; 12]) -> Self {
        Self { matrices }
    }

    /// Returns the transition matrix for a 1-indexed month.
    ///
    /// # Panics
    ///
    /// Panics if `month` is 0 or greater than 12.
    pub fn for_month(&self, month: u8) -> &TransitionMatrix {
        assert!(
            (1..=12).contains(&month),
            "month must be 1..=12, got {month}"
        );
        &self.matrices[(month - 1) as usize]
    }

    /// Returns all 12 transition matrices.
    pub fn matrices(&self) -> &[TransitionMatrix; 12] {
        &self.matrices
    }
}

/// Normalizes a probability vector in-place, using a fallback if the sum is zero.
///
/// 1. Replaces non-finite and negative values with 0.0.
/// 2. If the sum is positive, divides each element by the sum.
/// 3. Otherwise, copies `fallback` into `probs`.
fn normalize_probs(probs: &mut [f64; 3], fallback: [f64; 3]) {
    // Step 1: sanitize
    for p in probs.iter_mut() {
        if !p.is_finite() || *p < 0.0 {
            *p = 0.0;
        }
    }
    // Step 2-3: normalize or fallback
    let s: f64 = probs.iter().sum();
    if s > 0.0 {
        for p in probs.iter_mut() {
            *p /= s;
        }
    } else {
        *probs = fallback;
    }
}

/// Estimates monthly transition matrices from precipitation data.
///
/// # Arguments
///
/// * `precip` - Daily precipitation values.
/// * `months` - 1-indexed calendar months corresponding to each day.
/// * `thresholds` - Resolved state thresholds for classification.
/// * `config` - Markov configuration (provides Dirichlet alpha and spell factors).
///
/// # Errors
///
/// Returns [`MarkovError`] if the inputs are empty, mismatched in length,
/// or contain fewer than 2 observations.
pub fn estimate_monthly_transitions(
    precip: &[f64],
    months: &[u8],
    thresholds: &StateThresholds,
    config: &MarkovConfig,
) -> Result<MonthlyTransitions, MarkovError> {
    // Validate inputs.
    if precip.is_empty() {
        return Err(MarkovError::EmptyData);
    }
    if precip.len() != months.len() {
        return Err(MarkovError::LengthMismatch {
            precip_len: precip.len(),
            months_len: months.len(),
        });
    }
    if precip.len() < 2 {
        return Err(MarkovError::InsufficientData {
            n: precip.len(),
            min: 2,
        });
    }

    // Classify the full series into states.
    let states = thresholds.classify_series(precip, months)?;

    let alpha = config.dirichlet_alpha();
    let dry_factors = config.dry_spell_factors();
    let wet_factors = config.wet_spell_factors();

    let mut matrices = [TransitionMatrix::from_probs([[0.0; 3]; 3]); 12];

    for m in 1..=12u8 {
        let mi = (m - 1) as usize;

        // Count transitions for this month.
        let mut counts = [[0.0_f64; 3]; 3];
        for t in 1..states.len() {
            if months[t] == m {
                counts[states[t - 1].as_index()][states[t].as_index()] += 1.0;
            }
        }

        // Total transitions for this month.
        let n_m: f64 = counts.iter().flat_map(|row| row.iter()).sum();

        let mut probs = [[0.0_f64; 3]; 3];

        if n_m == 0.0 {
            // No transitions observed for this month: all rows default to [1, 0, 0].
            probs.fill([1.0, 0.0, 0.0]);
        } else {
            let alpha_eff = alpha / n_m.sqrt();

            for i in 0..3 {
                let row_sum: f64 = counts[i].iter().sum();
                for j in 0..3 {
                    probs[i][j] = (counts[i][j] + alpha_eff) / (row_sum + 3.0 * alpha_eff);
                }
            }

            // Spell factor adjustments.
            let dry_factor = dry_factors[mi];
            let wet_factor = wet_factors[mi];

            // Dry row (i=0): divide transitions away from Dry by dry_factor.
            if (dry_factor - 1.0).abs() > 1e-10 {
                probs[0][1] /= dry_factor;
                probs[0][2] /= dry_factor;
                normalize_probs(&mut probs[0], [1.0, 0.0, 0.0]);
            }

            // Wet row (i=1): divide transition to Dry by wet_factor.
            if (wet_factor - 1.0).abs() > 1e-10 {
                probs[1][0] /= wet_factor;
                normalize_probs(&mut probs[1], [0.0, 1.0, 0.0]);
            }

            // Extreme row (i=2): no spell adjustment.
        }

        matrices[mi] = TransitionMatrix::from_probs(probs);
    }

    Ok(MonthlyTransitions::from_matrices(matrices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ThresholdSpec;

    // Helper to build StateThresholds for testing via from_baseline with Fixed specs.
    fn test_thresholds(wet: f64, extreme: f64) -> StateThresholds {
        let config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(wet))
            .with_extreme_spec(ThresholdSpec::Fixed(extreme));
        // Need some baseline data to create thresholds.
        StateThresholds::from_baseline(&[0.0, 1.0], &[1, 1], &config).unwrap()
    }

    // 1. normalize_probs_standard
    #[test]
    fn normalize_probs_standard() {
        let mut probs = [2.0, 3.0, 5.0];
        normalize_probs(&mut probs, [1.0, 0.0, 0.0]);
        assert!((probs[0] - 0.2).abs() < 1e-10);
        assert!((probs[1] - 0.3).abs() < 1e-10);
        assert!((probs[2] - 0.5).abs() < 1e-10);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // 2. normalize_probs_all_zero
    #[test]
    fn normalize_probs_all_zero() {
        let mut probs = [0.0, 0.0, 0.0];
        let fallback = [0.5, 0.3, 0.2];
        normalize_probs(&mut probs, fallback);
        assert_eq!(probs, fallback);
    }

    // 3. normalize_probs_nan
    #[test]
    fn normalize_probs_nan() {
        let mut probs = [f64::NAN, 2.0, 3.0];
        normalize_probs(&mut probs, [1.0, 0.0, 0.0]);
        assert!((probs[0] - 0.0).abs() < 1e-10);
        assert!((probs[1] - 0.4).abs() < 1e-10);
        assert!((probs[2] - 0.6).abs() < 1e-10);
    }

    // 4. normalize_probs_negative
    #[test]
    fn normalize_probs_negative() {
        let mut probs = [-1.0, 2.0, 3.0];
        normalize_probs(&mut probs, [1.0, 0.0, 0.0]);
        assert!((probs[0] - 0.0).abs() < 1e-10);
        assert!((probs[1] - 0.4).abs() < 1e-10);
        assert!((probs[2] - 0.6).abs() < 1e-10);
    }

    // 5. normalize_probs_infinity
    #[test]
    fn normalize_probs_infinity() {
        let mut probs = [f64::INFINITY, 2.0, 3.0];
        normalize_probs(&mut probs, [1.0, 0.0, 0.0]);
        assert!((probs[0] - 0.0).abs() < 1e-10);
        assert!((probs[1] - 0.4).abs() < 1e-10);
        assert!((probs[2] - 0.6).abs() < 1e-10);
    }

    // 6. transition_matrix_row_access
    #[test]
    fn transition_matrix_row_access() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        assert_eq!(tm.row(PrecipState::Dry), &[0.5, 0.3, 0.2]);
        assert_eq!(tm.row(PrecipState::Wet), &[0.1, 0.7, 0.2]);
        assert_eq!(tm.row(PrecipState::Extreme), &[0.2, 0.3, 0.5]);
    }

    // 7. transition_matrix_prob_access
    #[test]
    fn transition_matrix_prob_access() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        assert!((tm.prob(PrecipState::Dry, PrecipState::Wet) - 0.3).abs() < 1e-10);
        assert!((tm.prob(PrecipState::Wet, PrecipState::Extreme) - 0.2).abs() < 1e-10);
        assert!((tm.prob(PrecipState::Extreme, PrecipState::Dry) - 0.2).abs() < 1e-10);
    }

    // 8. transition_matrix_sample_distribution
    #[test]
    fn transition_matrix_sample_distribution() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            let s = tm.sample(PrecipState::Dry, &mut rng);
            counts[s.as_index()] += 1;
        }

        let f0 = counts[0] as f64 / n as f64;
        let f1 = counts[1] as f64 / n as f64;
        let f2 = counts[2] as f64 / n as f64;

        assert!(
            (f0 - 0.5).abs() < 0.03,
            "Dry frequency: {f0}, expected ~0.5"
        );
        assert!(
            (f1 - 0.3).abs() < 0.03,
            "Wet frequency: {f1}, expected ~0.3"
        );
        assert!(
            (f2 - 0.2).abs() < 0.03,
            "Extreme frequency: {f2}, expected ~0.2"
        );
    }

    // 9. transition_matrix_deterministic
    #[test]
    fn transition_matrix_deterministic() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let tm = TransitionMatrix::from_probs([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            assert_eq!(tm.sample(PrecipState::Dry, &mut rng), PrecipState::Dry);
        }
    }

    // 10. transition_matrix_validate_ok
    #[test]
    fn transition_matrix_validate_ok() {
        let tm = TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        assert!(tm.validate().is_ok());
    }

    // 11. transition_matrix_validate_bad_sum
    #[test]
    fn transition_matrix_validate_bad_sum() {
        let tm = TransitionMatrix::from_probs([
            [0.5, 0.3, 0.3], // sums to 1.1
            [0.1, 0.7, 0.2],
            [0.2, 0.3, 0.5],
        ]);
        assert!(tm.validate().is_err());
    }

    // 12. monthly_transitions_for_month
    #[test]
    fn monthly_transitions_for_month() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut matrices = [TransitionMatrix::from_probs(identity); 12];
        // Make month 6 (June) distinctive.
        matrices[5] =
            TransitionMatrix::from_probs([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]);
        let mt = MonthlyTransitions::from_matrices(matrices);

        // Check all months are accessible.
        for m in 1..=12u8 {
            let _ = mt.for_month(m);
        }

        // Check month 6 is the distinctive one.
        assert!((mt.for_month(6).prob(PrecipState::Dry, PrecipState::Wet) - 0.3).abs() < 1e-10);
        // Check month 1 is the identity.
        assert!((mt.for_month(1).prob(PrecipState::Dry, PrecipState::Dry) - 1.0).abs() < 1e-10);
    }

    // 13. estimate_all_dry
    #[test]
    fn estimate_all_dry() {
        // All precipitation is 0.0, so all states are Dry.
        let n = 100;
        let precip = vec![0.0; n];
        let months: Vec<u8> = (0..n).map(|i| (i % 12) as u8 + 1).collect();
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new();

        let mt = estimate_monthly_transitions(&precip, &months, &thresholds, &config).unwrap();

        // Every month should have high p(Dry->Dry).
        for m in 1..=12u8 {
            let p00 = mt.for_month(m).prob(PrecipState::Dry, PrecipState::Dry);
            assert!(p00 > 0.9, "month {m}: p00 = {p00}, expected > 0.9");
        }
    }

    // 14. estimate_known_sequence
    #[test]
    fn estimate_known_sequence() {
        // Hand-crafted sequence for January only (month=1).
        // States: Dry, Dry, Wet, Wet, Dry, Wet, Extreme, Dry
        // Transitions (for month=1): D->D, D->W, W->W, W->D, D->W, W->E, E->D
        // Counts: D->D=1, D->W=2, D->E=0, W->D=1, W->W=1, W->E=1, E->D=1, E->W=0, E->E=0
        // n_m = 7, alpha=1.0, alpha_eff = 1/sqrt(7)
        let precip = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 20.0, 0.0];
        let months = vec![1u8; 8];
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new(); // alpha=1.0, spell factors=1.0

        let mt = estimate_monthly_transitions(&precip, &months, &thresholds, &config).unwrap();

        let tm = mt.for_month(1);

        // n_m = 7, alpha_eff = 1.0 / sqrt(7)
        let alpha_eff = 1.0 / 7.0_f64.sqrt();

        // Dry row: counts = [1, 2, 0], row_sum = 3
        let row_sum_dry = 3.0;
        let expected_d0 = (1.0 + alpha_eff) / (row_sum_dry + 3.0 * alpha_eff);
        let expected_d1 = (2.0 + alpha_eff) / (row_sum_dry + 3.0 * alpha_eff);
        let expected_d2 = (0.0 + alpha_eff) / (row_sum_dry + 3.0 * alpha_eff);

        assert!(
            (tm.prob(PrecipState::Dry, PrecipState::Dry) - expected_d0).abs() < 1e-10,
            "D->D: got {}, expected {}",
            tm.prob(PrecipState::Dry, PrecipState::Dry),
            expected_d0
        );
        assert!(
            (tm.prob(PrecipState::Dry, PrecipState::Wet) - expected_d1).abs() < 1e-10,
            "D->W: got {}, expected {}",
            tm.prob(PrecipState::Dry, PrecipState::Wet),
            expected_d1
        );
        assert!(
            (tm.prob(PrecipState::Dry, PrecipState::Extreme) - expected_d2).abs() < 1e-10,
            "D->E: got {}, expected {}",
            tm.prob(PrecipState::Dry, PrecipState::Extreme),
            expected_d2
        );

        // Verify rows sum to 1.
        let row_sum: f64 = tm.row(PrecipState::Dry).iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }

    // 15. estimate_zero_transition_month
    #[test]
    fn estimate_zero_transition_month() {
        // Only January data; other months should fall back to [1, 0, 0].
        let precip = vec![0.0, 0.0, 1.0];
        let months = vec![1u8, 1, 1];
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new();

        let mt = estimate_monthly_transitions(&precip, &months, &thresholds, &config).unwrap();

        // Month 2 should have no transitions, so all rows = [1, 0, 0].
        let tm = mt.for_month(2);
        for state in &PrecipState::ALL {
            assert!(
                (tm.prob(*state, PrecipState::Dry) - 1.0).abs() < 1e-10,
                "month 2, state {:?}: p->Dry should be 1.0, got {}",
                state,
                tm.prob(*state, PrecipState::Dry)
            );
            assert!(
                tm.prob(*state, PrecipState::Wet).abs() < 1e-10,
                "month 2, state {:?}: p->Wet should be 0.0",
                state
            );
            assert!(
                tm.prob(*state, PrecipState::Extreme).abs() < 1e-10,
                "month 2, state {:?}: p->Extreme should be 0.0",
                state
            );
        }
    }

    // 16. estimate_spell_factors
    #[test]
    fn estimate_spell_factors() {
        // Create a sequence with mixed states for January.
        let precip = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0];
        let months = vec![1u8; 10];
        let thresholds = test_thresholds(0.3, 8.0);

        // First estimate without spell factors.
        let config_no_spell = MarkovConfig::new();
        let mt_no =
            estimate_monthly_transitions(&precip, &months, &thresholds, &config_no_spell).unwrap();

        // Now estimate with dry_factor = 2.0 for January.
        let mut dry_factors = [1.0; 12];
        dry_factors[0] = 2.0; // January
        let config_dry = MarkovConfig::new().with_dry_spell_factors(dry_factors);
        let mt_dry =
            estimate_monthly_transitions(&precip, &months, &thresholds, &config_dry).unwrap();

        // With dry_factor > 1, the probability of leaving Dry (D->W, D->E) should decrease,
        // so p(D->D) should increase relative to no spell factors.
        let p_dd_no = mt_no.for_month(1).prob(PrecipState::Dry, PrecipState::Dry);
        let p_dd_dry = mt_dry.for_month(1).prob(PrecipState::Dry, PrecipState::Dry);
        assert!(
            p_dd_dry > p_dd_no,
            "Dry spell factor should increase D->D: no_spell={p_dd_no}, with_spell={p_dd_dry}"
        );

        // Now test wet spell factor.
        let mut wet_factors = [1.0; 12];
        wet_factors[0] = 2.0; // January
        let config_wet = MarkovConfig::new().with_wet_spell_factors(wet_factors);
        let mt_wet =
            estimate_monthly_transitions(&precip, &months, &thresholds, &config_wet).unwrap();

        // With wet_factor > 1, the probability of leaving Wet to Dry (W->D) should decrease,
        // so p(W->W) should increase relative to no spell factors.
        let p_ww_no = mt_no.for_month(1).prob(PrecipState::Wet, PrecipState::Wet);
        let p_ww_wet = mt_wet.for_month(1).prob(PrecipState::Wet, PrecipState::Wet);
        assert!(
            p_ww_wet > p_ww_no,
            "Wet spell factor should increase W->W: no_spell={p_ww_no}, with_spell={p_ww_wet}"
        );
    }

    // 17. estimate_rows_sum_to_one
    #[test]
    fn estimate_rows_sum_to_one() {
        let n = 365;
        let precip: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 3.0).collect();
        let months: Vec<u8> = (0..n).map(|i| (i % 12) as u8 + 1).collect();
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new();

        let mt = estimate_monthly_transitions(&precip, &months, &thresholds, &config).unwrap();

        for m in 1..=12u8 {
            let tm = mt.for_month(m);
            for state in &PrecipState::ALL {
                let row_sum: f64 = tm.row(*state).iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-10,
                    "month {m}, state {:?}: row sum = {row_sum}",
                    state
                );
            }
        }
    }

    // 18. estimate_empty_error
    #[test]
    fn estimate_empty_error() {
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new();
        let result = estimate_monthly_transitions(&[], &[], &thresholds, &config);
        assert!(matches!(result, Err(MarkovError::EmptyData)));
    }

    // 19. estimate_insufficient_data
    #[test]
    fn estimate_insufficient_data() {
        let thresholds = test_thresholds(0.3, 8.0);
        let config = MarkovConfig::new();
        let result = estimate_monthly_transitions(&[0.0], &[1], &thresholds, &config);
        assert!(matches!(
            result,
            Err(MarkovError::InsufficientData { n: 1, min: 2 })
        ));
    }
}
