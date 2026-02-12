//! State threshold computation.
//!
//! Resolves wet/dry and wet/extreme thresholds from baseline precipitation
//! data and classifies daily precipitation into [`PrecipState`] values.

use crate::config::{MarkovConfig, ThresholdSpec};
use crate::error::MarkovError;
use crate::state::PrecipState;

/// Resolved precipitation thresholds for state classification.
///
/// The wet threshold is a single scalar applied globally. The extreme
/// thresholds are resolved per calendar month (1-indexed internally,
/// stored as a 12-element array indexed 0..12).
#[derive(Clone, Debug)]
pub struct StateThresholds {
    wet_threshold: f64,
    extreme_thresholds: [f64; 12],
}

impl StateThresholds {
    /// Resolve thresholds from baseline precipitation observations.
    ///
    /// # Arguments
    ///
    /// * `precip` - Daily precipitation values (may include zeros).
    /// * `months` - Corresponding 1-indexed calendar months (1 = January, ..., 12 = December).
    /// * `config` - Markov configuration containing threshold specifications.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError`] if the inputs are invalid (empty, mismatched
    /// lengths, non-finite values, or months outside 1..=12) or if the
    /// configuration fails validation.
    pub fn from_baseline(
        precip: &[f64],
        months: &[u8],
        config: &MarkovConfig,
    ) -> Result<Self, MarkovError> {
        // --- Validation ---
        if precip.is_empty() {
            return Err(MarkovError::EmptyData);
        }
        if precip.len() != months.len() {
            return Err(MarkovError::LengthMismatch {
                precip_len: precip.len(),
                months_len: months.len(),
            });
        }
        if precip.iter().any(|v| !v.is_finite()) {
            return Err(MarkovError::NonFiniteData);
        }
        for &m in months {
            if !(1..=12).contains(&m) {
                return Err(MarkovError::InvalidMonth { month: m });
            }
        }
        config.validate()?;

        // --- Wet threshold ---
        let wet_threshold = match config.wet_spec() {
            ThresholdSpec::Fixed(v) => v,
            ThresholdSpec::Quantile(q) => {
                let mut sorted: Vec<f64> = precip.to_vec();
                sorted.sort_by(|a, b| {
                    a.partial_cmp(b)
                        .expect("non-finite values excluded by validation")
                });
                quantile_type7(&sorted, q)
            }
        };

        // --- Extreme thresholds (per month) ---
        let extreme_thresholds = match config.extreme_spec() {
            ThresholdSpec::Fixed(v) => [v; 12],
            ThresholdSpec::Quantile(q) => {
                // Global fallback: quantile over all positive-precip days.
                let mut global_positive: Vec<f64> =
                    precip.iter().copied().filter(|&v| v > 0.0).collect();
                global_positive.sort_by(|a, b| {
                    a.partial_cmp(b)
                        .expect("non-finite values excluded by validation")
                });
                let global_fallback = if global_positive.is_empty() {
                    f64::INFINITY
                } else {
                    quantile_type7(&global_positive, q)
                };

                let mut thresholds = [0.0_f64; 12];
                for m in 0..12u8 {
                    let month_1 = m + 1;
                    // Collect positive-precip values for this month.
                    let mut month_positive: Vec<f64> = precip
                        .iter()
                        .zip(months.iter())
                        .filter(|&(&p, &mo)| mo == month_1 && p > 0.0)
                        .map(|(&p, _)| p)
                        .collect();

                    if month_positive.is_empty() {
                        // Check if the month has ANY data at all.
                        let has_any_data = months.contains(&month_1);
                        if has_any_data {
                            // Month exists but all values are zero → infinity.
                            thresholds[m as usize] = f64::INFINITY;
                        } else {
                            // Month has no data at all → global fallback.
                            thresholds[m as usize] = global_fallback;
                        }
                    } else {
                        month_positive.sort_by(|a, b| {
                            a.partial_cmp(b)
                                .expect("non-finite values excluded by validation")
                        });
                        thresholds[m as usize] = quantile_type7(&month_positive, q);
                    }
                }
                thresholds
            }
        };

        Ok(Self {
            wet_threshold,
            extreme_thresholds,
        })
    }

    /// Classify a single precipitation value into a [`PrecipState`].
    ///
    /// # Precondition
    ///
    /// `month` must be in 1..=12. This is **not** validated at runtime
    /// (hot path).
    #[inline]
    pub fn classify(&self, precip: f64, month: u8) -> PrecipState {
        if precip <= self.wet_threshold {
            PrecipState::Dry
        } else if precip > self.extreme_thresholds[month as usize - 1] {
            PrecipState::Extreme
        } else {
            PrecipState::Wet
        }
    }

    /// Classify a series of precipitation values.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::LengthMismatch`] if `precip` and `months`
    /// differ in length, or [`MarkovError::InvalidMonth`] if any month is
    /// outside 1..=12.
    pub fn classify_series(
        &self,
        precip: &[f64],
        months: &[u8],
    ) -> Result<Vec<PrecipState>, MarkovError> {
        if precip.len() != months.len() {
            return Err(MarkovError::LengthMismatch {
                precip_len: precip.len(),
                months_len: months.len(),
            });
        }
        for &m in months {
            if !(1..=12).contains(&m) {
                return Err(MarkovError::InvalidMonth { month: m });
            }
        }
        Ok(precip
            .iter()
            .zip(months.iter())
            .map(|(&p, &m)| self.classify(p, m))
            .collect())
    }

    /// Returns the global wet/dry threshold.
    pub fn wet_threshold(&self) -> f64 {
        self.wet_threshold
    }

    /// Returns the extreme threshold for a 1-indexed month.
    pub fn extreme_threshold(&self, month: u8) -> f64 {
        self.extreme_thresholds[month as usize - 1]
    }

    /// Returns all 12 extreme thresholds (indexed 0 = January, ..., 11 = December).
    pub fn extreme_thresholds(&self) -> &[f64; 12] {
        &self.extreme_thresholds
    }
}

/// R's default quantile algorithm (type = 7).
///
/// **Expects pre-sorted input** (caller's responsibility).
///
/// # Panics
///
/// Panics if `sorted` is empty.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    assert!(
        !sorted.is_empty(),
        "quantile_type7: input must not be empty"
    );
    let n = sorted.len();
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - h.floor()) * (sorted[hi] - sorted[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build config with both specs as Fixed.
    fn fixed_config(wet: f64, extreme: f64) -> MarkovConfig {
        MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(wet))
            .with_extreme_spec(ThresholdSpec::Fixed(extreme))
    }

    /// Helper: build config with both specs as Quantile.
    fn quantile_config(wet_q: f64, extreme_q: f64) -> MarkovConfig {
        MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Quantile(wet_q))
            .with_extreme_spec(ThresholdSpec::Quantile(extreme_q))
    }

    // 1. fixed_thresholds
    #[test]
    fn fixed_thresholds() {
        let precip = vec![0.0, 1.0, 2.0, 5.0, 10.0];
        let months = vec![1u8, 1, 1, 1, 1];
        let config = fixed_config(0.3, 8.0);

        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();
        assert!((st.wet_threshold() - 0.3).abs() < f64::EPSILON);
        for m in 1..=12u8 {
            assert!((st.extreme_threshold(m) - 8.0).abs() < f64::EPSILON);
        }
    }

    // 2. quantile_resolution
    #[test]
    fn quantile_resolution() {
        // 10 values across 2 months, some zeros.
        // Month 1: 0.0, 1.0, 2.0, 3.0, 4.0
        // Month 2: 0.0, 0.0, 5.0, 6.0, 7.0
        let precip = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0];
        let months = vec![1u8, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        // wet_q = 0.5 over ALL values (including zeros).
        // sorted all: [0,0,0,1,2,3,4,5,6,7] → median = quantile(0.5)
        // h = 9*0.5 = 4.5 → lo=4, hi=5 → 2 + 0.5*(3-2) = 2.5
        let config = quantile_config(0.5, 0.8);
        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();
        assert!(
            (st.wet_threshold() - 2.5).abs() < 1e-10,
            "wet_threshold: expected 2.5, got {}",
            st.wet_threshold()
        );

        // Extreme quantile is computed over positive-precip per month (NOT precip > wet_threshold).
        // Month 1 positive: [1,2,3,4], q=0.8 → h=3*0.8=2.4, lo=2,hi=3 → 3+0.4*(4-3)=3.4
        assert!(
            (st.extreme_threshold(1) - 3.4).abs() < 1e-10,
            "extreme month 1: expected 3.4, got {}",
            st.extreme_threshold(1)
        );
        // Month 2 positive: [5,6,7], q=0.8 → h=2*0.8=1.6, lo=1,hi=2 → 6+0.6*(7-6)=6.6
        assert!(
            (st.extreme_threshold(2) - 6.6).abs() < 1e-10,
            "extreme month 2: expected 6.6, got {}",
            st.extreme_threshold(2)
        );
    }

    // 3. classify_boundaries
    #[test]
    fn classify_boundaries() {
        let precip = vec![0.0, 1.0, 5.0];
        let months = vec![1u8, 1, 1];
        let config = fixed_config(0.3, 8.0);
        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();

        // At wet threshold → Dry (<=)
        assert_eq!(st.classify(0.3, 1), PrecipState::Dry);
        // Just above wet threshold → Wet
        assert_eq!(st.classify(0.31, 1), PrecipState::Wet);
        // At extreme threshold → Wet (not Extreme, because > is required)
        assert_eq!(st.classify(8.0, 1), PrecipState::Wet);
        // Just above extreme threshold → Extreme
        assert_eq!(st.classify(8.01, 1), PrecipState::Extreme);
        // Well below wet → Dry
        assert_eq!(st.classify(0.0, 1), PrecipState::Dry);
        // Well above extreme → Extreme
        assert_eq!(st.classify(100.0, 1), PrecipState::Extreme);
    }

    // 4. classify_series_ok
    #[test]
    fn classify_series_ok() {
        let precip = vec![0.0, 1.0, 5.0, 10.0];
        let months = vec![1u8, 1, 1, 1];
        let config = fixed_config(0.3, 8.0);
        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();

        let states = st
            .classify_series(&[0.0, 0.3, 4.0, 9.0], &[1, 1, 1, 1])
            .unwrap();
        assert_eq!(
            states,
            vec![
                PrecipState::Dry,
                PrecipState::Dry,
                PrecipState::Wet,
                PrecipState::Extreme,
            ]
        );
    }

    // 5. empty_data_error
    #[test]
    fn empty_data_error() {
        let config = MarkovConfig::new();
        let result = StateThresholds::from_baseline(&[], &[], &config);
        assert!(matches!(result, Err(MarkovError::EmptyData)));
    }

    // 6. length_mismatch_error
    #[test]
    fn length_mismatch_error() {
        let config = MarkovConfig::new();
        let result = StateThresholds::from_baseline(&[1.0, 2.0], &[1u8], &config);
        assert!(matches!(result, Err(MarkovError::LengthMismatch { .. })));
    }

    // 7. non_finite_error
    #[test]
    fn non_finite_error() {
        let config = MarkovConfig::new();
        // NaN
        let result = StateThresholds::from_baseline(&[f64::NAN, 1.0], &[1u8, 1], &config);
        assert!(matches!(result, Err(MarkovError::NonFiniteData)));
        // Infinity
        let result = StateThresholds::from_baseline(&[f64::INFINITY, 1.0], &[1u8, 1], &config);
        assert!(matches!(result, Err(MarkovError::NonFiniteData)));
    }

    // 8. invalid_month_error
    #[test]
    fn invalid_month_error() {
        let config = MarkovConfig::new();
        // Month 0
        let result = StateThresholds::from_baseline(&[1.0], &[0u8], &config);
        assert!(matches!(
            result,
            Err(MarkovError::InvalidMonth { month: 0 })
        ));
        // Month 13
        let result = StateThresholds::from_baseline(&[1.0], &[13u8], &config);
        assert!(matches!(
            result,
            Err(MarkovError::InvalidMonth { month: 13 })
        ));
    }

    // 9. month_no_positive_precip
    #[test]
    fn month_no_positive_precip() {
        // Month 1 has only zeros → extreme threshold should be INFINITY.
        let precip = vec![0.0, 0.0, 5.0];
        let months = vec![1u8, 1, 2];
        let config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();
        assert!(
            st.extreme_threshold(1).is_infinite(),
            "month with all zeros should have INFINITY threshold, got {}",
            st.extreme_threshold(1)
        );
    }

    // 10. fallback_global_quantile
    #[test]
    fn fallback_global_quantile() {
        // Only months 1 and 2 have data. Month 3 has no data at all →
        // should fall back to global quantile of positive-precip.
        let precip = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let months = vec![1u8, 1, 1, 2, 2, 2];
        let config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let st = StateThresholds::from_baseline(&precip, &months, &config).unwrap();

        // Global positive: [2, 4, 6, 8, 10], q=0.8
        // h = 4*0.8 = 3.2, lo=3, hi=4 → 8 + 0.2*(10-8) = 8.4
        let expected_global = 8.4;
        assert!(
            (st.extreme_threshold(3) - expected_global).abs() < 1e-10,
            "month 3 (no data) should use global fallback {}, got {}",
            expected_global,
            st.extreme_threshold(3)
        );
    }

    // 11. quantile_type7_basic
    #[test]
    fn quantile_type7_basic() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        // p=0.0 → 1.0
        assert!((quantile_type7(&sorted, 0.0) - 1.0).abs() < 1e-10);
        // p=0.25 → h=1.0, lo=1, hi=2 → 2.0
        assert!((quantile_type7(&sorted, 0.25) - 2.0).abs() < 1e-10);
        // p=0.5 → h=2.0, lo=2, hi=3 → 3.0
        assert!((quantile_type7(&sorted, 0.5) - 3.0).abs() < 1e-10);
        // p=1.0 → h=4.0, lo=4, hi=4 → 5.0
        assert!((quantile_type7(&sorted, 1.0) - 5.0).abs() < 1e-10);
        // p=0.1 → h=0.4, lo=0, hi=1 → 1 + 0.4*(2-1) = 1.4
        assert!((quantile_type7(&sorted, 0.1) - 1.4).abs() < 1e-10);
        // Single element
        assert!((quantile_type7(&[42.0], 0.5) - 42.0).abs() < 1e-10);
    }
}
