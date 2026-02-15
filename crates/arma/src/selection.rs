//! AIC-based ARMA model order selection.

use tracing::{debug, trace_span};

use crate::error::ArmaError;
use crate::fit::ArmaFit;
use crate::spec::ArmaSpec;

/// Selects the best ARMA(p,q) model from a grid search over orders
/// 0..=`max_p` and 0..=`max_q`, ranked by Akaike Information Criterion
/// (AIC).
///
/// Fits every candidate `(p, q)` via [`ArmaSpec::fit()`], collects
/// those that converge, and returns the [`ArmaFit`] with the lowest
/// [`ArmaFit::aic()`]. Candidates that fail to fit are silently
/// skipped.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`ArmaError::AllCandidatesFailed`] | every `(p, q)` combination failed to fit |
///
/// # Example
///
/// ```ignore
/// let best = select_best_aic(&data, 3, 2)?;
/// println!("Best order: {:?}, AIC = {}", best.order(), best.aic());
/// ```
#[tracing::instrument(skip(data), fields(n = data.len(), max_p, max_q))]
pub fn select_best_aic(data: &[f64], max_p: usize, max_q: usize) -> Result<ArmaFit, ArmaError> {
    let mut best: Option<ArmaFit> = None;

    for p in 0..=max_p {
        for q in 0..=max_q {
            let _candidate = trace_span!("arma_candidate", p, q).entered();
            match ArmaSpec::new(p, q).fit(data) {
                Ok(fit) => {
                    let dominated = best.as_ref().is_some_and(|b| b.aic() <= fit.aic());
                    if !dominated {
                        debug!(p, q, aic = fit.aic(), "new best candidate");
                        best = Some(fit);
                    }
                }
                Err(_) => continue,
            }
        }
    }

    best.ok_or(ArmaError::AllCandidatesFailed { max_p, max_q })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn basic_usage() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();
        let fit = select_best_aic(&data, 2, 2).unwrap();
        assert!(fit.log_likelihood().is_finite());
        assert!(fit.sigma2() > 0.0);
    }

    #[test]
    fn all_fail_returns_error() {
        // Constant data should fail for all orders
        let data = vec![5.0; 100];
        let result = select_best_aic(&data, 2, 2);
        assert!(matches!(result, Err(ArmaError::AllCandidatesFailed { .. })));
    }

    #[test]
    fn max_zero_gives_arma00() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng)).collect();
        let fit = select_best_aic(&data, 0, 0).unwrap();
        assert_eq!(fit.order(), (0, 0));
    }

    #[test]
    fn white_noise_prefers_simple() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
        let fit = select_best_aic(&data, 2, 2).unwrap();
        let (p, q) = fit.order();
        // White noise: AIC should prefer (0,0) or a simple model
        let total = p + q;
        assert!(total <= 2, "Expected simple model, got ({}, {})", p, q);
    }
}
