//! AIC-based ARMA model order selection.

use crate::error::ArmaError;
use crate::fit::ArmaFit;
#[allow(unused_imports)]
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
pub fn select_best_aic(_data: &[f64], _max_p: usize, _max_q: usize) -> Result<ArmaFit, ArmaError> {
    todo!()
}
