//! Result type for quantile mapping.

use crate::fit::BaselineFit;
use crate::gamma::GammaParams;

/// The output of a quantile-mapping adjustment.
///
/// Contains the adjusted precipitation series together with the fitted
/// baseline parameters, per-year target parameters, and bookkeeping about
/// which calendar months were successfully transformed or skipped.
#[derive(Debug, Clone)]
pub struct QmResult {
    /// The adjusted precipitation values.
    adjusted: Vec<f64>,
    /// The fitted baseline Gamma parameters.
    baseline: BaselineFit,
    /// Target Gamma parameters per year (0-indexed year, 0-indexed month).
    target_params: Vec<[Option<GammaParams>; 12]>,
    /// 1-indexed months that were successfully transformed.
    perturbed_months: Vec<u8>,
    /// 1-indexed months that were skipped.
    skipped_months: Vec<u8>,
}

impl QmResult {
    /// Creates a new `QmResult` from its constituent parts.
    pub(crate) fn new(
        adjusted: Vec<f64>,
        baseline: BaselineFit,
        target_params: Vec<[Option<GammaParams>; 12]>,
        perturbed_months: Vec<u8>,
        skipped_months: Vec<u8>,
    ) -> Self {
        Self {
            adjusted,
            baseline,
            target_params,
            perturbed_months,
            skipped_months,
        }
    }

    /// Returns the adjusted precipitation values as a slice.
    pub fn adjusted(&self) -> &[f64] {
        &self.adjusted
    }

    /// Consumes `self` and returns the owned adjusted precipitation vector.
    pub fn into_adjusted(self) -> Vec<f64> {
        self.adjusted
    }

    /// Returns a reference to the fitted baseline parameters.
    pub fn baseline(&self) -> &BaselineFit {
        &self.baseline
    }

    /// Returns the target Gamma parameters for all years.
    pub fn target_params(&self) -> &[[Option<GammaParams>; 12]] {
        &self.target_params
    }

    /// Returns the target Gamma parameters for a specific year and month.
    ///
    /// `year_idx` is 0-indexed, `month` is 1-indexed (1..=12).
    pub fn target_params_for(&self, year_idx: usize, month: u8) -> Option<GammaParams> {
        self.target_params[year_idx][(month - 1) as usize]
    }

    /// Returns the 1-indexed months that were successfully transformed.
    pub fn perturbed_months(&self) -> &[u8] {
        &self.perturbed_months
    }

    /// Returns the 1-indexed months that were skipped.
    pub fn skipped_months(&self) -> &[u8] {
        &self.skipped_months
    }
}
