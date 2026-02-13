//! Configuration for quantile mapping.

use crate::error::QuantileMapError;

/// Method used to fit gamma distribution parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FitMethod {
    /// Method of moments estimation.
    #[default]
    Mme,
}

/// Configuration for quantile-mapping estimation and transformation.
///
/// Use the builder methods to customise parameters.
///
/// # Example
///
/// ```
/// use zeus_quantile_map::QmConfig;
///
/// let config = QmConfig::new()
///     .with_intensity_threshold(0.3)
///     .with_min_events(20);
/// ```
#[derive(Clone, Debug)]
pub struct QmConfig {
    intensity_threshold: f64,
    fit_method: FitMethod,
    min_events: usize,
    scale_var_with_mean: bool,
    exaggerate_extremes: bool,
    extreme_prob_threshold: f64,
    extreme_k: f64,
    enforce_target_mean: bool,
    min_group_for_enforcement: usize,
    validate_output: bool,
}

impl QmConfig {
    /// Creates a new configuration with defaults.
    ///
    /// Defaults: `intensity_threshold = 0.0`, `fit_method = Mme`,
    /// `min_events = 10`, `scale_var_with_mean = true`,
    /// `exaggerate_extremes = false`, `extreme_prob_threshold = 0.95`,
    /// `extreme_k = 1.2`, `enforce_target_mean = true`,
    /// `min_group_for_enforcement = 5`, `validate_output = true`.
    pub fn new() -> Self {
        Self {
            intensity_threshold: 0.0,
            fit_method: FitMethod::Mme,
            min_events: 10,
            scale_var_with_mean: true,
            exaggerate_extremes: false,
            extreme_prob_threshold: 0.95,
            extreme_k: 1.2,
            enforce_target_mean: true,
            min_group_for_enforcement: 5,
            validate_output: true,
        }
    }

    // --- Builder methods ---

    /// Sets the precipitation intensity threshold.
    pub fn with_intensity_threshold(mut self, v: f64) -> Self {
        self.intensity_threshold = v;
        self
    }

    /// Sets the distribution fitting method.
    pub fn with_fit_method(mut self, m: FitMethod) -> Self {
        self.fit_method = m;
        self
    }

    /// Sets the minimum number of events required for fitting.
    pub fn with_min_events(mut self, n: usize) -> Self {
        self.min_events = n;
        self
    }

    /// Sets whether to scale variance with mean.
    pub fn with_scale_var_with_mean(mut self, b: bool) -> Self {
        self.scale_var_with_mean = b;
        self
    }

    /// Sets whether to exaggerate extremes in the mapping.
    pub fn with_exaggerate_extremes(mut self, b: bool) -> Self {
        self.exaggerate_extremes = b;
        self
    }

    /// Sets the probability threshold for identifying extremes.
    pub fn with_extreme_prob_threshold(mut self, p: f64) -> Self {
        self.extreme_prob_threshold = p;
        self
    }

    /// Sets the extreme exaggeration factor.
    pub fn with_extreme_k(mut self, k: f64) -> Self {
        self.extreme_k = k;
        self
    }

    /// Sets whether to enforce the target mean after mapping.
    pub fn with_enforce_target_mean(mut self, b: bool) -> Self {
        self.enforce_target_mean = b;
        self
    }

    /// Sets the minimum group size for mean enforcement.
    pub fn with_min_group_for_enforcement(mut self, n: usize) -> Self {
        self.min_group_for_enforcement = n;
        self
    }

    /// Sets whether to validate output after mapping.
    pub fn with_validate_output(mut self, b: bool) -> Self {
        self.validate_output = b;
        self
    }

    // --- Accessors ---

    /// Returns the precipitation intensity threshold.
    pub fn intensity_threshold(&self) -> f64 {
        self.intensity_threshold
    }

    /// Returns the distribution fitting method.
    pub fn fit_method(&self) -> FitMethod {
        self.fit_method
    }

    /// Returns the minimum number of events required for fitting.
    pub fn min_events(&self) -> usize {
        self.min_events
    }

    /// Returns whether to scale variance with mean.
    pub fn scale_var_with_mean(&self) -> bool {
        self.scale_var_with_mean
    }

    /// Returns whether to exaggerate extremes in the mapping.
    pub fn exaggerate_extremes(&self) -> bool {
        self.exaggerate_extremes
    }

    /// Returns the probability threshold for identifying extremes.
    pub fn extreme_prob_threshold(&self) -> f64 {
        self.extreme_prob_threshold
    }

    /// Returns the extreme exaggeration factor.
    pub fn extreme_k(&self) -> f64 {
        self.extreme_k
    }

    /// Returns whether to enforce the target mean after mapping.
    pub fn enforce_target_mean(&self) -> bool {
        self.enforce_target_mean
    }

    /// Returns the minimum group size for mean enforcement.
    pub fn min_group_for_enforcement(&self) -> usize {
        self.min_group_for_enforcement
    }

    /// Returns whether to validate output after mapping.
    pub fn validate_output(&self) -> bool {
        self.validate_output
    }

    /// Validates this configuration.
    ///
    /// Checks that `intensity_threshold` is finite and non-negative,
    /// `min_events` is at least 1, `extreme_prob_threshold` is in the open
    /// interval (0, 1) and finite, `extreme_k` is finite and positive, and
    /// `min_group_for_enforcement` is at least 1.
    pub fn validate(&self) -> Result<(), QuantileMapError> {
        if !self.intensity_threshold.is_finite() || self.intensity_threshold < 0.0 {
            return Err(QuantileMapError::InvalidConfig {
                reason: format!(
                    "intensity_threshold must be finite and >= 0, got {}",
                    self.intensity_threshold
                ),
            });
        }

        if self.min_events < 1 {
            return Err(QuantileMapError::InvalidConfig {
                reason: format!("min_events must be >= 1, got {}", self.min_events),
            });
        }

        if !self.extreme_prob_threshold.is_finite()
            || self.extreme_prob_threshold <= 0.0
            || self.extreme_prob_threshold >= 1.0
        {
            return Err(QuantileMapError::InvalidConfig {
                reason: format!(
                    "extreme_prob_threshold must be in (0, 1) and finite, got {}",
                    self.extreme_prob_threshold
                ),
            });
        }

        if !self.extreme_k.is_finite() || self.extreme_k <= 0.0 {
            return Err(QuantileMapError::InvalidConfig {
                reason: format!("extreme_k must be finite and > 0, got {}", self.extreme_k),
            });
        }

        if self.min_group_for_enforcement < 1 {
            return Err(QuantileMapError::InvalidConfig {
                reason: format!(
                    "min_group_for_enforcement must be >= 1, got {}",
                    self.min_group_for_enforcement
                ),
            });
        }

        Ok(())
    }
}

impl Default for QmConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let cfg = QmConfig::new();
        assert!((cfg.intensity_threshold() - 0.0).abs() < f64::EPSILON);
        assert_eq!(cfg.fit_method(), FitMethod::Mme);
        assert_eq!(cfg.min_events(), 10);
        assert!(cfg.scale_var_with_mean());
        assert!(!cfg.exaggerate_extremes());
        assert!((cfg.extreme_prob_threshold() - 0.95).abs() < f64::EPSILON);
        assert!((cfg.extreme_k() - 1.2).abs() < f64::EPSILON);
        assert!(cfg.enforce_target_mean());
        assert_eq!(cfg.min_group_for_enforcement(), 5);
        assert!(cfg.validate_output());
    }

    #[test]
    fn builder_chaining() {
        let cfg = QmConfig::new()
            .with_intensity_threshold(0.5)
            .with_fit_method(FitMethod::Mme)
            .with_min_events(20)
            .with_scale_var_with_mean(false)
            .with_exaggerate_extremes(true)
            .with_extreme_prob_threshold(0.99)
            .with_extreme_k(1.5);

        assert!((cfg.intensity_threshold() - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.fit_method(), FitMethod::Mme);
        assert_eq!(cfg.min_events(), 20);
        assert!(!cfg.scale_var_with_mean());
        assert!(cfg.exaggerate_extremes());
        assert!((cfg.extreme_prob_threshold() - 0.99).abs() < f64::EPSILON);
        assert!((cfg.extreme_k() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn validate_ok() {
        assert!(QmConfig::new().validate().is_ok());
    }

    #[test]
    fn validate_negative_threshold() {
        assert!(
            QmConfig::new()
                .with_intensity_threshold(-0.1)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_nan_threshold() {
        assert!(
            QmConfig::new()
                .with_intensity_threshold(f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_prob() {
        // At 0.0 (boundary, exclusive)
        assert!(
            QmConfig::new()
                .with_extreme_prob_threshold(0.0)
                .validate()
                .is_err()
        );
        // At 1.0 (boundary, exclusive)
        assert!(
            QmConfig::new()
                .with_extreme_prob_threshold(1.0)
                .validate()
                .is_err()
        );
        // NaN
        assert!(
            QmConfig::new()
                .with_extreme_prob_threshold(f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_k() {
        // At 0.0
        assert!(QmConfig::new().with_extreme_k(0.0).validate().is_err());
        // Negative
        assert!(QmConfig::new().with_extreme_k(-1.0).validate().is_err());
        // NaN
        assert!(QmConfig::new().with_extreme_k(f64::NAN).validate().is_err());
    }

    #[test]
    fn validate_zero_min_events() {
        assert!(QmConfig::new().with_min_events(0).validate().is_err());
    }

    #[test]
    fn default_trait() {
        let from_new = QmConfig::new();
        let from_default = QmConfig::default();
        assert!(
            (from_new.intensity_threshold() - from_default.intensity_threshold()).abs()
                < f64::EPSILON
        );
    }
}
