//! Configuration for Markov chain estimation and simulation.

use crate::error::MarkovError;

/// Specifies how a threshold is determined.
#[derive(Debug, Clone, Copy)]
pub enum ThresholdSpec {
    /// A fixed precipitation value (e.g. 0.3 mm).
    Fixed(f64),
    /// A quantile resolved per-month from the baseline data (e.g. 0.8).
    Quantile(f64),
}

/// Configuration for Markov chain estimation.
///
/// Use the builder methods to customise parameters.
///
/// # Example
///
/// ```
/// use zeus_markov::MarkovConfig;
///
/// let config = MarkovConfig::new()
///     .with_dirichlet_alpha(0.5);
/// ```
#[derive(Clone, Debug)]
pub struct MarkovConfig {
    wet_spec: ThresholdSpec,
    extreme_spec: ThresholdSpec,
    dirichlet_alpha: f64,
    dry_spell_factors: [f64; 12],
    wet_spell_factors: [f64; 12],
}

impl MarkovConfig {
    /// Creates a new configuration with defaults.
    ///
    /// Defaults: `wet_spec = Fixed(0.3)`, `extreme_spec = Quantile(0.8)`,
    /// `dirichlet_alpha = 1.0`, all spell factors = 1.0.
    pub fn new() -> Self {
        Self {
            wet_spec: ThresholdSpec::Fixed(0.3),
            extreme_spec: ThresholdSpec::Quantile(0.8),
            dirichlet_alpha: 1.0,
            dry_spell_factors: [1.0; 12],
            wet_spell_factors: [1.0; 12],
        }
    }

    /// Sets the wet/dry threshold specification.
    pub fn with_wet_spec(mut self, spec: ThresholdSpec) -> Self {
        self.wet_spec = spec;
        self
    }

    /// Sets the wet/extreme threshold specification.
    pub fn with_extreme_spec(mut self, spec: ThresholdSpec) -> Self {
        self.extreme_spec = spec;
        self
    }

    /// Sets the Dirichlet smoothing parameter.
    pub fn with_dirichlet_alpha(mut self, alpha: f64) -> Self {
        self.dirichlet_alpha = alpha;
        self
    }

    /// Sets the monthly dry-spell persistence factors.
    pub fn with_dry_spell_factors(mut self, factors: [f64; 12]) -> Self {
        self.dry_spell_factors = factors;
        self
    }

    /// Sets the monthly wet-spell persistence factors.
    pub fn with_wet_spell_factors(mut self, factors: [f64; 12]) -> Self {
        self.wet_spell_factors = factors;
        self
    }

    // --- Accessors ---

    /// Returns the wet/dry threshold specification.
    pub fn wet_spec(&self) -> ThresholdSpec {
        self.wet_spec
    }

    /// Returns the wet/extreme threshold specification.
    pub fn extreme_spec(&self) -> ThresholdSpec {
        self.extreme_spec
    }

    /// Returns the Dirichlet smoothing parameter.
    pub fn dirichlet_alpha(&self) -> f64 {
        self.dirichlet_alpha
    }

    /// Returns the monthly dry-spell persistence factors.
    pub fn dry_spell_factors(&self) -> &[f64; 12] {
        &self.dry_spell_factors
    }

    /// Returns the monthly wet-spell persistence factors.
    pub fn wet_spell_factors(&self) -> &[f64; 12] {
        &self.wet_spell_factors
    }

    /// Validates this configuration.
    ///
    /// Checks that alpha is finite and positive, spell factors are finite and
    /// positive, `Fixed` values are finite and non-negative, and `Quantile`
    /// values are in the open interval (0, 1).
    pub fn validate(&self) -> Result<(), MarkovError> {
        // Alpha
        if !self.dirichlet_alpha.is_finite() || self.dirichlet_alpha <= 0.0 {
            return Err(MarkovError::InvalidThreshold {
                reason: format!(
                    "dirichlet_alpha must be finite and positive, got {}",
                    self.dirichlet_alpha
                ),
            });
        }

        // Threshold specs
        Self::validate_spec(self.wet_spec, "wet_spec")?;
        Self::validate_spec(self.extreme_spec, "extreme_spec")?;

        // Spell factors
        for (i, &f) in self.dry_spell_factors.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(MarkovError::InvalidSpellFactor {
                    month: (i + 1) as u8,
                    value: f,
                });
            }
        }
        for (i, &f) in self.wet_spell_factors.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(MarkovError::InvalidSpellFactor {
                    month: (i + 1) as u8,
                    value: f,
                });
            }
        }

        Ok(())
    }

    fn validate_spec(spec: ThresholdSpec, name: &str) -> Result<(), MarkovError> {
        match spec {
            ThresholdSpec::Fixed(v) => {
                if !v.is_finite() || v < 0.0 {
                    return Err(MarkovError::InvalidThreshold {
                        reason: format!(
                            "{name}: Fixed value must be finite and non-negative, got {v}"
                        ),
                    });
                }
            }
            ThresholdSpec::Quantile(q) => {
                if !q.is_finite() || q <= 0.0 || q >= 1.0 {
                    return Err(MarkovError::InvalidThreshold {
                        reason: format!("{name}: Quantile must be in (0, 1), got {q}"),
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for MarkovConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let cfg = MarkovConfig::new();
        assert!(
            matches!(cfg.wet_spec(), ThresholdSpec::Fixed(v) if (v - 0.3).abs() < f64::EPSILON)
        );
        assert!(
            matches!(cfg.extreme_spec(), ThresholdSpec::Quantile(q) if (q - 0.8).abs() < f64::EPSILON)
        );
        assert!((cfg.dirichlet_alpha() - 1.0).abs() < f64::EPSILON);
        assert!(
            cfg.dry_spell_factors()
                .iter()
                .all(|&f| (f - 1.0).abs() < f64::EPSILON)
        );
        assert!(
            cfg.wet_spell_factors()
                .iter()
                .all(|&f| (f - 1.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn builder_chaining() {
        let dry_factors = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2];
        let wet_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 1.1, 1.2];

        let cfg = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.5))
            .with_extreme_spec(ThresholdSpec::Quantile(0.95))
            .with_dirichlet_alpha(0.5)
            .with_dry_spell_factors(dry_factors)
            .with_wet_spell_factors(wet_factors);

        assert!(
            matches!(cfg.wet_spec(), ThresholdSpec::Fixed(v) if (v - 0.5).abs() < f64::EPSILON)
        );
        assert!(
            matches!(cfg.extreme_spec(), ThresholdSpec::Quantile(q) if (q - 0.95).abs() < f64::EPSILON)
        );
        assert!((cfg.dirichlet_alpha() - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.dry_spell_factors(), &dry_factors);
        assert_eq!(cfg.wet_spell_factors(), &wet_factors);
    }

    #[test]
    fn validate_ok() {
        assert!(MarkovConfig::new().validate().is_ok());
    }

    #[test]
    fn validate_bad_alpha() {
        // Zero
        assert!(
            MarkovConfig::new()
                .with_dirichlet_alpha(0.0)
                .validate()
                .is_err()
        );
        // Negative
        assert!(
            MarkovConfig::new()
                .with_dirichlet_alpha(-1.0)
                .validate()
                .is_err()
        );
        // NaN
        assert!(
            MarkovConfig::new()
                .with_dirichlet_alpha(f64::NAN)
                .validate()
                .is_err()
        );
        // Infinity
        assert!(
            MarkovConfig::new()
                .with_dirichlet_alpha(f64::INFINITY)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_fixed() {
        // Negative
        assert!(
            MarkovConfig::new()
                .with_wet_spec(ThresholdSpec::Fixed(-0.1))
                .validate()
                .is_err()
        );
        // NaN
        assert!(
            MarkovConfig::new()
                .with_wet_spec(ThresholdSpec::Fixed(f64::NAN))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_quantile() {
        // 0.0 (boundary, exclusive)
        assert!(
            MarkovConfig::new()
                .with_extreme_spec(ThresholdSpec::Quantile(0.0))
                .validate()
                .is_err()
        );
        // 1.0 (boundary, exclusive)
        assert!(
            MarkovConfig::new()
                .with_extreme_spec(ThresholdSpec::Quantile(1.0))
                .validate()
                .is_err()
        );
        // Negative
        assert!(
            MarkovConfig::new()
                .with_extreme_spec(ThresholdSpec::Quantile(-0.5))
                .validate()
                .is_err()
        );
        // NaN
        assert!(
            MarkovConfig::new()
                .with_extreme_spec(ThresholdSpec::Quantile(f64::NAN))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_spell_factor() {
        // Zero in dry
        let mut factors = [1.0; 12];
        factors[5] = 0.0;
        assert!(
            MarkovConfig::new()
                .with_dry_spell_factors(factors)
                .validate()
                .is_err()
        );

        // Negative in dry
        factors[5] = -1.0;
        assert!(
            MarkovConfig::new()
                .with_dry_spell_factors(factors)
                .validate()
                .is_err()
        );

        // NaN in dry
        factors[5] = f64::NAN;
        assert!(
            MarkovConfig::new()
                .with_dry_spell_factors(factors)
                .validate()
                .is_err()
        );

        // Infinity in dry
        factors[5] = f64::INFINITY;
        assert!(
            MarkovConfig::new()
                .with_dry_spell_factors(factors)
                .validate()
                .is_err()
        );

        // Zero in wet
        let mut wet_factors = [1.0; 12];
        wet_factors[0] = 0.0;
        assert!(
            MarkovConfig::new()
                .with_wet_spell_factors(wet_factors)
                .validate()
                .is_err()
        );

        // Negative in wet
        wet_factors[0] = -2.0;
        assert!(
            MarkovConfig::new()
                .with_wet_spell_factors(wet_factors)
                .validate()
                .is_err()
        );

        // NaN in wet
        wet_factors[0] = f64::NAN;
        assert!(
            MarkovConfig::new()
                .with_wet_spell_factors(wet_factors)
                .validate()
                .is_err()
        );

        // Infinity in wet
        wet_factors[0] = f64::INFINITY;
        assert!(
            MarkovConfig::new()
                .with_wet_spell_factors(wet_factors)
                .validate()
                .is_err()
        );
    }
}
