//! Configuration types for the perturbation pipeline.

use crate::error::PerturbError;
use zeus_quantile_map::{QmConfig, ScenarioFactors};

/// Temperature perturbation configuration.
#[derive(Debug, Clone)]
pub struct TempConfig {
    /// Monthly additive deltas in degrees.
    deltas: [f64; 12],
    /// Whether to ramp from 0 to 2*delta over the simulation period.
    transient: bool,
}

impl TempConfig {
    /// Creates a new temperature configuration.
    ///
    /// # Arguments
    ///
    /// * `deltas` — Monthly additive temperature deltas (degrees).
    /// * `transient` — If `true`, linearly ramp from zero to twice the delta.
    pub fn new(deltas: [f64; 12], transient: bool) -> Self {
        Self { deltas, transient }
    }

    /// Returns the monthly deltas.
    pub fn deltas(&self) -> &[f64; 12] {
        &self.deltas
    }

    /// Returns whether transient ramping is enabled.
    pub fn transient(&self) -> bool {
        self.transient
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), PerturbError> {
        for (i, &d) in self.deltas.iter().enumerate() {
            if !d.is_finite() {
                return Err(PerturbError::InvalidConfig {
                    reason: format!("temp delta at month index {i} is not finite: {d}"),
                });
            }
        }
        Ok(())
    }
}

/// Precipitation occurrence adjustment configuration.
#[derive(Debug, Clone)]
pub struct OccurrenceConfig {
    /// Monthly occurrence multipliers (>0).
    factors: [f64; 12],
    /// Whether to ramp from 1.0 to 2*factor - 1 over the simulation period.
    transient: bool,
    /// Wet-day threshold (should match QM threshold).
    intensity_threshold: f64,
}

impl OccurrenceConfig {
    /// Creates a new occurrence configuration.
    ///
    /// # Arguments
    ///
    /// * `factors` — Monthly occurrence multipliers (must be > 0).
    /// * `transient` — If `true`, linearly ramp factors over years.
    /// * `intensity_threshold` — Wet-day threshold (mm/day).
    pub fn new(factors: [f64; 12], transient: bool, intensity_threshold: f64) -> Self {
        Self {
            factors,
            transient,
            intensity_threshold,
        }
    }

    /// Returns the monthly occurrence factors.
    pub fn factors(&self) -> &[f64; 12] {
        &self.factors
    }

    /// Returns whether transient ramping is enabled.
    pub fn transient(&self) -> bool {
        self.transient
    }

    /// Returns the wet-day intensity threshold.
    pub fn intensity_threshold(&self) -> f64 {
        self.intensity_threshold
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), PerturbError> {
        for (i, &f) in self.factors.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(PerturbError::InvalidConfig {
                    reason: format!(
                        "occurrence factor at month index {i} must be finite and > 0, got {f}"
                    ),
                });
            }
        }
        if !self.intensity_threshold.is_finite() || self.intensity_threshold < 0.0 {
            return Err(PerturbError::InvalidConfig {
                reason: format!(
                    "intensity_threshold must be finite and >= 0, got {}",
                    self.intensity_threshold
                ),
            });
        }
        Ok(())
    }
}

/// Master configuration for the perturbation pipeline.
#[derive(Debug, Clone)]
pub struct PerturbConfig {
    /// Temperature perturbation (None = skip).
    temp: Option<TempConfig>,
    /// Quantile mapping configuration (None = skip QM step).
    qm_config: Option<QmConfig>,
    /// Monthly mean scaling factors for QM (None = skip QM step).
    mean_factors: Option<ScenarioFactors>,
    /// Monthly variance scaling factors for QM (None = skip QM step).
    var_factors: Option<ScenarioFactors>,
    /// Occurrence adjustment (None = skip).
    occurrence: Option<OccurrenceConfig>,
    /// Floor for precipitation values after processing.
    precip_floor: f64,
    /// Cap for precipitation values after processing.
    precip_cap: f64,
    /// Optional RNG seed for reproducibility.
    seed: Option<u64>,
}

impl PerturbConfig {
    /// Creates a new configuration with all modules disabled.
    pub fn new() -> Self {
        Self {
            temp: None,
            qm_config: None,
            mean_factors: None,
            var_factors: None,
            occurrence: None,
            precip_floor: 0.0,
            precip_cap: 500.0,
            seed: None,
        }
    }

    /// Sets the temperature configuration.
    pub fn with_temp(mut self, config: TempConfig) -> Self {
        self.temp = Some(config);
        self
    }

    /// Sets the quantile mapping configuration and factors.
    pub fn with_qm(
        mut self,
        config: QmConfig,
        mean_factors: ScenarioFactors,
        var_factors: ScenarioFactors,
    ) -> Self {
        self.qm_config = Some(config);
        self.mean_factors = Some(mean_factors);
        self.var_factors = Some(var_factors);
        self
    }

    /// Sets the occurrence adjustment configuration.
    pub fn with_occurrence(mut self, config: OccurrenceConfig) -> Self {
        self.occurrence = Some(config);
        self
    }

    /// Sets the precipitation floor.
    pub fn with_precip_floor(mut self, floor: f64) -> Self {
        self.precip_floor = floor;
        self
    }

    /// Sets the precipitation cap.
    pub fn with_precip_cap(mut self, cap: f64) -> Self {
        self.precip_cap = cap;
        self
    }

    /// Sets the RNG seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // --- Accessors ---

    /// Returns the temperature configuration, if set.
    pub fn temp(&self) -> Option<&TempConfig> {
        self.temp.as_ref()
    }

    /// Returns the QM configuration, if set.
    pub fn qm_config(&self) -> Option<&QmConfig> {
        self.qm_config.as_ref()
    }

    /// Returns the mean scaling factors, if set.
    pub fn mean_factors(&self) -> Option<&ScenarioFactors> {
        self.mean_factors.as_ref()
    }

    /// Returns the variance scaling factors, if set.
    pub fn var_factors(&self) -> Option<&ScenarioFactors> {
        self.var_factors.as_ref()
    }

    /// Returns the occurrence configuration, if set.
    pub fn occurrence(&self) -> Option<&OccurrenceConfig> {
        self.occurrence.as_ref()
    }

    /// Returns the precipitation floor.
    pub fn precip_floor(&self) -> f64 {
        self.precip_floor
    }

    /// Returns the precipitation cap.
    pub fn precip_cap(&self) -> f64 {
        self.precip_cap
    }

    /// Returns the RNG seed, if set.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Validates the entire configuration.
    pub fn validate(&self) -> Result<(), PerturbError> {
        if let Some(ref t) = self.temp {
            t.validate()?;
        }
        if let Some(ref o) = self.occurrence {
            o.validate()?;
        }
        if !self.precip_floor.is_finite() {
            return Err(PerturbError::InvalidConfig {
                reason: format!("precip_floor must be finite, got {}", self.precip_floor),
            });
        }
        if !self.precip_cap.is_finite() {
            return Err(PerturbError::InvalidConfig {
                reason: format!("precip_cap must be finite, got {}", self.precip_cap),
            });
        }
        if self.precip_floor > self.precip_cap {
            return Err(PerturbError::InvalidConfig {
                reason: format!(
                    "precip_floor ({}) must be <= precip_cap ({})",
                    self.precip_floor, self.precip_cap
                ),
            });
        }
        Ok(())
    }
}

impl Default for PerturbConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        assert!(PerturbConfig::new().validate().is_ok());
    }

    #[test]
    fn temp_config_validates_ok() {
        let tc = TempConfig::new([1.0; 12], false);
        assert!(tc.validate().is_ok());
    }

    #[test]
    fn temp_config_nan_fails() {
        let mut deltas = [1.0; 12];
        deltas[3] = f64::NAN;
        let tc = TempConfig::new(deltas, false);
        assert!(tc.validate().is_err());
    }

    #[test]
    fn occurrence_config_validates_ok() {
        let oc = OccurrenceConfig::new([1.0; 12], false, 0.0);
        assert!(oc.validate().is_ok());
    }

    #[test]
    fn occurrence_zero_factor_fails() {
        let mut factors = [1.0; 12];
        factors[5] = 0.0;
        let oc = OccurrenceConfig::new(factors, false, 0.0);
        assert!(oc.validate().is_err());
    }

    #[test]
    fn occurrence_negative_factor_fails() {
        let mut factors = [1.0; 12];
        factors[0] = -0.5;
        let oc = OccurrenceConfig::new(factors, false, 0.0);
        assert!(oc.validate().is_err());
    }

    #[test]
    fn occurrence_negative_threshold_fails() {
        let oc = OccurrenceConfig::new([1.0; 12], false, -1.0);
        assert!(oc.validate().is_err());
    }

    #[test]
    fn precip_floor_gt_cap_fails() {
        let config = PerturbConfig::new()
            .with_precip_floor(100.0)
            .with_precip_cap(50.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn precip_floor_nan_fails() {
        let config = PerturbConfig::new().with_precip_floor(f64::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn builder_with_temp() {
        let config = PerturbConfig::new().with_temp(TempConfig::new([2.0; 12], true));
        assert!(config.temp().is_some());
        assert!(config.temp().unwrap().transient());
    }

    #[test]
    fn builder_with_qm() {
        let qm = QmConfig::new();
        let factors = ScenarioFactors::uniform(1, [1.0; 12]);
        let config = PerturbConfig::new().with_qm(qm, factors.clone(), factors);
        assert!(config.qm_config().is_some());
        assert!(config.mean_factors().is_some());
        assert!(config.var_factors().is_some());
    }

    #[test]
    fn builder_with_seed() {
        let config = PerturbConfig::new().with_seed(42);
        assert_eq!(config.seed(), Some(42));
    }

    #[test]
    fn default_precip_bounds() {
        let config = PerturbConfig::new();
        assert!((config.precip_floor() - 0.0).abs() < f64::EPSILON);
        assert!((config.precip_cap() - 500.0).abs() < f64::EPSILON);
    }
}
