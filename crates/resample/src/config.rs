//! Configuration for daily resampling.

use crate::error::ResampleError;
use zeus_knn::Sampling;

/// Configuration for the daily disaggregation resampler.
///
/// Use the builder methods to customise parameters.
///
/// # Example
///
/// ```
/// use zeus_resample::ResampleConfig;
///
/// let config = ResampleConfig::new()
///     .with_annual_knn_n(200)
///     .with_year_start_month(10);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ResampleConfig {
    annual_knn_n: usize,
    precip_weight: f64,
    temp_weight: f64,
    sd_floor: f64,
    narrow_window: u16,
    wide_window: u16,
    year_start_month: u8,
    sampling: Sampling,
    epsilon: f64,
}

impl ResampleConfig {
    /// Creates a new configuration with defaults.
    ///
    /// Defaults: `annual_knn_n = 100`, `precip_weight = 100.0`,
    /// `temp_weight = 10.0`, `sd_floor = 0.1`, `narrow_window = 3`,
    /// `wide_window = 30`, `year_start_month = 1`, `sampling = Rank`,
    /// `epsilon = 1e-8`.
    pub fn new() -> Self {
        Self {
            annual_knn_n: 100,
            precip_weight: 100.0,
            temp_weight: 10.0,
            sd_floor: 0.1,
            narrow_window: 3,
            wide_window: 30,
            year_start_month: 1,
            sampling: Sampling::Rank,
            epsilon: 1e-8,
        }
    }

    /// Sets the number of annual KNN samples.
    pub fn with_annual_knn_n(mut self, n: usize) -> Self {
        self.annual_knn_n = n;
        self
    }

    /// Sets the precipitation feature weight for daily KNN.
    pub fn with_precip_weight(mut self, w: f64) -> Self {
        self.precip_weight = w;
        self
    }

    /// Sets the temperature feature weight for daily KNN.
    pub fn with_temp_weight(mut self, w: f64) -> Self {
        self.temp_weight = w;
        self
    }

    /// Sets the SD floor for weight computation.
    pub fn with_sd_floor(mut self, f: f64) -> Self {
        self.sd_floor = f;
        self
    }

    /// Sets the narrow temporal window (plus/minus days).
    pub fn with_narrow_window(mut self, w: u16) -> Self {
        self.narrow_window = w;
        self
    }

    /// Sets the wide temporal window (plus/minus days).
    pub fn with_wide_window(mut self, w: u16) -> Self {
        self.wide_window = w;
        self
    }

    /// Sets the year start month (1 = calendar year, 10 = water year).
    pub fn with_year_start_month(mut self, m: u8) -> Self {
        self.year_start_month = m;
        self
    }

    /// Sets the KNN sampling strategy.
    pub fn with_sampling(mut self, s: Sampling) -> Self {
        self.sampling = s;
        self
    }

    /// Sets the epsilon floor for Gaussian KNN mode.
    pub fn with_epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
        self
    }

    // --- Accessors ---

    /// Returns the number of annual KNN samples.
    pub fn annual_knn_n(&self) -> usize {
        self.annual_knn_n
    }

    /// Returns the precipitation feature weight.
    pub fn precip_weight(&self) -> f64 {
        self.precip_weight
    }

    /// Returns the temperature feature weight.
    pub fn temp_weight(&self) -> f64 {
        self.temp_weight
    }

    /// Returns the SD floor.
    pub fn sd_floor(&self) -> f64 {
        self.sd_floor
    }

    /// Returns the narrow window size.
    pub fn narrow_window(&self) -> u16 {
        self.narrow_window
    }

    /// Returns the wide window size.
    pub fn wide_window(&self) -> u16 {
        self.wide_window
    }

    /// Returns the year start month.
    pub fn year_start_month(&self) -> u8 {
        self.year_start_month
    }

    /// Returns the KNN sampling strategy.
    pub fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    /// Returns the epsilon floor.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Validates this configuration.
    ///
    /// Checks that all values are within valid ranges.
    pub fn validate(&self) -> Result<(), ResampleError> {
        if self.annual_knn_n == 0 {
            return Err(ResampleError::InvalidConfig {
                reason: "annual_knn_n must be >= 1".to_string(),
            });
        }
        if !self.precip_weight.is_finite() || self.precip_weight <= 0.0 {
            return Err(ResampleError::InvalidConfig {
                reason: format!(
                    "precip_weight must be finite and positive, got {}",
                    self.precip_weight
                ),
            });
        }
        if !self.temp_weight.is_finite() || self.temp_weight <= 0.0 {
            return Err(ResampleError::InvalidConfig {
                reason: format!(
                    "temp_weight must be finite and positive, got {}",
                    self.temp_weight
                ),
            });
        }
        if !self.sd_floor.is_finite() || self.sd_floor <= 0.0 {
            return Err(ResampleError::InvalidConfig {
                reason: format!(
                    "sd_floor must be finite and positive, got {}",
                    self.sd_floor
                ),
            });
        }
        if self.narrow_window == 0 {
            return Err(ResampleError::InvalidConfig {
                reason: "narrow_window must be >= 1".to_string(),
            });
        }
        if self.wide_window == 0 {
            return Err(ResampleError::InvalidConfig {
                reason: "wide_window must be >= 1".to_string(),
            });
        }
        if self.wide_window < self.narrow_window {
            return Err(ResampleError::InvalidConfig {
                reason: format!(
                    "wide_window ({}) must be >= narrow_window ({})",
                    self.wide_window, self.narrow_window
                ),
            });
        }
        if !(1..=12).contains(&self.year_start_month) {
            return Err(ResampleError::InvalidConfig {
                reason: format!(
                    "year_start_month must be 1..=12, got {}",
                    self.year_start_month
                ),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(ResampleError::InvalidConfig {
                reason: format!("epsilon must be finite and positive, got {}", self.epsilon),
            });
        }
        Ok(())
    }
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let cfg = ResampleConfig::new();
        assert_eq!(cfg.annual_knn_n(), 100);
        assert!((cfg.precip_weight() - 100.0).abs() < f64::EPSILON);
        assert!((cfg.temp_weight() - 10.0).abs() < f64::EPSILON);
        assert!((cfg.sd_floor() - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.narrow_window(), 3);
        assert_eq!(cfg.wide_window(), 30);
        assert_eq!(cfg.year_start_month(), 1);
        assert_eq!(cfg.sampling(), &Sampling::Rank);
        assert!((cfg.epsilon() - 1e-8).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_chaining() {
        let cfg = ResampleConfig::new()
            .with_annual_knn_n(200)
            .with_precip_weight(50.0)
            .with_temp_weight(5.0)
            .with_sd_floor(0.05)
            .with_narrow_window(5)
            .with_wide_window(60)
            .with_year_start_month(10)
            .with_sampling(Sampling::Uniform)
            .with_epsilon(1e-6);

        assert_eq!(cfg.annual_knn_n(), 200);
        assert!((cfg.precip_weight() - 50.0).abs() < f64::EPSILON);
        assert!((cfg.temp_weight() - 5.0).abs() < f64::EPSILON);
        assert!((cfg.sd_floor() - 0.05).abs() < f64::EPSILON);
        assert_eq!(cfg.narrow_window(), 5);
        assert_eq!(cfg.wide_window(), 60);
        assert_eq!(cfg.year_start_month(), 10);
        assert_eq!(cfg.sampling(), &Sampling::Uniform);
        assert!((cfg.epsilon() - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn validate_ok() {
        assert!(ResampleConfig::new().validate().is_ok());
    }

    #[test]
    fn validate_bad_annual_knn_n() {
        assert!(
            ResampleConfig::new()
                .with_annual_knn_n(0)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_precip_weight() {
        assert!(
            ResampleConfig::new()
                .with_precip_weight(0.0)
                .validate()
                .is_err()
        );
        assert!(
            ResampleConfig::new()
                .with_precip_weight(-1.0)
                .validate()
                .is_err()
        );
        assert!(
            ResampleConfig::new()
                .with_precip_weight(f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_temp_weight() {
        assert!(
            ResampleConfig::new()
                .with_temp_weight(0.0)
                .validate()
                .is_err()
        );
        assert!(
            ResampleConfig::new()
                .with_temp_weight(f64::INFINITY)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_sd_floor() {
        assert!(ResampleConfig::new().with_sd_floor(0.0).validate().is_err());
        assert!(
            ResampleConfig::new()
                .with_sd_floor(-0.1)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_narrow_window() {
        assert!(
            ResampleConfig::new()
                .with_narrow_window(0)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_wide_window() {
        assert!(
            ResampleConfig::new()
                .with_wide_window(0)
                .validate()
                .is_err()
        );
        // wide < narrow
        assert!(
            ResampleConfig::new()
                .with_narrow_window(10)
                .with_wide_window(5)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_year_start_month() {
        assert!(
            ResampleConfig::new()
                .with_year_start_month(0)
                .validate()
                .is_err()
        );
        assert!(
            ResampleConfig::new()
                .with_year_start_month(13)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_bad_epsilon() {
        assert!(ResampleConfig::new().with_epsilon(0.0).validate().is_err());
        assert!(
            ResampleConfig::new()
                .with_epsilon(f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn default_matches_new() {
        let d = ResampleConfig::default();
        let n = ResampleConfig::new();
        assert_eq!(d.annual_knn_n(), n.annual_knn_n());
        assert!((d.precip_weight() - n.precip_weight()).abs() < f64::EPSILON);
    }
}
