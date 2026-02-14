//! Evaluation configuration.

/// Configuration for the evaluation pipeline.
#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    precip_threshold: f64,
    precip_var: String,
    temp_max_var: String,
    temp_min_var: String,
}

impl Default for EvaluateConfig {
    fn default() -> Self {
        Self {
            precip_threshold: 0.01,
            precip_var: "precip".to_string(),
            temp_max_var: "temp_max".to_string(),
            temp_min_var: "temp_min".to_string(),
        }
    }
}

impl EvaluateConfig {
    /// Set the precipitation threshold (mm) for wet/dry classification.
    pub fn with_precip_threshold(mut self, threshold: f64) -> Self {
        self.precip_threshold = threshold;
        self
    }

    /// Set the precipitation variable name.
    pub fn with_precip_var(mut self, name: impl Into<String>) -> Self {
        self.precip_var = name.into();
        self
    }

    /// Set the max temperature variable name.
    pub fn with_temp_max_var(mut self, name: impl Into<String>) -> Self {
        self.temp_max_var = name.into();
        self
    }

    /// Set the min temperature variable name.
    pub fn with_temp_min_var(mut self, name: impl Into<String>) -> Self {
        self.temp_min_var = name.into();
        self
    }

    /// Returns the precipitation threshold.
    pub fn precip_threshold(&self) -> f64 {
        self.precip_threshold
    }

    /// Returns the precipitation variable name.
    pub fn precip_var(&self) -> &str {
        &self.precip_var
    }

    /// Returns the max temperature variable name.
    pub fn temp_max_var(&self) -> &str {
        &self.temp_max_var
    }

    /// Returns the min temperature variable name.
    pub fn temp_min_var(&self) -> &str {
        &self.temp_min_var
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = EvaluateConfig::default();
        assert_eq!(config.precip_threshold(), 0.01);
        assert_eq!(config.precip_var(), "precip");
        assert_eq!(config.temp_max_var(), "temp_max");
        assert_eq!(config.temp_min_var(), "temp_min");
    }

    #[test]
    fn test_builder_methods() {
        let config = EvaluateConfig::default()
            .with_precip_threshold(1.0)
            .with_precip_var("rainfall")
            .with_temp_max_var("tmax")
            .with_temp_min_var("tmin");

        assert_eq!(config.precip_threshold(), 1.0);
        assert_eq!(config.precip_var(), "rainfall");
        assert_eq!(config.temp_max_var(), "tmax");
        assert_eq!(config.temp_min_var(), "tmin");
    }

    #[test]
    fn test_clone() {
        let config1 = EvaluateConfig::default().with_precip_threshold(2.0);
        let config2 = config1.clone();
        assert_eq!(config2.precip_threshold(), 2.0);

        let config3 = config2.with_precip_threshold(3.0);
        assert_eq!(config1.precip_threshold(), 2.0);
        assert_eq!(config3.precip_threshold(), 3.0);
    }
}
