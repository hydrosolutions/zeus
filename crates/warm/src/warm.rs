//! WARM simulation engine.

use zeus_wavelet::MraConfig;

use crate::error::WarmError;

/// Configuration for a WARM simulation.
///
/// Use the builder methods to customize simulation parameters.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::{WaveletFilter, MraConfig};
/// use zeus_warm::WarmConfig;
///
/// let mra_config = MraConfig::new(WaveletFilter::La8);
/// let config = WarmConfig::new(mra_config, 1000, 30)
///     .with_seed(42)
///     .with_bypass_n(50);
/// ```
#[derive(Clone, Debug)]
pub struct WarmConfig {
    mra_config: MraConfig,
    n_sim: usize,
    n_years: usize,
    seed: Option<u64>,
    bypass_n: usize,
    max_arma_order: (usize, usize),
}

impl WarmConfig {
    /// Creates a new WARM configuration with defaults.
    ///
    /// Defaults: `seed = None`, `bypass_n = 30`, `max_arma_order = (5, 3)`.
    pub fn new(mra_config: MraConfig, n_sim: usize, n_years: usize) -> Self {
        Self {
            mra_config,
            n_sim,
            n_years,
            seed: None,
            bypass_n: 30,
            max_arma_order: (5, 3),
        }
    }

    /// Sets the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the number of initial simulations to bypass (burn-in).
    pub fn with_bypass_n(mut self, bypass_n: usize) -> Self {
        self.bypass_n = bypass_n;
        self
    }

    /// Sets the maximum ARMA order `(max_p, max_q)` for AIC grid search.
    pub fn with_max_arma_order(mut self, max_p: usize, max_q: usize) -> Self {
        self.max_arma_order = (max_p, max_q);
        self
    }

    /// Returns the MRA configuration.
    pub fn mra_config(&self) -> &MraConfig {
        &self.mra_config
    }

    /// Returns the number of simulations.
    pub fn n_sim(&self) -> usize {
        self.n_sim
    }

    /// Returns the number of years per simulation.
    pub fn n_years(&self) -> usize {
        self.n_years
    }

    /// Returns the random seed, if set.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Returns the burn-in count.
    pub fn bypass_n(&self) -> usize {
        self.bypass_n
    }

    /// Returns the maximum ARMA order `(max_p, max_q)`.
    pub fn max_arma_order(&self) -> (usize, usize) {
        self.max_arma_order
    }
}

/// Results of a WARM simulation.
///
/// Contains the simulated series, fitted ARMA orders per component,
/// and flags indicating bootstrap fallback usage.
#[derive(Clone, Debug)]
pub struct WarmResult {
    simulations: Vec<Vec<f64>>,
    component_orders: Vec<(usize, usize)>,
    bootstrap_fallbacks: Vec<bool>,
}

impl WarmResult {
    /// Creates a new `WarmResult` (crate-internal constructor).
    #[allow(dead_code)]
    pub(crate) fn new(
        simulations: Vec<Vec<f64>>,
        component_orders: Vec<(usize, usize)>,
        bootstrap_fallbacks: Vec<bool>,
    ) -> Self {
        Self {
            simulations,
            component_orders,
            bootstrap_fallbacks,
        }
    }

    /// Returns the simulated series.
    pub fn simulations(&self) -> &[Vec<f64>] {
        &self.simulations
    }

    /// Returns the ARMA orders `(p, q)` fitted to each MRA component.
    pub fn component_orders(&self) -> &[(usize, usize)] {
        &self.component_orders
    }

    /// Returns flags indicating which components fell back to bootstrap.
    pub fn bootstrap_fallbacks(&self) -> &[bool] {
        &self.bootstrap_fallbacks
    }

    /// Returns the number of simulations.
    pub fn n_sim(&self) -> usize {
        self.simulations.len()
    }

    /// Returns the number of years per simulation, or 0 if empty.
    pub fn n_years(&self) -> usize {
        self.simulations.first().map_or(0, |s| s.len())
    }
}

/// Runs the WARM simulation pipeline.
///
/// Decomposes the observed series via MRA, fits ARMA models to each
/// component, simulates synthetic components, and recombines them.
///
/// # Errors
///
/// | Variant | Trigger |
/// |---------|---------|
/// | [`WarmError::Wavelet`] | MRA decomposition fails |
/// | [`WarmError::Arma`] | ARMA fitting fails for all components |
/// | [`WarmError::NoComponentsToSimulate`] | MRA produces zero components |
pub fn simulate_warm(data: &[f64], config: &WarmConfig) -> Result<WarmResult, WarmError> {
    let _ = (data, config);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeus_wavelet::{MraConfig, WaveletFilter};

    #[test]
    fn config_defaults() {
        let mra = MraConfig::new(WaveletFilter::La8);
        let config = WarmConfig::new(mra, 1000, 30);

        assert_eq!(config.n_sim(), 1000);
        assert_eq!(config.n_years(), 30);
        assert_eq!(config.seed(), None);
        assert_eq!(config.bypass_n(), 30);
        assert_eq!(config.max_arma_order(), (5, 3));
        assert_eq!(config.mra_config().filter(), WaveletFilter::La8);
    }

    #[test]
    fn config_builder() {
        let mra = MraConfig::new(WaveletFilter::D4);
        let config = WarmConfig::new(mra, 500, 20)
            .with_seed(42)
            .with_bypass_n(50)
            .with_max_arma_order(3, 2);

        assert_eq!(config.n_sim(), 500);
        assert_eq!(config.n_years(), 20);
        assert_eq!(config.seed(), Some(42));
        assert_eq!(config.bypass_n(), 50);
        assert_eq!(config.max_arma_order(), (3, 2));
    }

    #[test]
    fn result_accessors() {
        let sims = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let orders = vec![(1, 0), (2, 1)];
        let fallbacks = vec![false, true];
        let result = WarmResult::new(sims, orders, fallbacks);

        assert_eq!(result.n_sim(), 2);
        assert_eq!(result.n_years(), 3);
        assert_eq!(result.simulations().len(), 2);
        assert_eq!(result.component_orders(), &[(1, 0), (2, 1)]);
        assert_eq!(result.bootstrap_fallbacks(), &[false, true]);
    }

    #[test]
    fn result_empty() {
        let result = WarmResult::new(vec![], vec![], vec![]);
        assert_eq!(result.n_sim(), 0);
        assert_eq!(result.n_years(), 0);
    }

    #[test]
    fn config_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WarmConfig>();
    }

    #[test]
    fn result_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WarmResult>();
    }
}
