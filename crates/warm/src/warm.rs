//! WARM simulation engine.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_arma::select_best_aic;
use zeus_wavelet::{MraConfig, TimeSeries, mra};

use crate::bootstrap::{block_bootstrap, is_arma_viable};
use crate::error::WarmError;
use tracing::{debug, debug_span};
use zeus_stats::{mean, sd};

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
    match_variance: bool,
    var_tol: f64,
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
            match_variance: true,
            var_tol: 0.1,
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

    /// Sets whether to apply variance matching to simulated components.
    pub fn with_match_variance(mut self, match_variance: bool) -> Self {
        self.match_variance = match_variance;
        self
    }

    /// Sets the variance tolerance for variance matching.
    pub fn with_var_tol(mut self, var_tol: f64) -> Self {
        self.var_tol = var_tol;
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

    /// Returns whether variance matching is enabled.
    pub fn match_variance(&self) -> bool {
        self.match_variance
    }

    /// Returns the variance tolerance threshold.
    pub fn var_tol(&self) -> f64 {
        self.var_tol
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

/// Rescales each simulation column so its standard deviation matches `target_sd`.
fn variance_match(sims: &mut [Vec<f64>], target_mean: f64, target_sd: f64, var_tol: f64) {
    for col in sims.iter_mut() {
        let col_mean = mean(col);
        let col_sd = sd(col);
        if col_sd > 1e-10 && target_sd > 1e-10 && ((col_sd - target_sd).abs() / target_sd > var_tol)
        {
            for val in col.iter_mut() {
                *val = (*val - col_mean) * (target_sd / col_sd) + target_mean;
            }
        }
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
#[tracing::instrument(skip(data, config), fields(n = data.len(), n_sim = config.n_sim(), n_years = config.n_years()))]
pub fn simulate_warm(data: &[f64], config: &WarmConfig) -> Result<WarmResult, WarmError> {
    let n_sim = config.n_sim();
    let n_years = config.n_years();
    let (max_p, max_q) = config.max_arma_order();

    // Set up RNG
    let mut rng = match config.seed() {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let data_mean = mean(data);
    let data_sd = sd(data);

    if data.len() < config.bypass_n() {
        // === BYPASS MODE ===
        let bp = max_p.min(2);
        let bq = max_q.min(2);

        if !is_arma_viable(data, bp, bq) {
            debug!("bypass mode: ARMA not viable, falling back to block bootstrap");
            // Bootstrap fallback on original data
            let sims = block_bootstrap(data, n_years, n_sim, &mut rng);
            return Ok(WarmResult::new(sims, vec![(0, 0)], vec![true]));
        }

        let centered: Vec<f64> = data.iter().map(|&x| x - data_mean).collect();
        match select_best_aic(&centered, bp, bq) {
            Ok(fit) => {
                let arr = fit.simulate(n_years, n_sim, &mut rng);
                let mut sims: Vec<Vec<f64>> = (0..n_sim)
                    .map(|j| (0..n_years).map(|i| arr[[i, j]] + data_mean).collect())
                    .collect();
                if config.match_variance() {
                    variance_match(&mut sims, data_mean, data_sd, config.var_tol());
                }
                let order = fit.order();
                Ok(WarmResult::new(sims, vec![order], vec![false]))
            }
            Err(e) => {
                debug!(error = %e, "bypass mode: ARMA fitting failed, falling back to block bootstrap");
                let sims = block_bootstrap(data, n_years, n_sim, &mut rng);
                Ok(WarmResult::new(sims, vec![(0, 0)], vec![true]))
            }
        }
    } else {
        // === COMPONENT MODE ===
        let ts = TimeSeries::new(data.to_vec())?;
        let mra_result = mra(&ts, config.mra_config())?;

        let n_details = mra_result.n_detail_levels();
        let n_components = mra_result.n_components(); // details + smooth
        if n_components == 0 {
            return Err(WarmError::NoComponentsToSimulate);
        }

        // Initialize accumulators for the total simulations
        let mut total_sims: Vec<Vec<f64>> = vec![vec![0.0; n_years]; n_sim];
        let mut component_orders: Vec<(usize, usize)> = Vec::new();
        let mut bootstrap_fallbacks: Vec<bool> = Vec::new();

        // Iterate over components: details 0..n_details, then smooth
        for comp_idx in 0..n_components {
            let comp_type = if comp_idx < n_details {
                "detail"
            } else {
                "smooth"
            };
            let _comp = debug_span!("component", idx = comp_idx, kind = comp_type).entered();

            let component: &[f64] = if comp_idx < n_details {
                mra_result.detail(comp_idx).expect("detail level exists")
            } else {
                mra_result.smooth()
            };

            let comp_mean = mean(component);
            let comp_sd = sd(component);

            // Constant component
            if comp_sd < 1e-10 {
                debug!(sd = comp_sd, "constant component: skipping ARMA");
                for sim in total_sims.iter_mut() {
                    for val in sim.iter_mut() {
                        *val += comp_mean;
                    }
                }
                component_orders.push((0, 0));
                bootstrap_fallbacks.push(false);
                continue;
            }

            // Center the component
            let centered: Vec<f64> = component.iter().map(|&x| x - comp_mean).collect();

            // Determine max orders for this component
            let (cp, cq) = if comp_idx >= n_details {
                // Smooth component
                (max_p.min(1), 0)
            } else {
                // Detail component
                (max_p.min(2), max_q.min(2))
            };

            if !is_arma_viable(&centered, cp, cq) {
                debug!("ARMA not viable: falling back to block bootstrap");
                // Bootstrap on original (non-centered) component
                let comp_sims = block_bootstrap(component, n_years, n_sim, &mut rng);
                for (sim, comp_sim) in total_sims.iter_mut().zip(comp_sims.iter()) {
                    for (s, &c) in sim.iter_mut().zip(comp_sim.iter()) {
                        *s += c;
                    }
                }
                component_orders.push((0, 0));
                bootstrap_fallbacks.push(true);
                continue;
            }

            match select_best_aic(&centered, cp, cq) {
                Ok(fit) => {
                    let arr = fit.simulate(n_years, n_sim, &mut rng);
                    let mut comp_sims: Vec<Vec<f64>> = (0..n_sim)
                        .map(|j| (0..n_years).map(|i| arr[[i, j]] + comp_mean).collect())
                        .collect();
                    if config.match_variance() {
                        variance_match(&mut comp_sims, comp_mean, comp_sd, config.var_tol());
                    }
                    for (sim, comp_sim) in total_sims.iter_mut().zip(comp_sims.iter()) {
                        for (s, &c) in sim.iter_mut().zip(comp_sim.iter()) {
                            *s += c;
                        }
                    }
                    let order = fit.order();
                    component_orders.push(order);
                    bootstrap_fallbacks.push(false);
                }
                Err(e) => {
                    debug!(error = %e, "ARMA fitting failed: falling back to block bootstrap");
                    // ARMA fitting failed, fall back to bootstrap
                    let comp_sims = block_bootstrap(component, n_years, n_sim, &mut rng);
                    for (sim, comp_sim) in total_sims.iter_mut().zip(comp_sims.iter()) {
                        for (s, &c) in sim.iter_mut().zip(comp_sim.iter()) {
                            *s += c;
                        }
                    }
                    component_orders.push((0, 0));
                    bootstrap_fallbacks.push(true);
                }
            }
        }

        Ok(WarmResult::new(
            total_sims,
            component_orders,
            bootstrap_fallbacks,
        ))
    }
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

    #[test]
    fn config_match_variance_defaults() {
        let mra = MraConfig::new(WaveletFilter::La8);
        let config = WarmConfig::new(mra, 100, 30);
        assert!(config.match_variance());
        assert!((config.var_tol() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn config_match_variance_builder() {
        let mra = MraConfig::new(WaveletFilter::La8);
        let config = WarmConfig::new(mra, 100, 30)
            .with_match_variance(false)
            .with_var_tol(0.05);
        assert!(!config.match_variance());
        assert!((config.var_tol() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn simulate_warm_bypass_mode_short_data() {
        // Data shorter than bypass_n should trigger bypass mode
        let mra_config = MraConfig::new(WaveletFilter::La8);
        let config = WarmConfig::new(mra_config, 10, 20)
            .with_seed(42)
            .with_bypass_n(100); // bypass_n > data.len()
        let data: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let result = simulate_warm(&data, &config).unwrap();
        assert_eq!(result.n_sim(), 10);
        assert_eq!(result.n_years(), 20);
    }

    #[test]
    fn simulate_warm_deterministic() {
        let data: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
        let c1 = WarmConfig::new(mra_config.clone(), 5, 30).with_seed(123);
        let c2 = WarmConfig::new(mra_config, 5, 30).with_seed(123);
        let r1 = simulate_warm(&data, &c1).unwrap();
        let r2 = simulate_warm(&data, &c2).unwrap();
        assert_eq!(r1.simulations(), r2.simulations());
    }

    #[test]
    fn simulate_warm_component_mode() {
        // Data >= bypass_n (default 30) should trigger component mode
        let data: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
            .collect();
        let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
        let config = WarmConfig::new(mra_config, 5, 30).with_seed(42);
        let result = simulate_warm(&data, &config).unwrap();
        assert_eq!(result.n_sim(), 5);
        assert_eq!(result.n_years(), 30);
        assert!(!result.component_orders().is_empty());
    }
}
