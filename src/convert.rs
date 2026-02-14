//! Pure conversion functions: TOML config structs -> crate API config types.

use anyhow::{Result, bail};

use crate::config::*;

// Import crate types
use zeus_evaluate::EvaluateConfig;
use zeus_io::{Compression, ReaderConfig, WriterConfig};
use zeus_markov::{MarkovConfig, ThresholdSpec};
use zeus_perturb::{OccurrenceConfig, PerturbConfig, QmConfig, ScenarioFactors, TempConfig};
use zeus_resample::{ResampleConfig, Sampling};
use zeus_warm::{FilterBounds, WarmConfig};
use zeus_wavelet::{MraConfig, WaveletFilter};

/// Parses a wavelet filter name string into the corresponding enum variant.
pub fn parse_wavelet_filter(s: &str) -> Result<WaveletFilter> {
    match s.to_lowercase().as_str() {
        "haar" => Ok(WaveletFilter::Haar),
        "d4" => Ok(WaveletFilter::D4),
        "d6" => Ok(WaveletFilter::D6),
        "d8" => Ok(WaveletFilter::D8),
        "la8" => Ok(WaveletFilter::La8),
        "la16" => Ok(WaveletFilter::La16),
        other => bail!("unknown wavelet filter: {other:?}"),
    }
}

/// Parses a compression algorithm name string into the corresponding enum variant.
pub fn parse_compression(s: &str) -> Result<Compression> {
    match s.to_lowercase().as_str() {
        "none" => Ok(Compression::None),
        "snappy" => Ok(Compression::Snappy),
        "zstd" => Ok(Compression::Zstd),
        other => bail!("unknown compression: {other:?}"),
    }
}

/// Parses a KNN sampling strategy name string into the corresponding enum variant.
pub fn parse_sampling(s: &str) -> Result<Sampling> {
    match s.to_lowercase().as_str() {
        "uniform" => Ok(Sampling::Uniform),
        "rank" => Ok(Sampling::Rank),
        "gaussian" => Ok(Sampling::Gaussian { bandwidth: None }),
        other => bail!("unknown sampling method: {other:?}"),
    }
}

/// Converts a TOML threshold specification into a `ThresholdSpec`.
///
/// Exactly one of `fixed` or `quantile` must be set.
pub fn parse_threshold(t: &ThresholdToml) -> Result<ThresholdSpec> {
    match (t.fixed, t.quantile) {
        (Some(v), None) => Ok(ThresholdSpec::Fixed(v)),
        (None, Some(v)) => Ok(ThresholdSpec::Quantile(v)),
        (Some(_), Some(_)) => {
            bail!("threshold must have exactly one of fixed or quantile, got both")
        }
        (None, None) => {
            bail!("threshold must have exactly one of fixed or quantile, got neither")
        }
    }
}

/// Builds a [`ReaderConfig`] from the TOML I/O configuration.
pub fn build_reader_config(io: &IoConfig) -> Result<ReaderConfig> {
    let mut cfg = ReaderConfig::default()
        .with_precip_var(&io.precip_var)
        .with_start_month(io.start_month)
        .with_trim_to_water_years(io.trim_to_water_years);
    if let Some(ref v) = io.temp_max_var {
        cfg = cfg.with_temp_max_var(Some(v));
    } else {
        cfg = cfg.with_temp_max_var(None::<&str>);
    }
    if let Some(ref v) = io.temp_min_var {
        cfg = cfg.with_temp_min_var(Some(v));
    } else {
        cfg = cfg.with_temp_min_var(None::<&str>);
    }
    Ok(cfg)
}

/// Builds a [`WarmConfig`] from the TOML warm configuration.
///
/// An optional global seed is forwarded to the WARM RNG.
pub fn build_warm_config(warm: &WarmToml, seed: Option<u64>) -> Result<WarmConfig> {
    let filter = parse_wavelet_filter(&warm.wavelet_filter)?;
    let mut mra = MraConfig::new(filter);
    if let Some(levels) = warm.mra_levels {
        mra = mra.with_levels(levels);
    }
    let mut cfg = WarmConfig::new(mra, warm.n_sim, warm.n_years)
        .with_bypass_n(warm.bypass_n)
        .with_max_arma_order(warm.max_arma_order[0], warm.max_arma_order[1])
        .with_match_variance(warm.match_variance)
        .with_var_tol(warm.var_tol);
    if let Some(s) = seed {
        cfg = cfg.with_seed(s);
    }
    Ok(cfg)
}

/// Builds a [`FilterBounds`] from the TOML filter configuration.
pub fn build_filter_bounds(filter: &FilterToml) -> FilterBounds {
    FilterBounds::default()
        .with_n_select(filter.n_select)
        .with_mean_tol(filter.mean_tol)
        .with_sd_tol(filter.sd_tol)
        .with_tail_low_p(filter.tail_low_p)
        .with_tail_high_p(filter.tail_high_p)
        .with_spectral_corr_min(filter.spectral_corr_min)
        .with_peak_match_frac_min(filter.peak_match_frac_min)
        .with_n_sig_peaks_max(filter.n_sig_peaks_max)
}

/// Builds a [`ResampleConfig`] from the TOML resample configuration.
///
/// The `start_month` is passed through from the I/O layer to set the water-year
/// boundary.
pub fn build_resample_config(resample: &ResampleToml, start_month: u8) -> Result<ResampleConfig> {
    let sampling = parse_sampling(&resample.sampling)?;
    Ok(ResampleConfig::new()
        .with_annual_knn_n(resample.annual_knn_n)
        .with_precip_weight(resample.precip_weight)
        .with_temp_weight(resample.temp_weight)
        .with_sd_floor(resample.sd_floor)
        .with_narrow_window(resample.narrow_window as u16)
        .with_wide_window(resample.wide_window as u16)
        .with_sampling(sampling)
        .with_year_start_month(start_month))
}

/// Builds a [`MarkovConfig`] from the TOML Markov configuration.
pub fn build_markov_config(markov: &MarkovToml) -> Result<MarkovConfig> {
    let wet = parse_threshold(&markov.wet_threshold)?;
    let extreme = parse_threshold(&markov.extreme_threshold)?;
    let mut cfg = MarkovConfig::new()
        .with_wet_spec(wet)
        .with_extreme_spec(extreme)
        .with_dirichlet_alpha(markov.dirichlet_alpha);
    if let Some(factors) = markov.dry_spell_factors {
        cfg = cfg.with_dry_spell_factors(factors);
    }
    if let Some(factors) = markov.wet_spell_factors {
        cfg = cfg.with_wet_spell_factors(factors);
    }
    Ok(cfg)
}

/// Builds a [`PerturbConfig`] from the TOML perturbation configuration.
///
/// `n_years` is required by [`ScenarioFactors::uniform`] when QM is enabled.
/// An optional global seed is forwarded to the perturbation RNG.
pub fn build_perturb_config(
    perturb: &PerturbToml,
    n_years: usize,
    seed: Option<u64>,
) -> Result<PerturbConfig> {
    let mut cfg = PerturbConfig::new()
        .with_precip_floor(perturb.precip_floor)
        .with_precip_cap(perturb.precip_cap);
    if let Some(ref temp) = perturb.temperature {
        cfg = cfg.with_temp(TempConfig::new(temp.deltas, temp.transient));
    }
    if let Some(ref qm) = perturb.quantile_map {
        let qm_cfg = QmConfig::new().with_intensity_threshold(qm.intensity_threshold);
        let mean_factors = ScenarioFactors::uniform(n_years, qm.mean_factors);
        let var_factors = ScenarioFactors::uniform(n_years, qm.var_factors);
        cfg = cfg.with_qm(qm_cfg, mean_factors, var_factors);
    }
    if let Some(ref occ) = perturb.occurrence {
        cfg = cfg.with_occurrence(OccurrenceConfig::new(
            occ.factors,
            occ.transient,
            occ.intensity_threshold,
        ));
    }
    if let Some(s) = seed {
        cfg = cfg.with_seed(s);
    }
    Ok(cfg)
}

/// Builds a [`WriterConfig`] from the TOML I/O configuration.
pub fn build_writer_config(io: &IoConfig) -> Result<WriterConfig> {
    let compression = parse_compression(&io.compression)?;
    Ok(WriterConfig::default()
        .with_compression(compression)
        .with_row_group_size(io.row_group_size))
}

/// Builds an [`EvaluateConfig`] from the TOML evaluate configuration.
pub fn build_evaluate_config(eval: &EvaluateToml) -> EvaluateConfig {
    EvaluateConfig::default().with_precip_threshold(eval.precip_threshold)
}
