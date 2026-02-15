//! Orchestration functions that chain annual, year-subset, and daily stages.

use crate::config::ResampleConfig;
use crate::error::ResampleError;
use crate::obs_data::ObsData;
use crate::result::ResampleResult;
use tracing::debug_span;
use zeus_markov::{MarkovConfig, PrecipState};

/// Resamples a single simulated year (365 daily observation indices).
///
/// Chains: annual KNN -> year subset -> daily loop.
///
/// # Arguments
/// - `sim_annual_precip` — total annual precipitation for this simulated year
/// - `obs` — observed daily data with precomputed lookups
/// - `markov_config` — Markov chain configuration
/// - `initial_state` — precipitation state of the previous day (use `PrecipState::Dry` for year 1)
/// - `prev_obs_idx` — observation index of the previous day (None for year 1)
/// - `config` — resampler configuration
/// - `rng` — random number generator
#[tracing::instrument(skip_all, fields(sim_annual_precip, initial_state = ?initial_state, prev_obs_idx))]
#[allow(clippy::too_many_arguments)]
pub fn resample_year(
    sim_annual_precip: f64,
    obs: &ObsData,
    markov_config: &MarkovConfig,
    initial_state: PrecipState,
    prev_obs_idx: Option<usize>,
    config: &ResampleConfig,
    rng: &mut impl rand::Rng,
) -> Result<ResampleResult, ResampleError> {
    config.validate()?;

    let n = obs.n_obs_years();
    if n < 2 {
        return Err(ResampleError::InsufficientData { n, min: 2 });
    }

    // Derive simulated months and days from the calendar.
    let start = zeus_calendar::NoLeapDate::new(2000, config.year_start_month(), 1)?;
    let dates = zeus_calendar::noleap_sequence(start, 365);
    let sim_months: Vec<u8> = dates.iter().map(|d| d.month()).collect();
    let sim_days: Vec<u8> = dates.iter().map(|d| d.day()).collect();

    // Annual KNN year selection.
    let year_indices = crate::annual::select_annual_years(sim_annual_precip, obs, config, rng)?;

    // Build year subset with statistics and Markov model.
    let subset = crate::year_subset::YearSubset::build(&year_indices, obs, config, markov_config)?;

    // Daily resampling loop.
    crate::daily::resample_days(
        &subset,
        obs,
        &sim_months,
        &sim_days,
        initial_state,
        prev_obs_idx,
        config,
        rng,
    )
}

/// Resamples multiple years of daily observation indices.
///
/// Calls `resample_year` for each element in `sim_annual_precip`,
/// chaining Markov state and observation index across years.
///
/// Returns a flat `Vec<usize>` of length `sim_annual_precip.len() * 365`.
#[tracing::instrument(skip_all, fields(n_years = sim_annual_precip.len()))]
pub fn resample_dates(
    sim_annual_precip: &[f64],
    obs: &ObsData,
    markov_config: &MarkovConfig,
    config: &ResampleConfig,
    rng: &mut impl rand::Rng,
) -> Result<Vec<usize>, ResampleError> {
    config.validate()?;

    if sim_annual_precip.is_empty() {
        return Ok(Vec::new());
    }

    if let Some(idx) = sim_annual_precip.iter().position(|v| !v.is_finite()) {
        return Err(ResampleError::NonFiniteInput {
            field: "sim_annual_precip",
            first_bad_index: Some(idx),
        });
    }

    let mut state = PrecipState::Dry;
    let mut prev_obs_idx: Option<usize> = None;
    let mut all_indices = Vec::with_capacity(sim_annual_precip.len() * 365);

    for (year_idx, &sim_precip) in sim_annual_precip.iter().enumerate() {
        let _yr = debug_span!("year", idx = year_idx, target_precip = sim_precip).entered();
        let result = resample_year(
            sim_precip,
            obs,
            markov_config,
            state,
            prev_obs_idx,
            config,
            rng,
        )
        .map_err(|e| match e {
            ResampleError::NoCandidates { day, month, .. } => ResampleError::NoCandidates {
                day,
                month,
                year: Some(year_idx),
            },
            other => other,
        })?;
        all_indices.extend_from_slice(result.indices());
        state = result.final_state();
        prev_obs_idx = Some(result.last_obs_idx());
    }

    Ok(all_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Helper: create a multi-year observation dataset with ~40% dry days.
    fn make_obs(n_years: usize) -> ObsData {
        let days_per_year = 365;
        let n = n_years * days_per_year;
        let mut precip = Vec::with_capacity(n);
        let mut temp = Vec::with_capacity(n);
        let mut months = Vec::with_capacity(n);
        let mut days = Vec::with_capacity(n);
        let mut water_years = Vec::with_capacity(n);

        for y in 0..n_years {
            let wy = 2000 + y as i32;
            let mut doy = 0;
            for m in 1..=12u8 {
                let dim: u8 = match m {
                    2 => 28,
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                for d in 1..=dim {
                    if doy >= days_per_year {
                        break;
                    }
                    let p = if (doy * 7 + y * 13) % 10 < 4 {
                        0.0
                    } else {
                        ((doy as f64) * 0.1 + (y as f64) * 3.0).max(0.1)
                    };
                    precip.push(p);
                    temp.push(15.0 + (m as f64) * 2.0 + (y as f64) * 0.5);
                    months.push(m);
                    days.push(d);
                    water_years.push(wy);
                    doy += 1;
                }
            }
        }

        ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap()
    }

    /// Compute the mean of observed annual precipitation.
    fn mean_annual_precip(obs: &ObsData) -> f64 {
        let ap = obs.annual_precip();
        ap.iter().sum::<f64>() / ap.len() as f64
    }

    #[test]
    fn single_year_smoke() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_year(
            sim_precip,
            &obs,
            &markov_config,
            PrecipState::Dry,
            None,
            &config,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.indices().len(), 365);
        for &idx in result.indices() {
            assert!(idx < obs.len(), "index {idx} out of bounds");
        }
    }

    #[test]
    fn multi_year_smoke() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let sim_annual = vec![sim_precip; 3];
        let mut rng = StdRng::seed_from_u64(42);

        let indices = resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng).unwrap();

        assert_eq!(indices.len(), 1095);
        for &idx in &indices {
            assert!(idx < obs.len(), "index {idx} out of bounds");
        }
    }

    #[test]
    fn reproducible() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let sim_annual = vec![sim_precip; 2];

        let mut rng1 = StdRng::seed_from_u64(99);
        let idx1 = resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng1).unwrap();

        let mut rng2 = StdRng::seed_from_u64(99);
        let idx2 = resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng2).unwrap();

        assert_eq!(idx1, idx2);
    }

    #[test]
    fn different_seeds_differ() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let sim_annual = vec![sim_precip; 2];

        let mut rng1 = StdRng::seed_from_u64(1);
        let idx1 = resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng1).unwrap();

        let mut rng2 = StdRng::seed_from_u64(9999);
        let idx2 = resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng2).unwrap();

        assert_ne!(idx1, idx2);
    }

    #[test]
    fn empty_sim_precip() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let mut rng = StdRng::seed_from_u64(42);

        let indices = resample_dates(&[], &obs, &markov_config, &config, &mut rng).unwrap();

        assert!(indices.is_empty());
    }

    #[test]
    fn non_finite_sim_precip() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_dates(&[f64::NAN], &obs, &markov_config, &config, &mut rng);

        assert!(matches!(
            result,
            Err(ResampleError::NonFiniteInput {
                field: "sim_annual_precip",
                ..
            })
        ));
    }

    #[test]
    fn insufficient_years() {
        let obs = make_obs(1);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_year(
            sim_precip,
            &obs,
            &markov_config,
            PrecipState::Dry,
            None,
            &config,
            &mut rng,
        );

        assert!(matches!(
            result,
            Err(ResampleError::InsufficientData { n: 1, min: 2 })
        ));
    }

    #[test]
    fn water_year_mode() {
        let obs = make_obs(5);
        let config = ResampleConfig::new().with_year_start_month(10);
        let markov_config = MarkovConfig::new();
        let sim_precip = mean_annual_precip(&obs);
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_year(
            sim_precip,
            &obs,
            &markov_config,
            PrecipState::Dry,
            None,
            &config,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.indices().len(), 365);
        for &idx in result.indices() {
            assert!(idx < obs.len(), "index {idx} out of bounds");
        }
    }
}
