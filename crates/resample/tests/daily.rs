//! Integration tests for the daily resampling loop.
//!
//! Exercises `resample_year` and `resample_dates` to verify day-level
//! behaviour: correct number of indices, valid bounds, month alignment,
//! and sensitivity to sampling strategy and window parameters.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_resample::{MarkovConfig, ObsData, PrecipState, ResampleConfig, Sampling};

/// Build a synthetic multi-year observation dataset (~40 % dry days).
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

fn mean_annual_precip(obs: &ObsData) -> f64 {
    let ap = obs.annual_precip();
    ap.iter().sum::<f64>() / ap.len() as f64
}

#[test]
fn daily_basic() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let mut rng = StdRng::seed_from_u64(42);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    )
    .expect("resample_year should succeed");

    assert_eq!(
        result.indices().len(),
        365,
        "resample_year must produce exactly 365 indices"
    );
}

#[test]
fn daily_indices_valid() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let mut rng = StdRng::seed_from_u64(42);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    )
    .unwrap();

    for &idx in result.indices() {
        assert!(
            idx < obs.len(),
            "index {} out of bounds (obs.len() = {})",
            idx,
            obs.len()
        );
    }
}

#[test]
fn daily_first_day_within_expected_months() {
    let obs = make_obs(5);
    // Default year_start_month = 1 (January)
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let mut rng = StdRng::seed_from_u64(42);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    )
    .unwrap();

    let first_idx = result.indices()[0];
    let first_month = obs.months()[first_idx];
    // With year_start_month=1, the first day should be matched from month 1
    // (or nearby due to the narrow window). Typically months 12, 1, or 2.
    assert!(
        first_month <= 2 || first_month == 12,
        "first day month should be near January (year_start_month=1), got month {}",
        first_month,
    );
}

#[test]
fn daily_sampling_uniform_vs_rank() {
    let obs = make_obs(5);
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);

    let config_rank = ResampleConfig::new().with_sampling(Sampling::Rank);
    let config_uniform = ResampleConfig::new().with_sampling(Sampling::Uniform);

    let mut rng1 = StdRng::seed_from_u64(42);
    let result_rank = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config_rank,
        &mut rng1,
    )
    .unwrap();

    let mut rng2 = StdRng::seed_from_u64(42);
    let result_uniform = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config_uniform,
        &mut rng2,
    )
    .unwrap();

    // Different sampling strategies should produce different index sequences
    assert_ne!(
        result_rank.indices(),
        result_uniform.indices(),
        "Rank and Uniform sampling should produce different index sequences"
    );
}

#[test]
fn daily_window_size_affects_output() {
    let obs = make_obs(5);
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);

    let config_narrow = ResampleConfig::new()
        .with_narrow_window(1)
        .with_wide_window(1);
    let config_wide = ResampleConfig::new()
        .with_narrow_window(30)
        .with_wide_window(30);

    let mut rng1 = StdRng::seed_from_u64(42);
    let result_narrow = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config_narrow,
        &mut rng1,
    )
    .unwrap();

    let mut rng2 = StdRng::seed_from_u64(42);
    let result_wide = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config_wide,
        &mut rng2,
    )
    .unwrap();

    // Different window sizes should lead to different candidate pools
    assert_ne!(
        result_narrow.indices(),
        result_wide.indices(),
        "narrow_window=1 and narrow_window=30 should produce different index sequences"
    );
}
