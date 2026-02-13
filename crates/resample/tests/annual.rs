//! Integration tests for annual KNN selection behaviour.
//!
//! Since `select_annual_years` is `pub(crate)`, we exercise it indirectly
//! through `resample_year` and `resample_dates`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_resample::{MarkovConfig, ObsData, PrecipState, ResampleConfig};

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
fn annual_knn_basic() {
    let obs = make_obs(5);
    let config = ResampleConfig::new(); // annual_knn_n = 100 (default)
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
    );

    assert!(
        result.is_ok(),
        "resample_year should succeed with 5 obs years"
    );
    let result = result.unwrap();
    assert_eq!(result.indices().len(), 365);
}

#[test]
fn annual_knn_extreme_high_target() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let max_annual = obs
        .annual_precip()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let sim_precip = max_annual * 2.0;
    let mut rng = StdRng::seed_from_u64(99);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    );

    assert!(
        result.is_ok(),
        "resample_year should succeed even with extreme high target precip"
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}

#[test]
fn annual_knn_extreme_low_target() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = 0.001; // near-zero
    let mut rng = StdRng::seed_from_u64(77);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    );

    assert!(
        result.is_ok(),
        "resample_year should succeed with near-zero target precip"
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}

#[test]
fn annual_knn_custom_n() {
    let obs = make_obs(10);
    let config = ResampleConfig::new().with_annual_knn_n(200);
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let mut rng = StdRng::seed_from_u64(123);

    let result = zeus_resample::resample_year(
        sim_precip,
        &obs,
        &markov_config,
        PrecipState::Dry,
        None,
        &config,
        &mut rng,
    );

    assert!(
        result.is_ok(),
        "resample_year should succeed with annual_knn_n=200 and 10 obs years"
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}
