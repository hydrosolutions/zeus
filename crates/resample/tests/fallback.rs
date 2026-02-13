//! Integration tests for the fallback cascade in daily resampling.
//!
//! Constructs datasets that are sparse, uniform, or otherwise extreme to
//! exercise the fallback paths in the daily loop.

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

/// Build a sparse dataset: year 1 has only months 1-6, year 2 has only months 7-12.
fn make_sparse_obs() -> ObsData {
    let mut precip = Vec::new();
    let mut temp = Vec::new();
    let mut months = Vec::new();
    let mut days = Vec::new();
    let mut water_years = Vec::new();

    // Year 1: months 1-6
    for m in 1..=6u8 {
        let dim: u8 = match m {
            2 => 28,
            4 | 6 => 30,
            _ => 31,
        };
        for d in 1..=dim {
            precip.push(if d % 3 == 0 { 0.0 } else { (m as f64) * 0.5 });
            temp.push(10.0 + m as f64);
            months.push(m);
            days.push(d);
            water_years.push(2000);
        }
    }
    // Year 2: months 7-12
    for m in 7..=12u8 {
        let dim: u8 = match m {
            9 | 11 => 30,
            _ => 31,
        };
        for d in 1..=dim {
            precip.push(if d % 2 == 0 { 0.0 } else { (m as f64) * 0.8 });
            temp.push(20.0 + m as f64);
            months.push(m);
            days.push(d);
            water_years.push(2001);
        }
    }

    ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap()
}

#[test]
fn fallback_minimal_data() {
    let obs = make_obs(2);
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
    );

    assert!(
        result.is_ok(),
        "resample_year should succeed with minimal 2-year dataset: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}

#[test]
fn fallback_sparse_months() {
    let obs = make_sparse_obs();
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
    );

    assert!(
        result.is_ok(),
        "resample_year should handle sparse month coverage via fallback: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}

#[test]
fn fallback_extreme_precip() {
    // All precip is either 0.0 or 100.0 — hard state transitions.
    let days_per_year = 365;
    let n_years = 3;
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
                let p = if doy % 2 == 0 { 0.0 } else { 100.0 };
                precip.push(p);
                temp.push(15.0 + (m as f64) * 2.0);
                months.push(m);
                days.push(d);
                water_years.push(wy);
                doy += 1;
            }
        }
    }

    let obs = ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap();
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
    );

    assert!(
        result.is_ok(),
        "resample_year should not panic with extreme precip values: {:?}",
        result.err()
    );
}

#[test]
fn fallback_uniform_precip() {
    // All precip identical (1.0) — KNN has identical candidates.
    let days_per_year = 365;
    let n_years = 3;
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
                precip.push(1.0);
                temp.push(15.0 + (m as f64) * 2.0);
                months.push(m);
                days.push(d);
                water_years.push(wy);
                doy += 1;
            }
        }
    }

    let obs = ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap();
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
    );

    assert!(
        result.is_ok(),
        "resample_year should not panic with uniform precip: {:?}",
        result.err()
    );
}

#[test]
fn fallback_two_years_diverse() {
    // Year 1: all dry (precip = 0.0). Year 2: all wet (precip = 10.0).
    let days_per_year = 365;
    let mut precip = Vec::with_capacity(days_per_year * 2);
    let mut temp = Vec::with_capacity(days_per_year * 2);
    let mut months = Vec::with_capacity(days_per_year * 2);
    let mut days = Vec::with_capacity(days_per_year * 2);
    let mut water_years = Vec::with_capacity(days_per_year * 2);

    for y in 0..2usize {
        let wy = 2000 + y as i32;
        let p_val = if y == 0 { 0.0 } else { 10.0 };
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
                precip.push(p_val);
                temp.push(15.0 + (m as f64) * 2.0 + (y as f64) * 5.0);
                months.push(m);
                days.push(d);
                water_years.push(wy);
                doy += 1;
            }
        }
    }

    let obs = ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap();
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
    );

    assert!(
        result.is_ok(),
        "resample_year should work with one all-dry and one all-wet year: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().indices().len(), 365);
}

#[test]
fn fallback_cascade_doesnt_panic() {
    // Minimal 2-year dataset, simulate multiple years to stress the cascade.
    let obs = make_obs(2);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 5]; // 5 simulated years
    let mut rng = StdRng::seed_from_u64(42);

    let result =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng);

    assert!(
        result.is_ok(),
        "resample_dates with 2 obs years and 5 sim years should not panic: {:?}",
        result.err()
    );
    let indices = result.unwrap();
    assert_eq!(indices.len(), 5 * 365);
    for &idx in &indices {
        assert!(idx < obs.len(), "index {} out of bounds", idx);
    }
}
