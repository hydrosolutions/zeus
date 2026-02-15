use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_resample::{MarkovConfig, ObsData, PrecipState, ResampleConfig, ResampleError};

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
fn error_empty_data() {
    let result = ObsData::new(&[], &[], &[], &[], &[]);
    assert!(matches!(result, Err(ResampleError::EmptyData)));
}

#[test]
fn error_length_mismatch_temp() {
    let result = ObsData::new(&[1.0, 2.0], &[1.0], &[1, 1], &[1, 2], &[2000, 2000]);
    assert!(matches!(
        result,
        Err(ResampleError::LengthMismatch {
            field: "temp",
            expected: 2,
            got: 1,
        })
    ));
}

#[test]
fn error_length_mismatch_months() {
    let result = ObsData::new(&[1.0, 2.0], &[1.0, 2.0], &[1], &[1, 2], &[2000, 2000]);
    assert!(matches!(
        result,
        Err(ResampleError::LengthMismatch {
            field: "months",
            expected: 2,
            got: 1,
        })
    ));
}

#[test]
fn error_length_mismatch_days() {
    let result = ObsData::new(&[1.0, 2.0], &[1.0, 2.0], &[1, 1], &[1], &[2000, 2000]);
    assert!(matches!(
        result,
        Err(ResampleError::LengthMismatch {
            field: "days",
            expected: 2,
            got: 1,
        })
    ));
}

#[test]
fn error_length_mismatch_water_years() {
    let result = ObsData::new(&[1.0, 2.0], &[1.0, 2.0], &[1, 1], &[1, 2], &[2000]);
    assert!(matches!(
        result,
        Err(ResampleError::LengthMismatch {
            field: "water_years",
            expected: 2,
            got: 1,
        })
    ));
}

#[test]
fn error_non_finite_precip() {
    let result = ObsData::new(&[f64::NAN], &[1.0], &[1], &[1], &[2000]);
    assert!(matches!(
        result,
        Err(ResampleError::NonFiniteInput {
            field: "precip",
            ..
        })
    ));
}

#[test]
fn error_non_finite_temp() {
    let result = ObsData::new(&[1.0], &[f64::INFINITY], &[1], &[1], &[2000]);
    assert!(matches!(
        result,
        Err(ResampleError::NonFiniteInput { field: "temp", .. })
    ));
}

#[test]
fn error_non_finite_sim_precip() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let mut rng = StdRng::seed_from_u64(42);

    let result =
        zeus_resample::resample_dates(&[f64::NAN], &obs, &markov_config, &config, &mut rng);

    assert!(matches!(
        result,
        Err(ResampleError::NonFiniteInput {
            field: "sim_annual_precip",
            ..
        })
    ));
}

#[test]
fn error_invalid_config() {
    let obs = make_obs(5);
    let config = ResampleConfig::new().with_annual_knn_n(0);
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

    assert!(matches!(result, Err(ResampleError::InvalidConfig { .. })));
}

#[test]
fn error_insufficient_years() {
    let obs = make_obs(1);
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

    assert!(matches!(
        result,
        Err(ResampleError::InsufficientData { n: 1, min: 2 })
    ));
}
