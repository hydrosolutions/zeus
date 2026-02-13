use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_resample::{MarkovConfig, ObsData, PrecipState, ResampleConfig};

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
fn smoke_test() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 5];
    let mut rng = StdRng::seed_from_u64(42);

    let indices =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng)
            .unwrap();

    assert_eq!(indices.len(), 1825);
    for &idx in &indices {
        assert!(idx < obs.len(), "index {idx} out of bounds");
    }
}

#[test]
fn reproducibility() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 3];

    let mut rng1 = StdRng::seed_from_u64(99);
    let idx1 = zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng1)
        .unwrap();

    let mut rng2 = StdRng::seed_from_u64(99);
    let idx2 = zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng2)
        .unwrap();

    assert_eq!(idx1, idx2);
}

#[test]
fn different_seeds_differ() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 3];

    let mut rng1 = StdRng::seed_from_u64(1);
    let idx1 = zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng1)
        .unwrap();

    let mut rng2 = StdRng::seed_from_u64(9999);
    let idx2 = zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng2)
        .unwrap();

    assert_ne!(idx1, idx2);
}

#[test]
fn multi_year_state_chain() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 3];

    // Approach 1: call resample_year individually, chaining state
    let mut rng1 = StdRng::seed_from_u64(77);
    let mut state = PrecipState::Dry;
    let mut prev_obs_idx: Option<usize> = None;
    let mut manual_indices = Vec::new();

    for &sp in &sim_annual {
        let result = zeus_resample::resample_year(
            sp,
            &obs,
            &markov_config,
            state,
            prev_obs_idx,
            &config,
            &mut rng1,
        )
        .unwrap();
        manual_indices.extend_from_slice(result.indices());
        state = result.final_state();
        prev_obs_idx = Some(result.last_obs_idx());
    }

    // Approach 2: call resample_dates with same seed
    let mut rng2 = StdRng::seed_from_u64(77);
    let batch_indices =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng2)
            .unwrap();

    assert_eq!(manual_indices, batch_indices);
}

#[test]
fn water_year_mode() {
    let obs = make_obs(5);
    let config = ResampleConfig::new().with_year_start_month(10);
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

    assert_eq!(result.indices().len(), 365);
    for &idx in result.indices() {
        assert!(idx < obs.len(), "index {idx} out of bounds");
    }
}

#[test]
fn calendar_year_mode() {
    let obs = make_obs(5);
    let config = ResampleConfig::new().with_year_start_month(1);
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

    assert_eq!(result.indices().len(), 365);
    for &idx in result.indices() {
        assert!(idx < obs.len(), "index {idx} out of bounds");
    }
}

#[test]
fn single_year() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 1];
    let mut rng = StdRng::seed_from_u64(42);

    let indices =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng)
            .unwrap();

    assert_eq!(indices.len(), 365);
    for &idx in &indices {
        assert!(idx < obs.len(), "index {idx} out of bounds");
    }
}

#[test]
fn output_indices_recover_original_data() {
    let obs = make_obs(5);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 3];
    let mut rng = StdRng::seed_from_u64(42);

    let indices =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng)
            .unwrap();

    for &idx in &indices {
        let p = obs.precip()[idx];
        let t = obs.temp()[idx];
        assert!(p.is_finite(), "precip at index {idx} is not finite");
        assert!(p >= 0.0, "precip at index {idx} is negative: {p}");
        assert!(t.is_finite(), "temp at index {idx} is not finite");
    }
}

#[test]
fn large_observed_many_simulated() {
    let obs = make_obs(50);
    let config = ResampleConfig::new();
    let markov_config = MarkovConfig::new();
    let sim_precip = mean_annual_precip(&obs);
    let sim_annual = vec![sim_precip; 20];
    let mut rng = StdRng::seed_from_u64(42);

    let indices =
        zeus_resample::resample_dates(&sim_annual, &obs, &markov_config, &config, &mut rng)
            .unwrap();

    assert_eq!(indices.len(), 7300);
    for &idx in &indices {
        assert!(idx < obs.len(), "index {idx} out of bounds");
    }
}
