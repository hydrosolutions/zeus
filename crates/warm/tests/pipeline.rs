use zeus_warm::{FilterBounds, WarmConfig, WarmError, filter_warm_pool, simulate_warm};
use zeus_wavelet::{MraConfig, WaveletFilter};

/// Generates a 128-point AR(1)-like series with a sine component and offset.
fn synthetic_128() -> Vec<f64> {
    let mut data = vec![0.0_f64; 128];
    // Seed the noise manually for reproducibility (simple LCG).
    let mut lcg: u64 = 12345;
    data[0] = 50.0;
    for i in 1..128 {
        // Simple LCG pseudo-random in [-1, 1]
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((lcg >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
        let noise = u * 1.0; // noise amplitude
        let sine = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin();
        data[i] = 0.7 * data[i - 1] + noise + sine + 50.0 * 0.3;
    }
    data
}

#[test]
fn full_pipeline_smoke() {
    let data = synthetic_128();
    assert_eq!(data.len(), 128);

    let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
    let config = WarmConfig::new(mra_config, 50, 128).with_seed(42);

    // Simulate
    let result = simulate_warm(&data, &config).unwrap();
    assert_eq!(result.n_sim(), 50);
    assert_eq!(result.n_years(), 128);

    // Filter
    let bounds = FilterBounds::default().with_n_select(5);
    let pool = filter_warm_pool(&data, &result, &bounds).unwrap();

    assert!(
        pool.n_selected() > 0,
        "should select at least one simulation"
    );
    assert!(
        pool.n_selected() <= 5,
        "should select at most 5 simulations"
    );

    // All selected indices must be valid
    for &idx in pool.selected() {
        assert!(idx < 50, "selected index {} should be < 50", idx);
    }

    // Scores and selected must have the same length
    assert_eq!(
        pool.scores().len(),
        pool.selected().len(),
        "scores and selected must have the same length"
    );
}

#[test]
fn simulate_warm_bypass_mode() {
    // Data of length 20, which is < bypass_n default of 30
    let data: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.3).sin() * 4.0 + 8.0)
        .collect();
    assert_eq!(data.len(), 20);

    let mra_config = MraConfig::new(WaveletFilter::La8);
    let config = WarmConfig::new(mra_config, 10, 20).with_seed(77);

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

    let config_a = WarmConfig::new(mra_config.clone(), 10, 64).with_seed(999);
    let config_b = WarmConfig::new(mra_config, 10, 64).with_seed(999);

    let result_a = simulate_warm(&data, &config_a).unwrap();
    let result_b = simulate_warm(&data, &config_b).unwrap();

    assert_eq!(result_a.n_sim(), result_b.n_sim());
    assert_eq!(result_a.n_years(), result_b.n_years());

    // All simulations must be exactly equal
    for (sim_a, sim_b) in result_a
        .simulations()
        .iter()
        .zip(result_b.simulations().iter())
    {
        assert_eq!(sim_a.len(), sim_b.len());
        for (a, b) in sim_a.iter().zip(sim_b.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "simulations must be bit-identical"
            );
        }
    }
}

#[test]
fn filter_warm_pool_respects_n_select() {
    let data: Vec<f64> = (0..64)
        .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
        .collect();

    let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
    let config = WarmConfig::new(mra_config, 30, 64).with_seed(123);

    let result = simulate_warm(&data, &config).unwrap();
    assert_eq!(result.n_sim(), 30);

    let bounds = FilterBounds::default().with_n_select(3);
    let pool = filter_warm_pool(&data, &result, &bounds).unwrap();

    assert!(
        pool.n_selected() <= 3,
        "n_selected ({}) should be <= 3",
        pool.n_selected()
    );
}

#[test]
fn filter_warm_pool_relaxation_fallback() {
    let data: Vec<f64> = (0..64)
        .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
        .collect();

    let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
    let config = WarmConfig::new(mra_config, 20, 64).with_seed(55);

    let result = simulate_warm(&data, &config).unwrap();

    // Extremely tight bounds -- nothing should pass initially
    let bounds = FilterBounds::default()
        .with_mean_tol(0.00001)
        .with_sd_tol(0.00001)
        .with_n_select(3);

    let pool = filter_warm_pool(&data, &result, &bounds).unwrap();

    assert!(
        pool.n_selected() > 0,
        "relaxation fallback should still produce results"
    );
}

#[test]
fn filter_warm_pool_empty_pool_errors() {
    let data: Vec<f64> = (0..64)
        .map(|i| (i as f64 * 0.1).sin() * 5.0 + 10.0)
        .collect();

    let mra_config = MraConfig::new(WaveletFilter::La8).with_levels(3);
    // n_sim = 0 should produce an empty result
    let config = WarmConfig::new(mra_config, 0, 64).with_seed(42);

    let result = simulate_warm(&data, &config).unwrap();
    assert_eq!(result.n_sim(), 0);

    let bounds = FilterBounds::default();
    let err = filter_warm_pool(&data, &result, &bounds);

    assert!(err.is_err(), "filtering an empty pool should error");
    match err.unwrap_err() {
        WarmError::InsufficientSimulations {
            requested,
            available,
        } => {
            assert_eq!(available, 0);
            assert_eq!(requested, bounds.n_select());
        }
        other => panic!("expected InsufficientSimulations, got: {:?}", other),
    }
}
