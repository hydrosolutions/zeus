//! Integration tests for the zeus-perturb pipeline.

use zeus_perturb::{
    OccurrenceConfig, PerturbConfig, PerturbError, QmConfig, ScenarioFactors, TempConfig,
    apply_perturbations,
};

/// Generates synthetic daily data for n_years, 30 days/month (360 days/year).
/// Roughly `wet_frac` of days are wet with Gamma(2,3)-distributed amounts.
#[allow(clippy::type_complexity)]
fn make_synthetic(
    n_years: usize,
    wet_frac: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>, Vec<u32>) {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let gamma = Gamma::new(2.0, 3.0).unwrap();
    let days_per_month = 30;
    let total = n_years * 12 * days_per_month;

    let mut precip = Vec::with_capacity(total);
    let mut temp = Vec::with_capacity(total);
    let mut temp_min = Vec::with_capacity(total);
    let mut temp_max = Vec::with_capacity(total);
    let mut month = Vec::with_capacity(total);
    let mut year = Vec::with_capacity(total);

    for y in 1..=n_years {
        for m in 1u8..=12 {
            for _ in 0..days_per_month {
                use rand::Rng;
                let is_wet: bool = rng.random_bool(wet_frac);
                let p = if is_wet { gamma.sample(&mut rng) } else { 0.0 };
                precip.push(p);
                let base_temp = 10.0 + m as f64 * 2.0;
                temp.push(base_temp);
                temp_min.push(base_temp - 5.0);
                temp_max.push(base_temp + 5.0);
                month.push(m);
                year.push(y as u32);
            }
        }
    }

    (precip, temp, temp_min, temp_max, month, year)
}

// ---------------------------------------------------------------------------
// 1. No modules enabled -- passthrough
// ---------------------------------------------------------------------------

#[test]
fn all_disabled_passthrough() {
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(1, 0.5, 1);
    let config = PerturbConfig::new().with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    // Precip should pass through safety rails unchanged (already clean).
    assert_eq!(result.precip(), &precip[..]);
    // Temperature should be unchanged.
    assert_eq!(result.temp(), &temp[..]);
    assert_eq!(result.temp_min(), Some(temp_min.as_slice()));
    assert_eq!(result.temp_max(), Some(temp_max.as_slice()));

    let ma = result.modules_applied();
    assert!(!ma.temperature);
    assert!(!ma.quantile_map);
    assert!(!ma.occurrence);
    assert!(ma.safety_rails);
}

// ---------------------------------------------------------------------------
// 2. Temperature only -- precip unchanged, temp shifted
// ---------------------------------------------------------------------------

#[test]
fn temperature_only_precip_unchanged() {
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(1, 0.5, 2);
    let delta = 3.0;
    let config = PerturbConfig::new()
        .with_temp(TempConfig::new([delta; 12], false))
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    // Precip must be unchanged (data is clean, safety rails are a no-op).
    assert_eq!(result.precip(), &precip[..]);

    // Each temperature value should be shifted by exactly `delta`.
    for (i, &t) in result.temp().iter().enumerate() {
        assert!(
            (t - (temp[i] + delta)).abs() < 1e-10,
            "temp mismatch at index {i}: expected {}, got {t}",
            temp[i] + delta,
        );
    }

    // temp_min and temp_max should also be shifted.
    if let Some(tmin) = result.temp_min() {
        for (i, &v) in tmin.iter().enumerate() {
            assert!(
                (v - (temp_min[i] + delta)).abs() < 1e-10,
                "temp_min mismatch at index {i}",
            );
        }
    }
    if let Some(tmax) = result.temp_max() {
        for (i, &v) in tmax.iter().enumerate() {
            assert!(
                (v - (temp_max[i] + delta)).abs() < 1e-10,
                "temp_max mismatch at index {i}",
            );
        }
    }

    let ma = result.modules_applied();
    assert!(ma.temperature);
    assert!(!ma.quantile_map);
    assert!(!ma.occurrence);
}

// ---------------------------------------------------------------------------
// 3. QM with factor 1.0 -- near identity
// ---------------------------------------------------------------------------

#[test]
fn qm_with_factor_one_near_identity() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 3);

    let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let qm_config = QmConfig::new();

    let config = PerturbConfig::new()
        .with_qm(qm_config, mean_factors, var_factors)
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    let ma = result.modules_applied();
    assert!(ma.quantile_map);

    // With identity factors the total precipitation should stay within 20%
    // of the original (not exact due to Gamma fitting and remapping).
    let orig_total: f64 = precip.iter().sum();
    let adj_total: f64 = result.precip().iter().sum();
    let ratio = adj_total / orig_total;
    assert!(
        (0.80..=1.20).contains(&ratio),
        "total precip ratio {ratio:.4} is outside 0.80..1.20 (orig={orig_total:.2}, adj={adj_total:.2})",
    );
}

// ---------------------------------------------------------------------------
// 4. Occurrence with factor 0.5 -- roughly halve wet days
// ---------------------------------------------------------------------------

#[test]
fn occurrence_with_factor_half() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 4);

    let occ_config = OccurrenceConfig::new([0.5; 12], false, 0.0);
    let config = PerturbConfig::new()
        .with_occurrence(occ_config)
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    let ma = result.modules_applied();
    assert!(ma.occurrence);

    let orig_wet = precip.iter().filter(|&&p| p > 0.0).count();
    let adj_wet = result.precip().iter().filter(|&&p| p > 0.0).count();

    // The adjusted wet-day count should be less than 70% of the original.
    assert!(
        adj_wet < (orig_wet as f64 * 0.70) as usize,
        "expected adj_wet ({adj_wet}) < 70% of orig_wet ({orig_wet})",
    );
}

// ---------------------------------------------------------------------------
// 5. Full pipeline -- all modules enabled
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_all_enabled() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 5);

    let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let qm_config = QmConfig::new();
    let occ_config = OccurrenceConfig::new([1.0; 12], false, 0.0);

    let config = PerturbConfig::new()
        .with_temp(TempConfig::new([2.0; 12], false))
        .with_qm(qm_config, mean_factors, var_factors)
        .with_occurrence(occ_config)
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    // All modules should be applied.
    let ma = result.modules_applied();
    assert!(ma.temperature);
    assert!(ma.quantile_map);
    assert!(ma.occurrence);
    assert!(ma.safety_rails);

    // Output lengths must match input.
    assert_eq!(result.precip().len(), precip.len());
    assert_eq!(result.temp().len(), temp.len());

    // All precip values must be finite and non-negative.
    for (i, &p) in result.precip().iter().enumerate() {
        assert!(p.is_finite(), "precip[{i}] is not finite: {p}");
        assert!(p >= 0.0, "precip[{i}] is negative: {p}");
    }
}

// ---------------------------------------------------------------------------
// 6. Deterministic with seed -- two runs must be identical
// ---------------------------------------------------------------------------

#[test]
fn deterministic_with_seed() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 6);

    let make_config = || {
        let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
        PerturbConfig::new()
            .with_temp(TempConfig::new([1.5; 12], false))
            .with_qm(QmConfig::new(), mean_factors, var_factors)
            .with_occurrence(OccurrenceConfig::new([0.8; 12], false, 0.0))
            .with_seed(999)
    };

    let r1 = apply_perturbations(
        &precip,
        &temp,
        &temp_min,
        &temp_max,
        &month,
        &year,
        &make_config(),
    )
    .expect("run 1 should succeed");

    let r2 = apply_perturbations(
        &precip,
        &temp,
        &temp_min,
        &temp_max,
        &month,
        &year,
        &make_config(),
    )
    .expect("run 2 should succeed");

    assert_eq!(r1.precip(), r2.precip(), "precip differs between runs");
    assert_eq!(r1.temp(), r2.temp(), "temp differs between runs");
}

// ---------------------------------------------------------------------------
// 7. Empty input -- returns EmptyData error
// ---------------------------------------------------------------------------

#[test]
fn empty_input_returns_error() {
    let config = PerturbConfig::new();
    let result = apply_perturbations(&[], &[], &[], &[], &[], &[], &config);
    assert!(
        matches!(result, Err(PerturbError::EmptyData)),
        "expected EmptyData, got {result:?}",
    );
}

// ---------------------------------------------------------------------------
// 8. Length mismatch -- returns LengthMismatch error
// ---------------------------------------------------------------------------

#[test]
fn length_mismatch_returns_error() {
    let config = PerturbConfig::new();
    let result = apply_perturbations(
        &[1.0, 2.0],
        &[20.0], // wrong length
        &[15.0, 16.0],
        &[25.0, 26.0],
        &[1, 2],
        &[1, 1],
        &config,
    );
    assert!(
        matches!(result, Err(PerturbError::LengthMismatch { .. })),
        "expected LengthMismatch, got {result:?}",
    );
}

// ---------------------------------------------------------------------------
// 9. No negative precipitation -- even with occurrence factor 2.0
// ---------------------------------------------------------------------------

#[test]
fn no_negative_precipitation() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 9);

    let occ_config = OccurrenceConfig::new([2.0; 12], false, 0.0);
    let config = PerturbConfig::new()
        .with_occurrence(occ_config)
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    for (i, &p) in result.precip().iter().enumerate() {
        assert!(p >= 0.0, "precip[{i}] is negative: {p}");
    }
}

// ---------------------------------------------------------------------------
// 10. All output values finite -- no NaN or Inf
// ---------------------------------------------------------------------------

#[test]
fn all_output_values_finite() {
    let n_years = 3;
    let (precip, temp, temp_min, temp_max, month, year) = make_synthetic(n_years, 0.5, 10);

    let mean_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let var_factors = ScenarioFactors::uniform(n_years, [1.0; 12]);
    let occ_config = OccurrenceConfig::new([1.5; 12], false, 0.0);

    let config = PerturbConfig::new()
        .with_temp(TempConfig::new([1.0; 12], false))
        .with_qm(QmConfig::new(), mean_factors, var_factors)
        .with_occurrence(occ_config)
        .with_seed(42);

    let result = apply_perturbations(&precip, &temp, &temp_min, &temp_max, &month, &year, &config)
        .expect("should succeed");

    for (i, &p) in result.precip().iter().enumerate() {
        assert!(p.is_finite(), "precip[{i}] is not finite: {p}");
    }
    for (i, &t) in result.temp().iter().enumerate() {
        assert!(t.is_finite(), "temp[{i}] is not finite: {t}");
    }
    if let Some(tmin) = result.temp_min() {
        for (i, &v) in tmin.iter().enumerate() {
            assert!(v.is_finite(), "temp_min[{i}] is not finite: {v}");
        }
    }
    if let Some(tmax) = result.temp_max() {
        for (i, &v) in tmax.iter().enumerate() {
            assert!(v.is_finite(), "temp_max[{i}] is not finite: {v}");
        }
    }
}
