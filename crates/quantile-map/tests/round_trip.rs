use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma as GammaDist};
use zeus_quantile_map::{QmConfig, QuantileMapError, ScenarioFactors, adjust_precipitation};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generates Gamma-distributed seasonal precipitation with 30 days/month,
/// 12 months/year. Each month has a distinct shape and scale. For each day,
/// with probability `dry_prob` the value is set to 0.0; otherwise it is
/// sampled from the month-specific Gamma distribution.
///
/// Returns `(precip, months, years)` with months 1..=12 and years 1..=n_years.
fn synthetic_precip(n_years: usize, dry_prob: f64, seed: u64) -> (Vec<f64>, Vec<u8>, Vec<u32>) {
    use rand::Rng;

    let days_per_month = 30;
    let total = n_years * 12 * days_per_month;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut precip = Vec::with_capacity(total);
    let mut months = Vec::with_capacity(total);
    let mut years = Vec::with_capacity(total);

    for y in 1..=n_years {
        for m in 1u8..=12 {
            let shape = 1.0 + m as f64 * 0.3;
            let scale = 2.0 + m as f64 * 0.2;
            let dist = GammaDist::new(shape, scale).expect("valid gamma params");
            for _ in 0..days_per_month {
                let val = if rng.random_bool(dry_prob) {
                    0.0
                } else {
                    dist.sample(&mut rng)
                };
                precip.push(val);
                months.push(m);
                years.push(y as u32);
            }
        }
    }

    (precip, months, years)
}

/// Convenience wrapper: creates `ScenarioFactors` where every year and month
/// has the same `factor`.
fn uniform_factors(n_years: usize, factor: f64) -> ScenarioFactors {
    ScenarioFactors::uniform(n_years, [factor; 12])
}

// ---------------------------------------------------------------------------
// 1. year_contiguity_error
// ---------------------------------------------------------------------------
#[test]
fn year_contiguity_error() {
    let precip = vec![1.0, 2.0];
    let month = vec![1u8, 1];
    let year = vec![1u32, 3]; // gap at 2
    let factors = ScenarioFactors::uniform(2, [1.0; 12]);
    let config = QmConfig::new();

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config);
    assert!(
        matches!(result, Err(QuantileMapError::NonContiguousYears { .. })),
        "expected NonContiguousYears error, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. identity_round_trip
// ---------------------------------------------------------------------------
#[test]
fn identity_round_trip() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 42);
    let factors = uniform_factors(10, 1.0);
    let config = QmConfig::new()
        .with_enforce_target_mean(false)
        .with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();

    // Wet-day MAE should be negligible.
    let mut wet_abs_errors = Vec::new();
    for (i, (&orig, &adj)) in precip.iter().zip(adjusted.iter()).enumerate() {
        if orig == 0.0 {
            assert!(
                adj == 0.0,
                "dry day at index {i} should be bit-identical 0.0, got {adj}"
            );
        } else if !orig.is_nan() && orig > 0.0 {
            wet_abs_errors.push((adj - orig).abs());
        }
    }

    let mae: f64 = wet_abs_errors.iter().sum::<f64>() / wet_abs_errors.len() as f64;
    assert!(
        mae < 1e-6,
        "identity mapping should produce near-zero MAE, got {mae}"
    );
}

// ---------------------------------------------------------------------------
// 3. mean_scaling_factor_07
// ---------------------------------------------------------------------------
#[test]
fn mean_scaling_factor_07() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 123);
    let mean_factors = uniform_factors(10, 0.7);
    let var_factors = uniform_factors(10, 1.0);
    let config = QmConfig::new()
        .with_enforce_target_mean(true)
        .with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &mean_factors, &var_factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();

    // Compute wet-day means for input and output.
    let wet_input: Vec<f64> = precip
        .iter()
        .copied()
        .filter(|&x| !x.is_nan() && x > 0.0)
        .collect();
    let wet_output: Vec<f64> = adjusted
        .iter()
        .copied()
        .zip(precip.iter().copied())
        .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
        .map(|(a, _)| a)
        .collect();

    let mean_in = wet_input.iter().sum::<f64>() / wet_input.len() as f64;
    let mean_out = wet_output.iter().sum::<f64>() / wet_output.len() as f64;
    let ratio = mean_out / mean_in;

    assert!(
        (ratio - 0.7).abs() < 0.05,
        "wet-day mean ratio should be ~0.7, got {ratio:.4}"
    );
}

// ---------------------------------------------------------------------------
// 4. variance_scaling_factor_16
// ---------------------------------------------------------------------------
#[test]
fn variance_scaling_factor_16() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 456);
    let mean_factors = uniform_factors(10, 1.0);
    let var_factors = uniform_factors(10, 1.6);
    let config = QmConfig::new()
        .with_scale_var_with_mean(false)
        .with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &mean_factors, &var_factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();

    // Compute wet-day variance for input and output.
    let wet_input: Vec<f64> = precip
        .iter()
        .copied()
        .filter(|&x| !x.is_nan() && x > 0.0)
        .collect();
    let wet_output: Vec<f64> = adjusted
        .iter()
        .copied()
        .zip(precip.iter().copied())
        .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
        .map(|(a, _)| a)
        .collect();

    let var_in = sample_variance(&wet_input);
    let var_out = sample_variance(&wet_output);

    assert!(
        var_out > var_in * 1.2,
        "output variance ({var_out:.2}) should be > 1.2 * input variance ({:.2})",
        var_in * 1.2
    );
}

/// Compute sample variance (unbiased, Bessel-corrected).
fn sample_variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

// ---------------------------------------------------------------------------
// 5. scale_var_with_mean_interaction
// ---------------------------------------------------------------------------
#[test]
fn scale_var_with_mean_interaction() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 789);
    let mean_factors = uniform_factors(10, 1.5);
    let var_factors = uniform_factors(10, 1.0);

    let config_true = QmConfig::new()
        .with_scale_var_with_mean(true)
        .with_enforce_target_mean(false)
        .with_min_events(5);
    let config_false = QmConfig::new()
        .with_scale_var_with_mean(false)
        .with_enforce_target_mean(false)
        .with_min_events(5);

    let result_true = adjust_precipitation(
        &precip,
        &month,
        &year,
        &mean_factors,
        &var_factors,
        &config_true,
    )
    .expect("adjust_precipitation (true) should succeed");
    let result_false = adjust_precipitation(
        &precip,
        &month,
        &year,
        &mean_factors,
        &var_factors,
        &config_false,
    )
    .expect("adjust_precipitation (false) should succeed");

    let wet_true: Vec<f64> = result_true
        .adjusted()
        .iter()
        .copied()
        .zip(precip.iter().copied())
        .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
        .map(|(a, _)| a)
        .collect();
    let wet_false: Vec<f64> = result_false
        .adjusted()
        .iter()
        .copied()
        .zip(precip.iter().copied())
        .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
        .map(|(a, _)| a)
        .collect();

    let var_true = sample_variance(&wet_true);
    let var_false = sample_variance(&wet_false);

    assert!(
        var_true > var_false,
        "scale_var_with_mean=true should yield higher variance ({var_true:.2}) \
         than false ({var_false:.2})"
    );
}

// ---------------------------------------------------------------------------
// 6. tail_amplification_and_mean_enforcement
// ---------------------------------------------------------------------------
#[test]
fn tail_amplification_and_mean_enforcement() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 555);
    let mean_factors = uniform_factors(10, 1.0);
    let var_factors = uniform_factors(10, 1.0);
    let config = QmConfig::new()
        .with_exaggerate_extremes(true)
        .with_extreme_prob_threshold(0.95)
        .with_extreme_k(1.5)
        .with_enforce_target_mean(true)
        .with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &mean_factors, &var_factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();

    // Collect wet-day values for input and output.
    let wet_input: Vec<f64> = precip
        .iter()
        .copied()
        .filter(|&x| !x.is_nan() && x > 0.0)
        .collect();
    let wet_output: Vec<f64> = adjusted
        .iter()
        .copied()
        .zip(precip.iter().copied())
        .filter(|&(_a, p)| !p.is_nan() && p > 0.0)
        .map(|(a, _)| a)
        .collect();

    // Compute P99.
    let p99_in = percentile(&wet_input, 0.99);
    let p99_out = percentile(&wet_output, 0.99);

    assert!(
        p99_out > p99_in,
        "tail amplification should increase P99: output P99={p99_out:.2} should be > input P99={p99_in:.2}"
    );

    // Mean enforcement should bring the overall wet-day mean ratio close to 1.0.
    let mean_in = wet_input.iter().sum::<f64>() / wet_input.len() as f64;
    let mean_out = wet_output.iter().sum::<f64>() / wet_output.len() as f64;
    let ratio = mean_out / mean_in;

    assert!(
        (ratio - 1.0).abs() < 0.05,
        "mean enforcement should keep ratio ~1.0, got {ratio:.4}"
    );
}

/// Compute the percentile of a slice (e.g., 0.99 for P99).
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("no NaN expected in percentile input")
    });
    let idx = ((sorted.len() as f64) * p) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// 7. threshold_preservation
// ---------------------------------------------------------------------------
#[test]
fn threshold_preservation() {
    let (precip, month, year) = synthetic_precip(5, 0.3, 111);
    let factors = uniform_factors(5, 1.0);
    let config = QmConfig::new()
        .with_intensity_threshold(1.0)
        .with_enforce_target_mean(false)
        .with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();

    // Values <= threshold in input should be bit-identical in output.
    let mut input_wet_count = 0usize;
    let mut output_wet_count = 0usize;
    for (i, (&orig, &adj)) in precip.iter().zip(adjusted.iter()).enumerate() {
        if orig.is_nan() {
            continue;
        }
        if orig <= 1.0 {
            assert!(
                adj == orig,
                "sub-threshold value at index {i} should be bit-identical: expected {orig}, got {adj}"
            );
        }
        if orig > 1.0 {
            input_wet_count += 1;
        }
        if adj > 1.0 {
            output_wet_count += 1;
        }
    }

    assert_eq!(
        output_wet_count, input_wet_count,
        "wet-day count (>threshold) should be preserved: input={input_wet_count}, output={output_wet_count}"
    );
}

// ---------------------------------------------------------------------------
// 8. insufficient_data_skipped_months
// ---------------------------------------------------------------------------
#[test]
fn insufficient_data_skipped_months() {
    // Construct data for 1 year. Month 3 has only 2 wet days (below default
    // min_events=10). Other months have 30 days of decent Gamma data.
    let mut rng = StdRng::seed_from_u64(888);

    let mut precip = Vec::new();
    let mut month = Vec::new();
    let mut year = Vec::new();

    for m in 1u8..=12 {
        if m == 3 {
            // 2 wet days plus 28 dry days.
            precip.push(3.0);
            month.push(3u8);
            year.push(1u32);
            precip.push(5.0);
            month.push(3u8);
            year.push(1u32);
            for _ in 0..28 {
                precip.push(0.0);
                month.push(3u8);
                year.push(1u32);
            }
        } else {
            let shape = 1.0 + m as f64 * 0.3;
            let scale = 2.0 + m as f64 * 0.2;
            let dist = GammaDist::new(shape, scale).expect("valid gamma params");
            for _ in 0..30 {
                precip.push(dist.sample(&mut rng));
                month.push(m);
                year.push(1u32);
            }
        }
    }

    let factors = uniform_factors(1, 1.0);
    let config = QmConfig::new(); // default min_events=10

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    // Month 3 should be skipped.
    assert!(
        result.skipped_months().contains(&3),
        "month 3 should be in skipped_months, got {:?}",
        result.skipped_months()
    );

    // Month 3 values should pass through unchanged.
    for (i, (&m, (&orig, &adj))) in month
        .iter()
        .zip(precip.iter().zip(result.adjusted().iter()))
        .enumerate()
    {
        if m == 3 {
            assert!(
                adj == orig,
                "skipped month 3, index {i}: expected {orig}, got {adj}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 9. nan_pass_through
// ---------------------------------------------------------------------------
#[test]
fn nan_pass_through() {
    let (mut precip, month, year) = synthetic_precip(5, 0.3, 222);
    let nan_positions = [0, 50, 100, 500, 1000];
    for &pos in &nan_positions {
        if pos < precip.len() {
            precip[pos] = f64::NAN;
        }
    }

    let factors = uniform_factors(5, 1.0);
    let config = QmConfig::new().with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    let adjusted = result.adjusted();
    for &pos in &nan_positions {
        if pos < adjusted.len() {
            assert!(
                adjusted[pos].is_nan(),
                "NaN at position {pos} should remain NaN, got {}",
                adjusted[pos]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 10. output_length_matches_input
// ---------------------------------------------------------------------------
#[test]
fn output_length_matches_input() {
    let (precip, month, year) = synthetic_precip(5, 0.3, 333);
    let factors = uniform_factors(5, 1.0);
    let config = QmConfig::new().with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    assert_eq!(
        result.adjusted().len(),
        precip.len(),
        "output length must match input length"
    );
}

// ---------------------------------------------------------------------------
// 11. all_adjusted_values_finite_or_nan
// ---------------------------------------------------------------------------
#[test]
fn all_adjusted_values_finite_or_nan() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 444);
    let mean_factors = uniform_factors(10, 0.8);
    let var_factors = uniform_factors(10, 1.0);
    let config = QmConfig::new().with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &mean_factors, &var_factors, &config)
        .expect("adjust_precipitation should succeed");

    for (i, &v) in result.adjusted().iter().enumerate() {
        assert!(
            v.is_finite() || v.is_nan(),
            "value at index {i} is neither finite nor NaN: {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// 12. no_negative_adjusted_values
// ---------------------------------------------------------------------------
#[test]
fn no_negative_adjusted_values() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 444);
    let mean_factors = uniform_factors(10, 0.8);
    let var_factors = uniform_factors(10, 1.0);
    let config = QmConfig::new().with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &mean_factors, &var_factors, &config)
        .expect("adjust_precipitation should succeed");

    for (i, &v) in result.adjusted().iter().enumerate() {
        if !v.is_nan() {
            assert!(v >= 0.0, "negative value at index {i}: {v}");
        }
    }
}

// ---------------------------------------------------------------------------
// 13. factor_dimension_mismatch_error
// ---------------------------------------------------------------------------
#[test]
fn factor_dimension_mismatch_error() {
    let (precip, month, year) = synthetic_precip(5, 0.3, 999);
    let factors_3 = uniform_factors(3, 1.0); // 3 years, but data has 5
    let factors_5 = uniform_factors(5, 1.0);
    let config = QmConfig::new();

    // mean_factors have wrong dimension
    let result = adjust_precipitation(&precip, &month, &year, &factors_3, &factors_5, &config);
    assert!(
        matches!(result, Err(QuantileMapError::FactorYearMismatch { .. })),
        "expected FactorYearMismatch error, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 14. empty_input_error
// ---------------------------------------------------------------------------
#[test]
fn empty_input_error() {
    let factors = uniform_factors(1, 1.0);
    let config = QmConfig::new();

    let result = adjust_precipitation(&[], &[], &[], &factors, &factors, &config);
    assert!(
        matches!(result, Err(QuantileMapError::EmptyData)),
        "expected EmptyData error, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 15. perturbed_months_populated
// ---------------------------------------------------------------------------
#[test]
fn perturbed_months_populated() {
    let (precip, month, year) = synthetic_precip(10, 0.3, 666);
    let factors = uniform_factors(10, 1.0);
    let config = QmConfig::new().with_min_events(5);

    let result = adjust_precipitation(&precip, &month, &year, &factors, &factors, &config)
        .expect("adjust_precipitation should succeed");

    let perturbed = result.perturbed_months();

    // With 10 years of 30 days/month and dry_prob=0.3, each month should have
    // ~210 wet days, well above min_events=5. All 12 months should appear.
    for m in 1u8..=12 {
        assert!(
            perturbed.contains(&m),
            "month {m} should appear in perturbed_months, got {perturbed:?}"
        );
    }
}
