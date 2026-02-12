use rand::SeedableRng;
use rand::rngs::StdRng;
use zeus_markov::{
    MarkovConfig, MonthlyTransitions, PrecipState, StateThresholds, estimate_monthly_transitions,
    simulate_states, simulate_states_into,
};

/// Generate synthetic precipitation data for `n_days` days.
///
/// Returns `(precip, months)` where months cycle 1..=12 with ~30 days each.
/// Precipitation values are a mix of zeros and positive values drawn from a
/// seeded RNG to create realistic-ish data.
fn synthetic_data(n_days: usize, seed: u64) -> (Vec<f64>, Vec<u8>) {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut precip = Vec::with_capacity(n_days);
    let mut months = Vec::with_capacity(n_days);

    for i in 0..n_days {
        // Cycle through months: each month gets ~30 days.
        let month = ((i / 30) % 12) as u8 + 1;
        months.push(month);

        // ~40% chance of being dry (zero), otherwise positive value up to ~20 mm.
        let val: f64 = if rng.random_bool(0.4) {
            0.0
        } else {
            rng.random_range(0.01..20.0)
        };
        precip.push(val);
    }
    (precip, months)
}

/// Run the full threshold + transition pipeline and return the transitions.
fn run_pipeline(
    precip: &[f64],
    months: &[u8],
    config: &MarkovConfig,
) -> (StateThresholds, MonthlyTransitions) {
    let thresholds =
        StateThresholds::from_baseline(precip, months, config).expect("from_baseline failed");
    let transitions = estimate_monthly_transitions(precip, months, &thresholds, config)
        .expect("estimate_monthly_transitions failed");
    (thresholds, transitions)
}

// ---------------------------------------------------------------------------
// 1. full_pipeline_smoke
// ---------------------------------------------------------------------------
#[test]
fn full_pipeline_smoke() {
    let config = MarkovConfig::new();
    let (precip, months) = synthetic_data(3650, 1);

    let (_thresholds, transitions) = run_pipeline(&precip, &months, &config);

    // Simulate 365 days of state output.
    let sim_months: Vec<u8> = (0..365).map(|i| (i % 12) as u8 + 1).collect();
    let mut rng = StdRng::seed_from_u64(99);
    let result = simulate_states(&transitions, &sim_months, PrecipState::Dry, &mut rng);

    // Output length must match sim_months length.
    assert_eq!(
        result.len(),
        sim_months.len(),
        "output length must match sim_months length"
    );
    // Vector must not be empty.
    assert!(!result.is_empty(), "simulated output must not be empty");
}

// ---------------------------------------------------------------------------
// 2. deterministic_with_seed
// ---------------------------------------------------------------------------
#[test]
fn deterministic_with_seed() {
    let config = MarkovConfig::new();
    let (precip, months) = synthetic_data(3650, 2);
    let (_thresholds, transitions) = run_pipeline(&precip, &months, &config);

    let sim_months: Vec<u8> = (0..500).map(|i| (i % 12) as u8 + 1).collect();

    let mut rng1 = StdRng::seed_from_u64(42);
    let result1 = simulate_states(&transitions, &sim_months, PrecipState::Dry, &mut rng1);

    let mut rng2 = StdRng::seed_from_u64(42);
    let result2 = simulate_states(&transitions, &sim_months, PrecipState::Dry, &mut rng2);

    assert_eq!(result1, result2, "same seed must produce identical output");
}

// ---------------------------------------------------------------------------
// 3. simulate_into_matches_allocating
// ---------------------------------------------------------------------------
#[test]
fn simulate_into_matches_allocating() {
    let config = MarkovConfig::new();
    let (precip, months) = synthetic_data(3650, 3);
    let (_thresholds, transitions) = run_pipeline(&precip, &months, &config);

    let sim_months: Vec<u8> = (0..500).map(|i| (i % 12) as u8 + 1).collect();

    let mut rng1 = StdRng::seed_from_u64(77);
    let allocating = simulate_states(&transitions, &sim_months, PrecipState::Dry, &mut rng1);

    let mut rng2 = StdRng::seed_from_u64(77);
    let mut buf = vec![PrecipState::Dry; sim_months.len()];
    simulate_states_into(
        &transitions,
        &sim_months,
        PrecipState::Dry,
        &mut rng2,
        &mut buf,
    )
    .expect("simulate_states_into failed");

    assert_eq!(
        allocating, buf,
        "simulate_states and simulate_states_into must produce identical output"
    );
}

// ---------------------------------------------------------------------------
// 4. state_frequencies_plausible
// ---------------------------------------------------------------------------
#[test]
fn state_frequencies_plausible() {
    let config = MarkovConfig::new();
    let (precip, months) = synthetic_data(5000, 4);
    let (_thresholds, transitions) = run_pipeline(&precip, &months, &config);

    // Simulate 50,000 steps.
    let sim_months: Vec<u8> = (0..50_000).map(|i| (i % 12) as u8 + 1).collect();
    let mut rng = StdRng::seed_from_u64(12345);
    let result = simulate_states(&transitions, &sim_months, PrecipState::Dry, &mut rng);

    let mut counts = [0usize; 3];
    for s in &result {
        counts[s.as_index()] += 1;
    }

    let total = result.len() as f64;

    // All three states must appear at least once.
    assert!(counts[0] > 0, "Dry count must be > 0");
    assert!(counts[1] > 0, "Wet count must be > 0");
    assert!(counts[2] > 0, "Extreme count must be > 0");

    // No single state should account for more than 99% of the total.
    for (i, &c) in counts.iter().enumerate() {
        let frac = c as f64 / total;
        assert!(
            frac < 0.99,
            "state {} accounts for {:.2}% of total, which is degenerate",
            i,
            frac * 100.0
        );
    }
}

// ---------------------------------------------------------------------------
// 5. conditional_dataset_increases_dry_persistence
// ---------------------------------------------------------------------------
#[test]
fn conditional_dataset_increases_dry_persistence() {
    let config = MarkovConfig::new();
    let (precip, months) = synthetic_data(3650, 5);

    // --- Full baseline ---
    let (thresholds_full, transitions_full) = run_pipeline(&precip, &months, &config);

    // --- Dry-biased subset ---
    // Compute the median of positive precipitation values.
    let positive_precip: Vec<f64> = precip.iter().copied().filter(|&v| v > 0.0).collect();
    assert!(
        !positive_precip.is_empty(),
        "need some positive precip for this test"
    );
    let mut sorted_positive = positive_precip.clone();
    sorted_positive.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_positive = sorted_positive[sorted_positive.len() / 2];

    // Keep only days where precip is zero or below the median of positive values.
    // This removes the wettest portion of the dataset.
    let mut dry_precip = Vec::new();
    let mut dry_months = Vec::new();
    for (i, &p) in precip.iter().enumerate() {
        if p <= median_positive {
            dry_precip.push(p);
            dry_months.push(months[i]);
        }
    }

    assert!(
        dry_precip.len() >= 2,
        "dry-biased subset must have at least 2 observations"
    );

    // Estimate thresholds and transitions from the dry-biased subset.
    let thresholds_dry =
        StateThresholds::from_baseline(&dry_precip, &dry_months, &config).expect("dry baseline");
    let transitions_dry =
        estimate_monthly_transitions(&dry_precip, &dry_months, &thresholds_dry, &config)
            .expect("dry transitions");

    // Compare p00 (Dry -> Dry) for January (month = 1).
    let p00_full = transitions_full
        .for_month(1)
        .prob(PrecipState::Dry, PrecipState::Dry);
    let p00_dry = transitions_dry
        .for_month(1)
        .prob(PrecipState::Dry, PrecipState::Dry);

    // Ignore thresholds_full to avoid an unused-variable warning; we only needed it for the
    // transitions but let's keep the binding name clear.
    let _ = &thresholds_full;

    assert!(
        p00_dry >= p00_full,
        "dry-biased subset should have equal or higher Dry->Dry persistence \
         (p00_dry={p00_dry:.4}, p00_full={p00_full:.4})"
    );
}
