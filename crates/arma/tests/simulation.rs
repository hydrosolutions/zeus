//! Simulation integration tests for zeus-arma.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use zeus_arma::ArmaSpec;

#[test]
fn simulate_then_fit_round_trip() {
    // Fit an AR(1) model, simulate from it, then re-fit
    let phi = 0.6;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut data = vec![0.0; 1000];
    for t in 1..1000 {
        data[t] = phi * data[t - 1] + normal.sample(&mut rng);
    }

    let fit = ArmaSpec::new(1, 0).fit(&data).unwrap();

    // Simulate from fitted model
    let mut sim_rng = rand::rngs::StdRng::seed_from_u64(123);
    let simulated = fit.simulate(1000, 1, &mut sim_rng);

    // Re-fit from simulated data
    let sim_data: Vec<f64> = simulated.column(0).to_vec();
    let refit = ArmaSpec::new(1, 0).fit(&sim_data).unwrap();

    // The re-fitted coefficient should be reasonably close to original
    assert!(
        (refit.ar()[0] - fit.ar()[0]).abs() < 0.3,
        "Simulate-fit round trip: original phi={}, refit phi={}",
        fit.ar()[0],
        refit.ar()[0]
    );
}

#[test]
fn acf_structure_preserved() {
    // Fit MA(1), simulate, check that lag-1 autocorrelation is in the right ballpark
    let theta = 0.6;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = 2000;
    let mut data = vec![0.0; n];
    let mut eps = vec![0.0; n];
    for t in 0..n {
        eps[t] = normal.sample(&mut rng);
        data[t] = eps[t] + if t > 0 { theta * eps[t - 1] } else { 0.0 };
    }

    let fit = ArmaSpec::new(0, 1).fit(&data).unwrap();
    let mut sim_rng = rand::rngs::StdRng::seed_from_u64(456);
    let simulated = fit.simulate(5000, 1, &mut sim_rng);
    let col = simulated.column(0);

    // Compute lag-1 autocorrelation
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
    let cov: f64 = col
        .iter()
        .skip(1)
        .zip(col.iter())
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / col.len() as f64;
    let acf1 = cov / var;

    // Theoretical lag-1 ACF for MA(1): theta / (1 + theta^2)
    let theoretical_acf1 = theta / (1.0 + theta * theta);
    assert!(
        (acf1 - theoretical_acf1).abs() < 0.15,
        "ACF(1): expected ~{:.3}, got {:.3}",
        theoretical_acf1,
        acf1
    );
}
