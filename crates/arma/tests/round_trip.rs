//! Round-trip integration tests for zeus-arma.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use zeus_arma::{ArmaSpec, select_best_aic};

fn generate_ar1(phi: f64, sigma2: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma2.sqrt()).unwrap();
    let mut data = vec![0.0; n];
    for t in 1..n {
        data[t] = phi * data[t - 1] + normal.sample(&mut rng);
    }
    data
}

fn generate_ma1(theta: f64, sigma2: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma2.sqrt()).unwrap();
    let mut data = vec![0.0; n];
    let mut eps = vec![0.0; n];
    for t in 0..n {
        eps[t] = normal.sample(&mut rng);
        data[t] = eps[t] + if t > 0 { theta * eps[t - 1] } else { 0.0 };
    }
    data
}

fn generate_arma11(phi: f64, theta: f64, sigma2: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, sigma2.sqrt()).unwrap();
    let mut data = vec![0.0; n];
    let mut eps = vec![0.0; n];
    for t in 0..n {
        eps[t] = normal.sample(&mut rng);
        let ar_part = if t > 0 { phi * data[t - 1] } else { 0.0 };
        let ma_part = if t > 0 { theta * eps[t - 1] } else { 0.0 };
        data[t] = ar_part + eps[t] + ma_part;
    }
    data
}

#[test]
fn white_noise_round_trip() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();

    let fit = ArmaSpec::new(0, 0).fit(&data).unwrap();
    assert_eq!(fit.order(), (0, 0));
    assert!(fit.sigma2() > 0.5 && fit.sigma2() < 1.5);
    assert!(fit.log_likelihood().is_finite());
}

#[test]
fn ar1_recovery() {
    let phi = 0.7;
    let data = generate_ar1(phi, 1.0, 2000, 100);
    let fit = ArmaSpec::new(1, 0).fit(&data).unwrap();
    assert!(
        (fit.ar()[0] - phi).abs() < 0.15,
        "AR(1) phi: expected ~{}, got {}",
        phi,
        fit.ar()[0]
    );
    assert!(fit.sigma2() > 0.5 && fit.sigma2() < 1.5);
}

#[test]
fn ma1_recovery() {
    let theta = 0.5;
    let data = generate_ma1(theta, 1.0, 2000, 200);
    let fit = ArmaSpec::new(0, 1).fit(&data).unwrap();
    assert!(
        (fit.ma()[0] - theta).abs() < 0.15,
        "MA(1) theta: expected ~{}, got {}",
        theta,
        fit.ma()[0]
    );
}

#[test]
fn arma11_recovery() {
    let phi = 0.5;
    let theta = 0.3;
    let data = generate_arma11(phi, theta, 1.0, 2000, 300);
    let fit = ArmaSpec::new(1, 1).fit(&data).unwrap();
    assert!(
        (fit.ar()[0] - phi).abs() < 0.15,
        "ARMA(1,1) phi: expected ~{}, got {}",
        phi,
        fit.ar()[0]
    );
    assert!(
        (fit.ma()[0] - theta).abs() < 0.15,
        "ARMA(1,1) theta: expected ~{}, got {}",
        theta,
        fit.ma()[0]
    );
}

#[test]
fn aic_selects_correct_order() {
    // Generate AR(1) data, then AIC search should prefer (1,0) or nearby
    let data = generate_ar1(0.7, 1.0, 1000, 400);
    let best = select_best_aic(&data, 2, 2).unwrap();
    let (p, q) = best.order();
    // The AR(1) model should be preferred, but ARMA(1,1) is also acceptable
    assert!(
        p == 1 && (q == 0 || q == 1),
        "Expected AR(1) or ARMA(1,1) to be selected, got ({}, {})",
        p,
        q
    );
}
