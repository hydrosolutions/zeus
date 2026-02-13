# Zeus

Pure-Rust reimplementation of the [weathergenr](https://github.com/) semiparametric, multivariate, multisite stochastic weather generator (Steinschneider & Brown, 2013).

Zeus couples **wavelet-based low-frequency modeling** (WARM) with **daily Markov-chain / KNN resampling** to synthesize realistic climate sequences that preserve multi-scale variability, spatial coherence, and extreme-event characteristics.

> **Maintenance:** 🟢 Active · `tool` · `analysis`

## Architecture

```text
Observed Climate Data
        │
        ▼
┌──────────────────────────────────┐
│  WARM Pipeline (zeus-warm)       │
│   ├─ MRA decomposition (wavelet) │
│   ├─ ARMA fit per component      │
│   ├─ Simulate annual realizations│
│   └─ Filter by moment / spectral │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Daily Disaggregation            │
│   ├─ 3-state Markov (markov)     │
│   ├─ Annual KNN matching (knn)   │
│   └─ Daily KNN resampling        │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Post-Processing                 │
│   ├─ Quantile mapping (quantile) │
│   ├─ Climate perturbations       │
│   └─ Evaluation diagnostics      │
└──────────────────────────────────┘
        │
        ▼
   Synthetic Weather (NetCDF/CSV)
```

## Workspace

| Crate | Status | Description |
|-------|--------|-------------|
| **zeus-arma** | Done | ARMA(p,q) via exact MLE / Kalman filter, BFGS optimizer, AIC selection |
| **zeus-wavelet** | Done | MODWT/MRA decomposition, Morlet CWT, significance testing |
| **zeus-warm** | Done | WARM pipeline — wavelet-ARMA simulation with adaptive pool filtering |
| **zeus-markov** | Done | Three-state precipitation Markov chain with monthly transitions |
| **zeus-knn** | Done | k-nearest-neighbor sampling (uniform, rank, distance-weighted) |
| **zeus-resample** | Done | Daily disaggregation via Markov-conditioned KNN |
| **zeus-quantile-map** | Done | Gamma-to-Gamma parametric quantile mapping for precipitation adjustment |
| **zeus-perturb** | Done | Climate perturbation pipeline — temperature scaling, occurrence adjustment, safety rails |
| **zeus-stats** | Done | Centralised statistics — mean, variance, sd, quantile, median, robust scale, correlation |
| **zeus-calendar** | Done | 365-day no-leap calendar, water-year assignment |
| zeus-pet | Scaffold | Hargreaves potential evapotranspiration |
| zeus-evaluate | Scaffold | Simulation vs. observation diagnostics |
| zeus-io | Scaffold | NetCDF / CSV I/O |

## Build & Test

```sh
cargo build            # build all crates
cargo test --all       # run all tests
cargo clippy --all --all-targets -- -D warnings
cargo fmt --all
```

## References

- Steinschneider, S., & Brown, C. (2013). *A semiparametric multivariate, multisite weather generator with low-frequency variability for use in climate risk assessments.* Water Resources Research, 49(11), 7205–7220.
- Torrence, C., & Compo, G. P. (1998). *A practical guide to wavelet analysis.* Bulletin of the American Meteorological Society, 79(1), 61–78.
- Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time Series Analysis.* Cambridge University Press.

## License

All rights reserved.
