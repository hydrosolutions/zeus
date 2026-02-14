# Zeus

Pure-Rust reimplementation of the [weathergenr](https://github.com/) semiparametric, multivariate, multisite stochastic weather generator (Steinschneider & Brown, 2013).

Zeus couples **wavelet-based low-frequency modeling** (WARM) with **daily Markov-chain / KNN resampling** to synthesize realistic climate sequences that preserve multi-scale variability, spatial coherence, and extreme-event characteristics.

> **Maintenance:** 🟢 Active · `tool` · `analysis`

## Architecture

```text
Observed Climate Data (NetCDF)
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
   Synthetic Weather (Parquet)
```

## Usage

```sh
# Generate synthetic weather from observed data
zeus generate -c config.toml -s 42

# Apply climate perturbations
zeus perturb -i output/syn.parquet -o output/future.parquet --temp-delta 2.0

# Evaluate synthetic output against observations
zeus evaluate -c config.toml --synthetic output/syn.parquet

# Increase verbosity (-v info, -vv debug, -vvv trace)
zeus -vv generate -c config.toml
```

The generator reads observed climate data from NetCDF, runs the WARM simulation and daily resampling pipeline, and writes synthetic weather to Parquet with inline evaluation diagnostics. Climate perturbations are applied separately via `zeus perturb`, enabling grid sweeps over hundreds of scenarios without re-running the expensive generation step.

Configuration is driven by a TOML file with sections for `[io]`, `[warm]`, `[filter]`, `[resample]`, `[markov]`, and `[evaluate]`. All sections have sensible defaults — a minimal config only needs input/output paths:

```toml
seed = 42

[io]
input = "data/observed.nc"
output = "output/synthetic.parquet"
```

## Workspace

| Crate | Description |
|-------|-------------|
| **zeus-arma** | ARMA(p,q) via exact MLE / Kalman filter, BFGS optimizer, AIC selection |
| **zeus-wavelet** | MODWT/MRA decomposition, Morlet CWT, significance testing |
| **zeus-warm** | WARM pipeline — wavelet-ARMA simulation with adaptive pool filtering |
| **zeus-markov** | Three-state precipitation Markov chain with monthly transitions |
| **zeus-knn** | k-nearest-neighbor sampling (uniform, rank, distance-weighted) |
| **zeus-resample** | Daily disaggregation via Markov-conditioned KNN |
| **zeus-quantile-map** | Gamma-to-Gamma parametric quantile mapping for precipitation adjustment |
| **zeus-perturb** | Climate perturbation pipeline — temperature scaling, occurrence adjustment, safety rails |
| **zeus-stats** | Centralised statistics — mean, variance, sd, quantile, median, robust scale, correlation |
| **zeus-calendar** | 365-day no-leap calendar, water-year assignment |
| **zeus-evaluate** | Simulation vs. observation diagnostics — timeseries stats, correlations, MAE scorecard |
| **zeus-io** | NetCDF reader and Parquet writer for climate data |

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
