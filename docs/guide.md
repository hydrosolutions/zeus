# Zeus User Guide

Zeus is a stochastic weather generator written in pure Rust. It reads observed climate data from NetCDF files, simulates synthetic weather sequences using a Wavelet-ARMA (WARM) pipeline, and writes output to Parquet. An optional perturbation stage applies climate-change scenarios to the synthetic data.

## Quick Start

```bash
# Minimal run — uses zeus.toml in the current directory
zeus generate

# Specify a config file and increase verbosity
zeus generate -c my_project.toml -vv

# Override I/O paths from the command line
zeus generate -i data/obs.nc -o output/syn.parquet -s 42
```

Minimal config (`zeus.toml`):

```toml
seed = 42

[io]
input = "data/observed.nc"
output = "output/synthetic.parquet"
```

All other sections use sensible defaults. Add sections only when you need to override them.

---

## CLI Reference

### Global Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--config <PATH>` | `-c` | Path to TOML config file | `zeus.toml` |
| `--input <PATH>` | `-i` | Override `io.input` from config | — |
| `--output <PATH>` | `-o` | Override `io.output` from config | — |
| `--seed <SEED>` | `-s` | Override `seed` from config | — |
| `-v` | — | Verbosity (stackable: `-v` info, `-vv` debug, `-vvv` trace) | warn |

### `zeus generate`

Runs the full pipeline:

1. Read observed climate data from NetCDF.
2. Per-site WARM simulation (annual precipitation → wavelet MRA → ARMA fit → stochastic simulation).
3. Filter the simulation pool down to the best realisations.
4. Daily disaggregation via Markov-chain occurrence modelling and KNN resampling.
5. Optional climate perturbations (temperature shifts, quantile mapping, occurrence adjustment).
6. Write synthetic weather to Parquet.
7. Write evaluation diagnostics to a JSON sidecar (`<output_stem>.diagnostics.json`).

```bash
zeus generate
zeus generate -c project.toml -vv
zeus generate -i obs.nc -o syn.parquet -s 12345
```

### `zeus evaluate`

Compare synthetic output against observations. Reads the observed NetCDF from the config and a synthetic Parquet file.

```bash
zeus evaluate --synthetic output/synthetic.parquet
```

> **Note:** The `evaluate` subcommand is a V1 placeholder. Full standalone evaluation will be available once the Parquet reader is implemented. The inline evaluation that runs automatically at the end of `generate` is fully functional.

---

## Configuration Reference

The config file is TOML. Every section except `[io]` is optional — omit a section entirely to use its defaults. All sections enforce `deny_unknown_fields`, so typos in key names produce clear errors.

### Top-Level

```toml
seed = 42  # optional — omit for OS-random entropy
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `seed` | integer | no | OS random | Global RNG seed for reproducibility. |

---

### `[io]` — Input / Output

Controls NetCDF reading and Parquet writing.

```toml
[io]
input = "data/observed.nc"
output = "output/synthetic.parquet"
precip_var = "pr"
temp_max_var = "tasmax"
temp_min_var = "tasmin"
start_month = 10
trim_to_water_years = true
compression = "snappy"
row_group_size = 1000000
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `input` | string | yes* | — | Path to observed NetCDF file. Can be overridden with `-i`. |
| `output` | string | yes* | — | Path for synthetic Parquet output. Can be overridden with `-o`. |
| `precip_var` | string | no | `"pr"` | NetCDF variable name for precipitation. |
| `temp_max_var` | string or null | no | `"tasmax"` | NetCDF variable for Tmax. Set to `null` to skip. |
| `temp_min_var` | string or null | no | `"tasmin"` | NetCDF variable for Tmin. Set to `null` to skip. |
| `start_month` | integer (1–12) | no | `10` | First month of the water year. |
| `trim_to_water_years` | boolean | no | `true` | Trim input series to complete water years. |
| `compression` | string | no | `"snappy"` | Parquet compression: `"none"`, `"snappy"`, or `"zstd"`. |
| `row_group_size` | integer | no | `1000000` | Parquet row group size. |

*`input` and `output` must be provided either in the config file or via CLI flags.

---

### `[warm]` — WARM Simulation

Controls the Wavelet-ARMA simulation engine that generates synthetic annual precipitation series.

```toml
[warm]
n_sim = 1000
n_years = 50
wavelet_filter = "la8"
# mra_levels = <auto>
bypass_n = 30
max_arma_order = [5, 3]
match_variance = true
var_tol = 0.1
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `n_sim` | integer | no | `1000` | Number of stochastic simulations. |
| `n_years` | integer | no | `50` | Years per simulation. |
| `wavelet_filter` | string | no | `"la8"` | Wavelet filter. Options: `"haar"`, `"d4"`, `"d6"`, `"d8"`, `"la8"`, `"la16"`. |
| `mra_levels` | integer or null | no | auto | MRA decomposition levels. Auto-computed from series length when omitted. |
| `bypass_n` | integer | no | `30` | Burn-in simulations discarded before collecting results. |
| `max_arma_order` | [integer, integer] | no | `[5, 3]` | Maximum `[p, q]` for ARMA order selection via AIC. |
| `match_variance` | boolean | no | `true` | Apply variance matching to simulated MRA components. |
| `var_tol` | float | no | `0.1` | Variance tolerance as a fraction of observed variance. |

---

### `[filter]` — Pool Filtering

Selects the best realisations from the WARM simulation pool using statistical criteria.

```toml
[filter]
n_select = 5
mean_tol = 0.03
sd_tol = 0.03
tail_low_p = 0.20
tail_high_p = 0.80
spectral_corr_min = 0.60
peak_match_frac_min = 1.0
n_sig_peaks_max = 2
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `n_select` | integer | no | `5` | Number of top realisations to select. |
| `mean_tol` | float | no | `0.03` | Tolerance for mean annual precip (fraction of observed). |
| `sd_tol` | float | no | `0.03` | Tolerance for standard deviation (fraction of observed). |
| `tail_low_p` | float | no | `0.20` | Lower quantile for tail distribution check. |
| `tail_high_p` | float | no | `0.80` | Upper quantile for tail distribution check. |
| `spectral_corr_min` | float | no | `0.60` | Minimum spectral correlation with observed. |
| `peak_match_frac_min` | float | no | `1.0` | Minimum fraction of significant spectral peaks that must match. |
| `n_sig_peaks_max` | integer | no | `2` | Maximum number of significant peaks allowed. |

---

### `[resample]` — Daily Disaggregation

Controls how annual synthetic series are disaggregated into daily weather using KNN resampling.

```toml
[resample]
annual_knn_n = 100
precip_weight = 100.0
temp_weight = 10.0
sd_floor = 0.1
narrow_window = 3
wide_window = 30
sampling = "rank"
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `annual_knn_n` | integer | no | `100` | Pool size for annual-level KNN. |
| `precip_weight` | float | no | `100.0` | Feature weight for precipitation in KNN distance. |
| `temp_weight` | float | no | `10.0` | Feature weight for temperature in KNN distance. |
| `sd_floor` | float | no | `0.1` | Floor for standard deviation when computing feature weights. |
| `narrow_window` | integer | no | `3` | Days ± the target day for the narrow daily KNN window. |
| `wide_window` | integer | no | `30` | Days ± the target day for the wide daily KNN fallback window. |
| `sampling` | string | no | `"rank"` | KNN sampling strategy: `"uniform"`, `"rank"`, or `"gaussian"`. |

---

### `[markov]` — Precipitation Occurrence

Configures the 3-state (dry / wet / extreme) Markov chain that models precipitation occurrence.

```toml
[markov]
dirichlet_alpha = 1.0
dry_spell_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
wet_spell_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

[markov.wet_threshold]
fixed = 0.3

[markov.extreme_threshold]
quantile = 0.8
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `dirichlet_alpha` | float | no | `1.0` | Dirichlet smoothing parameter for transition matrices. |
| `dry_spell_factors` | [float; 12] | no | all `1.0` | Monthly scaling factors for dry-spell persistence. |
| `wet_spell_factors` | [float; 12] | no | all `1.0` | Monthly scaling factors for wet-spell persistence. |

#### Threshold Specification

Both `wet_threshold` and `extreme_threshold` accept exactly one of two forms:

```toml
# Fixed absolute value (mm/day)
[markov.wet_threshold]
fixed = 0.3

# Quantile of observed precipitation distribution
[markov.extreme_threshold]
quantile = 0.8
```

| Threshold | Default | Description |
|-----------|---------|-------------|
| `wet_threshold` | `fixed = 0.3` | Boundary between dry and wet states. |
| `extreme_threshold` | `quantile = 0.8` | Boundary between wet and extreme states. |

---

### `[perturb]` — Climate Perturbations (Optional)

Omit this entire section to skip perturbations. When present, perturbations are applied in this order: temperature → quantile mapping → occurrence → safety rails.

```toml
[perturb]
precip_floor = 0.0
precip_cap = 500.0
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `precip_floor` | float | no | `0.0` | Minimum precipitation after perturbation (mm/day). |
| `precip_cap` | float | no | `500.0` | Maximum precipitation after perturbation (mm/day). |

#### `[perturb.temperature]` — Temperature Shifts

```toml
[perturb.temperature]
deltas = [0.5, 0.6, 0.7, 0.6, 0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7]
transient = false
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `deltas` | [float; 12] | yes | — | Monthly temperature deltas (degrees). Applied to Tmax and Tmin. |
| `transient` | boolean | no | `false` | If true, ramp deltas linearly from 0 to 2×delta over the simulation period. |

#### `[perturb.quantile_map]` — Precipitation Intensity Adjustment

Uses Gamma-to-Gamma quantile mapping to adjust precipitation intensity distributions.

```toml
[perturb.quantile_map]
mean_factors = [0.95, 0.95, 0.95, 0.95, 1.05, 1.05, 1.05, 1.05, 1.0, 1.0, 1.0, 1.0]
var_factors  = [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]
intensity_threshold = 0.1
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `mean_factors` | [float; 12] | yes | — | Monthly scaling factors for mean precipitation intensity. |
| `var_factors` | [float; 12] | yes | — | Monthly scaling factors for precipitation variance. |
| `intensity_threshold` | float | no | `0.0` | Wet-day threshold for quantile mapping (mm/day). |

#### `[perturb.occurrence]` — Precipitation Occurrence Adjustment

```toml
[perturb.occurrence]
factors = [1.1, 1.1, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0]
transient = false
intensity_threshold = 0.3
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `factors` | [float; 12] | yes | — | Monthly multipliers for wet-day occurrence probability. |
| `transient` | boolean | no | `false` | If true, ramp factors linearly over the simulation period. |
| `intensity_threshold` | float | no | `0.3` | Wet-day threshold for occurrence adjustment (mm/day). |

---

### `[evaluate]` — Evaluation Diagnostics

Controls the diagnostics computed after generation. Results are written to a JSON sidecar file alongside the Parquet output.

```toml
[evaluate]
precip_threshold = 0.01
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `precip_threshold` | float | no | `0.01` | Wet-day threshold for evaluation statistics (mm/day). |

---

## Output Format

### Parquet Schema

The output Parquet file contains one row per simulated day, with columns:

| Column | Type | Description |
|--------|------|-------------|
| `realisation` | UInt32 | Realisation index (0-based). |
| `month` | UInt8 | Calendar month (1–12). |
| `water_year` | Int32 | Water year assignment. |
| `day_of_year` | UInt16 | Day of year (1–365, no-leap calendar). |
| `precip` | Float64 | Precipitation (mm/day). |
| `temp_max` | Float64 | Maximum temperature (present only if `temp_max_var` is set). |
| `temp_min` | Float64 | Minimum temperature (present only if `temp_min_var` is set). |

### Diagnostics JSON

When `generate` completes, a `<output_stem>.diagnostics.json` file is written alongside the Parquet output. It contains per-site comparison metrics between observed and synthetic data including means, variances, wet-day statistics, and cross-realisation consistency scores.

---

## Input Requirements

Zeus expects a CF-convention NetCDF file with:

- A `time` dimension and coordinate variable.
- A precipitation variable (default name: `pr`).
- Optional temperature variables (default names: `tasmax`, `tasmin`).
- One or more site/grid dimensions (multi-site data is processed independently per site).
- A 365-day no-leap calendar. Input data is trimmed to complete water years by default.

---

## Pipeline Architecture

Zeus is composed of 13 specialized crates that execute in sequence:

```
NetCDF input (zeus-io)
  │
  ▼
Annual aggregation (zeus-calendar)
  │
  ▼
Wavelet MRA decomposition (zeus-wavelet)
  │
  ▼
ARMA fitting & simulation per MRA level (zeus-arma)
  │
  ▼
WARM pool assembly & filtering (zeus-warm)
  │
  ▼
Markov-chain occurrence modelling (zeus-markov)
  │
  ▼
KNN daily disaggregation (zeus-knn, zeus-resample)
  │
  ▼
Climate perturbations (zeus-perturb, zeus-quantile-map)
  │
  ▼
Parquet output (zeus-io)
  │
  ▼
Evaluation diagnostics (zeus-evaluate)
```

---

## Examples

### Generate with defaults

```toml
# zeus.toml
seed = 42

[io]
input = "data/observed.nc"
output = "output/synthetic.parquet"
```

```bash
zeus generate
```

### Tighter filtering, more realisations

```toml
seed = 99

[io]
input = "data/obs.nc"
output = "output/syn.parquet"
compression = "zstd"

[warm]
n_sim = 2000
n_years = 30

[filter]
n_select = 10
mean_tol = 0.02
sd_tol = 0.02
spectral_corr_min = 0.70
```

### Climate change scenario

```toml
seed = 7

[io]
input = "data/obs.nc"
output = "output/future.parquet"

[perturb]
precip_cap = 300.0

[perturb.temperature]
deltas = [1.2, 1.3, 1.5, 1.4, 1.0, 0.8, 0.7, 0.7, 0.9, 1.1, 1.2, 1.3]
transient = true

[perturb.quantile_map]
mean_factors = [0.90, 0.90, 0.95, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.90, 0.90]
var_factors  = [1.10, 1.10, 1.05, 1.00, 1.00, 0.95, 0.95, 1.00, 1.05, 1.10, 1.10, 1.10]
intensity_threshold = 0.1

[perturb.occurrence]
factors = [1.05, 1.05, 1.00, 0.95, 0.90, 0.85, 0.85, 0.90, 0.95, 1.00, 1.05, 1.05]
transient = true
```

### Precipitation-only run

```toml
seed = 1

[io]
input = "data/precip_only.nc"
output = "output/precip.parquet"
temp_max_var = null
temp_min_var = null
```
