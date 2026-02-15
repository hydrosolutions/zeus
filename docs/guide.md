# Zeus User Guide

Zeus is a stochastic weather generator written in pure Rust. It reads observed climate data from NetCDF files, simulates synthetic weather sequences using a Wavelet-ARMA (WARM) pipeline, and writes output to Parquet. An optional perturbation stage applies climate-change scenarios to the synthetic data.

## Quick Start

```bash
# 1. Generate synthetic weather from observed data
zeus generate -c zeus.toml

# 2. Apply climate perturbations (optional)
zeus perturb -i output/synthetic.parquet -o output/perturbed.parquet --temp-delta 2.0

# 3. Evaluate synthetic output against observations
zeus evaluate -c zeus.toml --synthetic output/synthetic.parquet
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

### Global Flag

| Flag | Description |
|------|-------------|
| `-v` | Verbosity (stackable: `-v` info, `-vv` debug, `-vvv` trace). Default: warn. |

### `zeus generate`

Runs the full pipeline: read observed data, WARM simulation, daily disaggregation, write Parquet, run evaluation diagnostics.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config <PATH>` | `-c` | `zeus.toml` | Path to project TOML config file. |
| `--output <PATH>` | `-o` | — | Override `io.output` from config. |
| `--seed <SEED>` | `-s` | — | Override `seed` from config. |

```bash
zeus generate
zeus generate -c project.toml -o output/syn.parquet -s 42 -vv
```

### `zeus perturb`

Applies climate perturbations to an existing synthetic Parquet file. Perturbations can be specified via a TOML config file, CLI flags, or both (CLI flags override config).

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--input <PATH>` | `-i` | yes | Input synthetic Parquet file. |
| `--output <PATH>` | `-o` | yes | Output perturbed Parquet file. |
| `--config <PATH>` | `-c` | no | Perturbation TOML config file. |
| `--temp-delta <F>` | `--dt` | no | Uniform temperature delta (degrees) for all months. |
| `--precip-factor <F>` | `--dp` | no | Uniform precipitation scaling factor for all months. |

At least one of `--config`, `--temp-delta`, or `--precip-factor` must be provided.

```bash
# Apply +2°C temperature shift
zeus perturb -i syn.parquet -o warm.parquet --temp-delta 2.0

# Apply 10% precipitation reduction
zeus perturb -i syn.parquet -o dry.parquet --precip-factor 0.90

# Use a detailed perturbation config
zeus perturb -i syn.parquet -o future.parquet -c perturb.toml
```

### `zeus evaluate`

Compare synthetic output against observations. Reads the observed NetCDF from the project config and a synthetic Parquet file.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config <PATH>` | `-c` | `zeus.toml` | Path to project TOML config file. |
| `--synthetic <PATH>` | — | — | Path to synthetic Parquet file (required). |
| `--output <PATH>` | `-o` | auto | Path for diagnostics JSON. Default: `<synthetic>.diagnostics.json`. |

```bash
zeus evaluate --synthetic output/synthetic.parquet
zeus evaluate -c project.toml --synthetic output/perturbed.parquet -o diag.json
```

> **Note:** Standalone evaluate requires single-site observed data. Multi-site evaluation runs automatically at the end of `zeus generate`.

---

## Configuration Reference

The config file is TOML. Every section except `[io]` is optional — omit a section entirely to use its defaults. All sections enforce `deny_unknown_fields`, so typos in key names produce clear errors.

> **Note:** The `[perturb]` section has been removed from the project TOML. Climate perturbations are now applied via the standalone `zeus perturb` command with its own config file.

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
| `mra_levels` | integer or null | no | auto | MRA decomposition levels. Auto-computed from series length and wavelet filter when omitted. |
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

### Perturbation Config (standalone)

When using `zeus perturb -c <FILE>`, the TOML file uses the same format as the former `[perturb]` section — a flat file, not nested under `[perturb]`:

```toml
precip_floor = 0.0
precip_cap = 500.0

[temperature]
deltas = [1.2, 1.3, 1.5, 1.4, 1.0, 0.8, 0.7, 0.7, 0.9, 1.1, 1.2, 1.3]
transient = true

[quantile_map]
mean_factors = [0.90, 0.90, 0.95, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.90, 0.90]
var_factors  = [1.10, 1.10, 1.05, 1.00, 1.00, 0.95, 0.95, 1.00, 1.05, 1.10, 1.10, 1.10]
intensity_threshold = 0.1

[occurrence]
factors = [1.05, 1.05, 1.00, 0.95, 0.90, 0.85, 0.85, 0.90, 0.95, 1.00, 1.05, 1.05]
transient = true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `precip_floor` | float | `0.0` | Minimum precipitation after perturbation (mm/day). |
| `precip_cap` | float | `500.0` | Maximum precipitation after perturbation (mm/day). |

See below for `[temperature]`, `[quantile_map]`, and `[occurrence]` sub-tables — these use the same schema as the former `[perturb.*]` sections.

CLI flags `--temp-delta` and `--precip-factor` provide a shorthand: `--temp-delta 2.0` is equivalent to setting `[temperature] deltas = [2.0; 12]`, and `--precip-factor 0.90` is equivalent to setting `[quantile_map] mean_factors = [0.90; 12]`.

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

Diagnostics JSON is produced by both `zeus generate` (automatically) and `zeus evaluate` (standalone). When using `zeus evaluate`, the output path defaults to `<synthetic>.diagnostics.json` or can be set with `-o`. It contains per-site comparison metrics between observed and synthetic data including means, variances, wet-day statistics, and cross-realisation consistency scores.

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

Zeus decomposes weather generation into three composable commands:

```
zeus generate                    zeus perturb                  zeus evaluate
┌────────────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐
│ NetCDF input       │   │ Read synthetic Parquet   │   │ Read observed NetCDF │
│   ↓                │   │   ↓                     │   │ Read synthetic Parquet│
│ Annual aggregation │   │ Apply perturbations:    │   │   ↓                  │
│   ↓                │   │   temperature shift     │   │ Compute diagnostics  │
│ Wavelet MRA        │   │   quantile mapping      │   │   timeseries stats   │
│   ↓                │   │   occurrence adjustment  │   │   correlations       │
│ ARMA simulation    │   │   ↓                     │   │   MAE scorecard      │
│   ↓                │   │ Write perturbed Parquet  │   │   ↓                  │
│ Pool filtering     │   └─────────────────────────┘   │ Write diagnostics    │
│   ↓                │                                  │ JSON                 │
│ Markov-KNN resample│                                  └──────────────────────┘
│   ↓                │
│ Write Parquet      │
│   ↓                │
│ Inline evaluation  │
└────────────────────┘
```

The generate step is expensive. Perturbation and evaluation are cheap, enabling workflows like grid sweeps over 900+ scenarios via bash loops.

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

### Three-step workflow

```bash
# 1. Generate base synthetic data
zeus generate -c project.toml -s 42

# 2. Apply climate perturbations
zeus perturb -i output/syn.parquet -o output/future.parquet --temp-delta 2.5 --precip-factor 0.90

# 3. Evaluate perturbed output
zeus evaluate -c project.toml --synthetic output/future.parquet
```

### Grid sweep (30x30 temperature x precipitation scenarios)

```bash
zeus generate -c project.toml

for dt in $(seq 0.0 0.5 14.5); do
  for dp in $(seq 0.70 0.01 0.99); do
    zeus perturb \
      -i output/syn.parquet \
      -o "output/grid/dt${dt}_dp${dp}.parquet" \
      --temp-delta "$dt" --precip-factor "$dp"
  done
done
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

### Detailed perturbation config

```toml
# perturb.toml — used with zeus perturb -c perturb.toml
precip_cap = 300.0

[temperature]
deltas = [1.2, 1.3, 1.5, 1.4, 1.0, 0.8, 0.7, 0.7, 0.9, 1.1, 1.2, 1.3]
transient = true

[quantile_map]
mean_factors = [0.90, 0.90, 0.95, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 0.95, 0.90, 0.90]
var_factors  = [1.10, 1.10, 1.05, 1.00, 1.00, 0.95, 0.95, 1.00, 1.05, 1.10, 1.10, 1.10]
intensity_threshold = 0.1

[occurrence]
factors = [1.05, 1.05, 1.00, 0.95, 0.90, 0.85, 0.85, 0.90, 0.95, 1.00, 1.05, 1.05]
transient = true
```

```bash
zeus perturb -i output/syn.parquet -o output/future.parquet -c perturb.toml
```
