//! Perturb command: apply climate perturbations to synthetic Parquet data.

use anyhow::{Context, Result, bail};
use tracing::{debug_span, info, info_span};

use zeus_io::{OwnedSyntheticWeather, SyntheticWeather, WriterConfig, read_parquet, write_parquet};
use zeus_perturb::apply_perturbations;

use crate::cli::PerturbArgs;
use crate::config::{PerturbToml, QmPerturbToml, TempPerturbToml};
use crate::convert;

/// Run the perturbation pipeline.
pub fn run(args: PerturbArgs) -> Result<()> {
    let _cmd = info_span!("perturb").entered();
    // 1. Validate: at least one perturbation source must be provided
    if args.config.is_none() && args.temp_delta.is_none() && args.precip_factor.is_none() {
        bail!("no perturbation specified: provide --config, --temp-delta, or --precip-factor");
    }

    // 2. Load optional TOML config
    let mut perturb_toml = if let Some(ref config_path) = args.config {
        let toml_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("failed to read perturb config: {}", config_path.display()))?;
        let parsed: PerturbToml =
            toml::from_str(&toml_str).context("failed to parse perturbation TOML")?;
        parsed
    } else {
        PerturbToml {
            precip_floor: 0.0,
            precip_cap: 500.0,
            temperature: None,
            quantile_map: None,
            occurrence: None,
        }
    };

    // 3. Merge CLI flags — CLI overrides config file sections
    if let Some(dt) = args.temp_delta {
        perturb_toml.temperature = Some(TempPerturbToml {
            deltas: [dt; 12],
            transient: false,
        });
    }
    if let Some(dp) = args.precip_factor {
        perturb_toml.quantile_map = Some(QmPerturbToml {
            mean_factors: [dp; 12],
            var_factors: [1.0; 12],
            intensity_threshold: 0.0,
        });
    }

    // 4. Read base Parquet
    info!(path = %args.input.display(), "reading synthetic data");
    let mut realisations = read_parquet(&args.input)
        .with_context(|| format!("failed to read Parquet: {}", args.input.display()))?;
    info!(n = realisations.len(), "loaded realisations");

    if realisations.is_empty() {
        bail!("input Parquet file contains no realisations");
    }

    // 5. Determine n_years from first realisation
    let n_days = realisations[0].len();
    if n_days == 0 || n_days % 365 != 0 {
        bail!(
            "realisation length {} is not a multiple of 365 — cannot determine n_years",
            n_days
        );
    }
    let n_years = n_days / 365;

    // 6. Build PerturbConfig
    let perturb_cfg = convert::build_perturb_config(&perturb_toml, n_years, None)?;

    // 7. Apply perturbations to each realisation
    for owned in &mut realisations {
        let _real = debug_span!("realisation", idx = owned.realisation).entered();
        apply_to_realisation(owned, &perturb_cfg)?;
    }
    info!("perturbations applied to all realisations");

    // 8. Write perturbed Parquet
    let views: Vec<SyntheticWeather<'_>> = realisations
        .iter()
        .map(|o| o.as_view())
        .collect::<Result<Vec<_>, _>>()
        .context("failed to build synthetic weather views")?;

    info!(path = %args.output.display(), "writing perturbed data");
    write_parquet(&args.output, &views, &WriterConfig::default())
        .with_context(|| format!("failed to write Parquet: {}", args.output.display()))?;
    info!("perturbed output written");

    Ok(())
}

/// Apply perturbations to a single realisation in place.
fn apply_to_realisation(
    owned: &mut OwnedSyntheticWeather,
    perturb_cfg: &zeus_perturb::PerturbConfig,
) -> Result<()> {
    // Compute mean temp
    let temp_mean: Vec<f64> = match (&owned.temp_max, &owned.temp_min) {
        (Some(tmax), Some(tmin)) => tmax
            .iter()
            .zip(tmin.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect(),
        _ => vec![0.0; owned.precip.len()],
    };

    // Generate contiguous year indices (1-based, 365 per year)
    let n_total = owned.precip.len();
    let n_yrs = n_total / 365;
    let years: Vec<u32> = (0..n_yrs)
        .flat_map(|y| std::iter::repeat_n((y + 1) as u32, 365))
        .collect();

    let default_temp = vec![0.0; n_total];
    let tmin_slice = owned.temp_min.as_deref().unwrap_or(&default_temp);
    let tmax_slice = owned.temp_max.as_deref().unwrap_or(&default_temp);

    let result = apply_perturbations(
        &owned.precip,
        &temp_mean,
        tmin_slice,
        tmax_slice,
        &owned.months,
        &years,
        perturb_cfg,
    )
    .with_context(|| format!("perturbation failed for realisation {}", owned.realisation))?;

    owned.precip = result.precip().to_vec();
    if owned.temp_max.is_some() {
        owned.temp_max = result.temp_max().map(|s| s.to_vec());
        owned.temp_min = result.temp_min().map(|s| s.to_vec());
    }

    Ok(())
}
