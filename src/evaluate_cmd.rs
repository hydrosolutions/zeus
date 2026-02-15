//! Evaluate command: compare synthetic output against observed data.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use tracing::{info, info_span};

use zeus_evaluate::{MultiSiteSynthetic, evaluate};
use zeus_io::{SyntheticWeather, read_netcdf, read_parquet};

use crate::cli::EvaluateArgs;
use crate::config::ZeusConfig;
use crate::convert;

/// Run the standalone evaluation pipeline.
pub fn run(args: EvaluateArgs) -> Result<()> {
    let _cmd = info_span!("evaluate").entered();
    // 1. Load project TOML
    let toml_str = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config file: {}", args.config.display()))?;
    let config: ZeusConfig = toml::from_str(&toml_str).context("failed to parse TOML config")?;

    // 2. Read observed data
    let input =
        config.io.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no input path: set [io].input in config or use --input")
        })?;
    let reader_cfg = convert::build_reader_config(&config.io)?;

    info!(path = %input.display(), "reading observed data");
    let multi_site = read_netcdf(input, &reader_cfg)
        .with_context(|| format!("failed to read NetCDF: {}", input.display()))?;
    info!(
        n_sites = multi_site.n_sites(),
        n_timesteps = multi_site.n_timesteps(),
        "observed data loaded"
    );

    // 3. Enforce site count
    if multi_site.n_sites() == 0 {
        bail!("no valid grid cells after filtering missing data");
    }
    if multi_site.n_sites() > 1 {
        bail!(
            "standalone evaluate requires single-site observed data, got {} sites. \
             Multi-site evaluation is available via `zeus generate` inline evaluation.",
            multi_site.n_sites()
        );
    }

    // 4. Read synthetic Parquet
    info!(path = %args.synthetic.display(), "reading synthetic data");
    let mut site_map = read_parquet(&args.synthetic)
        .with_context(|| format!("failed to read Parquet: {}", args.synthetic.display()))?;

    // Extract realisations: use single site if only one, otherwise look up by observed site key
    let site_key = multi_site
        .keys()
        .next()
        .expect("single site validated above")
        .clone();
    let owned_realisations: Vec<zeus_io::OwnedSyntheticWeather> = if site_map.len() == 1 {
        site_map.into_values().next().unwrap_or_default()
    } else {
        site_map.remove(&site_key).ok_or_else(|| {
            let available: Vec<_> = site_map.keys().collect();
            anyhow::anyhow!(
                "observed site '{}' not found in synthetic data (available: {:?})",
                site_key,
                available,
            )
        })?
    };

    info!(
        n_realisations = owned_realisations.len(),
        "synthetic data loaded"
    );

    if owned_realisations.is_empty() {
        bail!("synthetic Parquet file contains no realisations");
    }

    // 5. Build SyntheticWeather views
    let views: Vec<SyntheticWeather<'_>> = owned_realisations
        .iter()
        .map(|o| o.as_view())
        .collect::<Result<Vec<_>, _>>()
        .context("failed to build synthetic weather views")?;

    // 6. Build MultiSiteSynthetic — assign all realisations to the observed site key
    let mut eval_views: BTreeMap<String, Vec<SyntheticWeather<'_>>> = BTreeMap::new();
    eval_views.insert(site_key, views);

    let multi_syn =
        MultiSiteSynthetic::new(eval_views).context("failed to build MultiSiteSynthetic")?;

    // 7. Run evaluation
    let eval_cfg = convert::build_evaluate_config(&config.evaluate);

    info!("running evaluation diagnostics");
    let json = evaluate(&multi_site, &multi_syn, &eval_cfg).context("evaluation failed")?;

    // 8. Write diagnostics JSON
    let diag_path = args.output.unwrap_or_else(|| {
        // Auto-derive: foo.parquet -> foo.diagnostics.json
        args.synthetic.with_extension("diagnostics.json")
    });

    std::fs::write(&diag_path, &json)
        .with_context(|| format!("failed to write diagnostics: {}", diag_path.display()))?;
    info!(path = %diag_path.display(), "diagnostics written");

    Ok(())
}
