use std::collections::BTreeMap;

use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tracing::info;

use zeus_calendar::{NoLeapDate, noleap_sequence};
use zeus_evaluate::{MultiSiteSynthetic, evaluate};
use zeus_io::{OwnedSyntheticWeather, SyntheticWeather, read_netcdf, write_parquet};
use zeus_resample::{ObsData, resample_dates};
use zeus_warm::{filter_warm_pool, simulate_warm};

use crate::cli::GenerateArgs;
use crate::config::ZeusConfig;
use crate::convert;

/// Run the full generation pipeline.
pub fn run(args: GenerateArgs) -> Result<()> {
    let toml_str = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config file: {}", args.config.display()))?;
    let mut config: ZeusConfig =
        toml::from_str(&toml_str).context("failed to parse TOML config")?;

    // CLI overrides
    if let Some(ref output) = args.output {
        config.io.output = Some(output.clone());
    }
    if let Some(seed) = args.seed {
        config.seed = Some(seed);
    }
    let config = config;

    // Step 1: Resolve paths
    let input =
        config.io.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no input path: set [io].input in config or use --input")
        })?;
    let output = config.io.output.as_ref().ok_or_else(|| {
        anyhow::anyhow!("no output path: set [io].output in config or use --output")
    })?;

    // Step 2: Build configs from TOML
    let reader_cfg = convert::build_reader_config(&config.io)?;
    let warm_cfg = convert::build_warm_config(&config.warm, config.seed)?;
    let bounds = convert::build_filter_bounds(&config.filter);
    let resample_cfg = convert::build_resample_config(&config.resample, config.io.start_month)?;
    let markov_cfg = convert::build_markov_config(&config.markov)?;
    let writer_cfg = convert::build_writer_config(&config.io)?;
    let eval_cfg = convert::build_evaluate_config(&config.evaluate);

    // Step 3: Create seeded RNG
    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    // Step 4: Read observed data
    info!(path = %input.display(), "reading observed data");
    let multi_site = read_netcdf(input, &reader_cfg)
        .with_context(|| format!("failed to read NetCDF: {}", input.display()))?;
    if multi_site.n_sites() == 0 {
        anyhow::bail!("no valid grid cells after filtering missing data");
    }
    info!(
        n_sites = multi_site.n_sites(),
        n_timesteps = multi_site.n_timesteps(),
        "observed data loaded"
    );

    // Per-site owned synthetic data accumulator
    let mut site_synthetics: BTreeMap<String, Vec<OwnedSyntheticWeather>> = BTreeMap::new();

    // Step 5: For each site
    for (site_key, obs) in multi_site.iter() {
        info!(site = %site_key, "processing site");

        // 5a: Compute annual precip (group daily precip by water year, sum)
        let annual_precip = compute_annual_precip(obs.precip(), obs.water_years());
        info!(
            site = %site_key,
            n_years = annual_precip.len(),
            "computed annual precipitation"
        );

        // 5b: WARM simulation
        let warm_result = simulate_warm(&annual_precip, &warm_cfg)
            .with_context(|| format!("WARM simulation failed for site {site_key}"))?;
        info!(
            site = %site_key,
            n_sim = warm_result.n_sim(),
            "WARM simulation complete"
        );

        // 5c: Filter WARM pool
        let filtered = filter_warm_pool(&annual_precip, &warm_result, &bounds)
            .with_context(|| format!("WARM filtering failed for site {site_key}"))?;
        info!(
            site = %site_key,
            n_selected = filtered.n_selected(),
            "filtered WARM pool"
        );

        // 5d: Bridge observed data for resampling
        let obs_data = bridge_obs(obs)?;

        // 5e: For each selected realisation, resample daily data
        let mut site_reals = Vec::new();
        for (real_idx, &sim_idx) in filtered.selected().iter().enumerate() {
            let sim_annual = warm_result.simulations()[sim_idx].as_slice();
            let n_sim_years = sim_annual.len();

            // Resample dates
            let indices =
                resample_dates(sim_annual, &obs_data, &markov_cfg, &resample_cfg, &mut rng)
                    .with_context(|| {
                        format!("resampling failed for site {site_key}, realisation {real_idx}")
                    })?;

            // Reconstruct synthetic daily arrays from observation indices
            let syn_precip: Vec<f64> = indices.iter().map(|&i| obs.precip()[i]).collect();
            let syn_tmax: Option<Vec<f64>> = obs
                .temp_max()
                .map(|t| indices.iter().map(|&i| t[i]).collect());
            let syn_tmin: Option<Vec<f64>> = obs
                .temp_min()
                .map(|t| indices.iter().map(|&i| t[i]).collect());

            // Generate calendar metadata for synthetic period
            // n_sim_years * 365 days (no-leap calendar)
            let start_date = NoLeapDate::new(1, 1, 1).context("failed to create start date")?;
            let n_days = n_sim_years * 365;
            let syn_dates = noleap_sequence(start_date, n_days);
            let syn_months: Vec<u8> = syn_dates.iter().map(|d| d.month()).collect();
            let syn_water_years: Vec<i32> = syn_dates
                .iter()
                .map(|d| {
                    zeus_calendar::water_year(d.year(), d.month(), config.io.start_month)
                        .unwrap_or(d.year())
                })
                .collect();
            let syn_days_of_year: Vec<u16> = syn_dates.iter().map(|d| d.doy().get()).collect();

            site_reals.push(OwnedSyntheticWeather {
                precip: syn_precip,
                temp_max: syn_tmax,
                temp_min: syn_tmin,
                months: syn_months,
                water_years: syn_water_years,
                days_of_year: syn_days_of_year,
                realisation: real_idx as u32,
            });
        }

        site_synthetics.insert(site_key.clone(), site_reals);
    }

    // Step 7: Build SyntheticWeather views and write parquet
    let all_views: Vec<SyntheticWeather<'_>> = site_synthetics
        .values()
        .flat_map(|reals| reals.iter().map(|o| o.as_view()))
        .collect::<Result<Vec<_>, _>>()
        .context("failed to build synthetic weather views")?;

    info!(
        path = %output.display(),
        n_realisations = all_views.len(),
        "writing synthetic data"
    );
    write_parquet(output, &all_views, &writer_cfg)
        .with_context(|| format!("failed to write Parquet: {}", output.display()))?;
    info!("parquet output written");

    // Step 8: Inline evaluation
    let eval_views: BTreeMap<String, Vec<SyntheticWeather<'_>>> = site_synthetics
        .iter()
        .map(|(key, reals)| {
            let views: Result<Vec<_>, _> = reals.iter().map(|o| o.as_view()).collect();
            views.map(|v| (key.clone(), v))
        })
        .collect::<Result<_, _>>()
        .context("failed to build evaluation views")?;

    let multi_syn =
        MultiSiteSynthetic::new(eval_views).context("failed to build MultiSiteSynthetic")?;

    info!("running evaluation diagnostics");
    let json = evaluate(&multi_site, &multi_syn, &eval_cfg).context("evaluation failed")?;

    // Write diagnostics JSON to {output_stem}.diagnostics.json
    let diag_path = output.with_extension("diagnostics.json");
    std::fs::write(&diag_path, &json)
        .with_context(|| format!("failed to write diagnostics: {}", diag_path.display()))?;
    info!(path = %diag_path.display(), "diagnostics written");

    Ok(())
}

/// Compute annual precipitation totals from daily data grouped by water year.
fn compute_annual_precip(precip: &[f64], water_years: &[i32]) -> Vec<f64> {
    let mut totals: BTreeMap<i32, f64> = BTreeMap::new();
    for (&p, &wy) in precip.iter().zip(water_years) {
        *totals.entry(wy).or_insert(0.0) += p;
    }
    totals.into_values().collect()
}

/// Bridge `io::ObservedData` to `resample::ObsData`.
fn bridge_obs(obs: &zeus_io::ObservedData) -> Result<ObsData> {
    let temp_mean: Vec<f64> = match (obs.temp_max(), obs.temp_min()) {
        (Some(tmax), Some(tmin)) => tmax.iter().zip(tmin).map(|(a, b)| (a + b) / 2.0).collect(),
        _ => vec![0.0; obs.len()],
    };
    let days: Vec<u8> = obs.dates().iter().map(|d| d.day()).collect();
    ObsData::new(
        obs.precip(),
        &temp_mean,
        obs.months(),
        &days,
        obs.water_years(),
    )
    .map_err(|e| anyhow::anyhow!("building ObsData: {e}"))
}
