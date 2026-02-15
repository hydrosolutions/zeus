use std::collections::BTreeMap;

use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tracing::{debug_span, info, info_span};

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
    let _cmd = info_span!("generate").entered();
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

    // Step 5: Compute area-averaged observed data across all sites
    let area_avg_obs = area_average_observed(&multi_site, config.io.start_month)?;
    info!(
        from_sites = multi_site.n_sites(),
        "area-averaged observations into single timeseries"
    );

    // Step 6: Compute area-averaged annual precipitation
    let area_avg_annual = compute_annual_precip(area_avg_obs.precip(), area_avg_obs.water_years());
    info!(
        n_years = area_avg_annual.len(),
        "computed area-averaged annual precipitation"
    );

    // Step 7: WARM simulation on area average (single shared simulation)
    let warm_result = simulate_warm(&area_avg_annual, &warm_cfg)
        .context("WARM simulation failed on area average")?;
    info!(n_sim = warm_result.n_sim(), "WARM simulation complete");

    // Step 8: Filter WARM pool on area average
    let filtered = filter_warm_pool(&area_avg_annual, &warm_result, &bounds)
        .context("WARM filtering failed on area average")?;
    info!(n_selected = filtered.n_selected(), "filtered WARM pool");

    // Step 9: Bridge area-averaged observations for resampling
    let area_avg_obs_data = bridge_obs(&area_avg_obs)?;

    // Step 10: Pre-compute shared calendar metadata
    let n_sim_years = warm_result.n_years();
    let calendar = compute_synthetic_calendar(n_sim_years, config.io.start_month)?;

    // Step 11: Initialise per-site accumulators
    let mut site_synthetics: BTreeMap<String, Vec<OwnedSyntheticWeather>> = BTreeMap::new();
    for site_key in multi_site.keys() {
        site_synthetics.insert(site_key.clone(), Vec::new());
    }

    // Step 12: For each realisation, resample ONCE on area average,
    // then distribute the shared indices to every site
    for (real_idx, &sim_idx) in filtered.selected().iter().enumerate() {
        let _real = debug_span!("realisation", idx = real_idx, sim = sim_idx).entered();
        let sim_annual = warm_result.simulations()[sim_idx].as_slice();

        // Resample once on area-averaged data
        let shared_indices = resample_dates(
            sim_annual,
            &area_avg_obs_data,
            &markov_cfg,
            &resample_cfg,
            &mut rng,
        )
        .with_context(|| format!("resampling failed for realisation {real_idx}"))?;

        // Distribute shared indices to each site
        for (site_key, obs) in multi_site.iter() {
            let weather = extract_site_weather(
                &shared_indices,
                obs,
                &calendar,
                real_idx as u32,
                site_key.clone(),
            );
            site_synthetics.get_mut(site_key).unwrap().push(weather);
        }
    }

    // Step 7: Build SyntheticWeather views and write parquet
    let all_views: Vec<SyntheticWeather<'_>> = site_synthetics
        .values()
        .flat_map(|reals| reals.iter().map(|o| o.as_view()))
        .collect::<Result<Vec<_>, _>>()
        .context("failed to build synthetic weather views")?;

    info!(
        path = %output.display(),
        n_sites = multi_site.n_sites(),
        n_realisations = filtered.n_selected(),
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

/// Compute element-wise area-averaged observed data across all sites.
///
/// Returns a single `ObservedData` whose precipitation (and optional
/// temperatures) are the per-timestep arithmetic mean across every site
/// in `multi_site`.  Dates come from the first site; all sites share the
/// same time axis per `MultiSiteData::new` validation.
fn area_average_observed(
    multi_site: &zeus_io::MultiSiteData,
    start_month: u8,
) -> Result<zeus_io::ObservedData> {
    let n_sites = multi_site.n_sites();
    let n_steps = multi_site.n_timesteps();
    anyhow::ensure!(n_sites > 0, "area_average_observed: no sites");

    // Accumulate sums
    let mut precip_sum = vec![0.0_f64; n_steps];
    let mut tmax_sum: Option<Vec<f64>> = None;
    let mut tmin_sum: Option<Vec<f64>> = None;
    let mut all_have_tmax = true;
    let mut all_have_tmin = true;
    let mut first_dates: Option<Vec<zeus_calendar::NoLeapDate>> = None;

    for (_key, obs) in multi_site.iter() {
        // Accumulate precip
        for (acc, &val) in precip_sum.iter_mut().zip(obs.precip()) {
            *acc += val;
        }

        // Track temperature availability
        match obs.temp_max() {
            Some(tmax) => {
                let sum = tmax_sum.get_or_insert_with(|| vec![0.0; n_steps]);
                for (acc, &val) in sum.iter_mut().zip(tmax) {
                    *acc += val;
                }
            }
            None => all_have_tmax = false,
        }
        match obs.temp_min() {
            Some(tmin) => {
                let sum = tmin_sum.get_or_insert_with(|| vec![0.0; n_steps]);
                for (acc, &val) in sum.iter_mut().zip(tmin) {
                    *acc += val;
                }
            }
            None => all_have_tmin = false,
        }

        // Capture dates from the first site
        if first_dates.is_none() {
            first_dates = Some(obs.dates().to_vec());
        }
    }

    let dates = first_dates.expect("at least one site guaranteed by ensure above");
    let divisor = n_sites as f64;

    // Average precip
    let avg_precip: Vec<f64> = precip_sum.into_iter().map(|s| s / divisor).collect();

    // Average temperatures only if ALL sites have them
    let avg_tmax: Option<Vec<f64>> = if all_have_tmax {
        tmax_sum.map(|sums| sums.into_iter().map(|s| s / divisor).collect())
    } else {
        None
    };
    let avg_tmin: Option<Vec<f64>> = if all_have_tmin {
        tmin_sum.map(|sums| sums.into_iter().map(|s| s / divisor).collect())
    } else {
        None
    };

    zeus_io::ObservedData::new(avg_precip, avg_tmax, avg_tmin, dates, start_month)
        .map_err(|e| anyhow::anyhow!("area_average_observed: {e}"))
}

/// Pre-computed calendar metadata for a synthetic simulation period.
///
/// Because every realisation within a WARM simulation shares the same
/// number of years (and thus the same number of days in a no-leap
/// calendar), the month / water-year / day-of-year arrays can be
/// computed once and reused.
struct SyntheticCalendar {
    months: Vec<u8>,
    water_years: Vec<i32>,
    days_of_year: Vec<u16>,
}

/// Build calendar metadata for `n_sim_years` years of no-leap daily data.
fn compute_synthetic_calendar(n_sim_years: usize, start_month: u8) -> Result<SyntheticCalendar> {
    let start_date =
        NoLeapDate::new(1, start_month, 1).context("failed to create synthetic start date")?;
    let n_days = n_sim_years * 365;
    let dates = noleap_sequence(start_date, n_days);

    let months: Vec<u8> = dates.iter().map(|d| d.month()).collect();
    let water_years: Vec<i32> = dates
        .iter()
        .map(|d| zeus_calendar::water_year(d.year(), d.month(), start_month).unwrap_or(d.year()))
        .collect();
    let days_of_year: Vec<u16> = dates.iter().map(|d| d.doy().get()).collect();

    Ok(SyntheticCalendar {
        months,
        water_years,
        days_of_year,
    })
}

/// Apply shared resampled indices to one site's observed data,
/// producing a single owned realisation of synthetic weather.
fn extract_site_weather(
    indices: &[usize],
    obs: &zeus_io::ObservedData,
    calendar: &SyntheticCalendar,
    realisation: u32,
    site: String,
) -> OwnedSyntheticWeather {
    let syn_precip: Vec<f64> = indices.iter().map(|&i| obs.precip()[i]).collect();
    let syn_tmax: Option<Vec<f64>> = obs
        .temp_max()
        .map(|t| indices.iter().map(|&i| t[i]).collect());
    let syn_tmin: Option<Vec<f64>> = obs
        .temp_min()
        .map(|t| indices.iter().map(|&i| t[i]).collect());

    OwnedSyntheticWeather {
        site,
        precip: syn_precip,
        temp_max: syn_tmax,
        temp_min: syn_tmin,
        months: calendar.months.clone(),
        water_years: calendar.water_years.clone(),
        days_of_year: calendar.days_of_year.clone(),
        realisation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use zeus_calendar::NoLeapDate;
    use zeus_io::{GridMetadata, MultiSiteData, ObservedData};

    /// Helper: build an ObservedData with the given daily precip/tmax/tmin.
    fn make_obs(precip: Vec<f64>, tmax: Option<Vec<f64>>, tmin: Option<Vec<f64>>) -> ObservedData {
        let n = precip.len();
        let dates: Vec<NoLeapDate> = (0..n)
            .map(|i| {
                let day = (i % 28) as u8 + 1;
                let month = ((i / 28) % 12) as u8 + 1;
                NoLeapDate::new(1, month, day).unwrap()
            })
            .collect();
        ObservedData::new(precip, tmax, tmin, dates, 1).unwrap()
    }

    fn make_multi(sites: Vec<(&str, ObservedData)>) -> MultiSiteData {
        let n = sites.len();
        let map: BTreeMap<String, ObservedData> =
            sites.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let grid = GridMetadata::new(vec![0.0; n], vec![0.0; n]).unwrap();
        MultiSiteData::new(map, grid).unwrap()
    }

    #[test]
    fn test_area_average_two_sites() {
        let obs_a = make_obs(
            vec![2.0, 4.0, 6.0],
            Some(vec![30.0, 32.0, 34.0]),
            Some(vec![20.0, 22.0, 24.0]),
        );
        let obs_b = make_obs(
            vec![4.0, 6.0, 8.0],
            Some(vec![34.0, 36.0, 38.0]),
            Some(vec![24.0, 26.0, 28.0]),
        );
        let ms = make_multi(vec![("a", obs_a), ("b", obs_b)]);
        let avg = area_average_observed(&ms, 1).unwrap();

        // Mean precip: (2+4)/2=3, (4+6)/2=5, (6+8)/2=7
        assert_eq!(avg.precip(), &[3.0, 5.0, 7.0]);
        // Mean tmax: (30+34)/2=32, (32+36)/2=34, (34+38)/2=36
        assert_eq!(avg.temp_max().unwrap(), &[32.0, 34.0, 36.0]);
        // Mean tmin: (20+24)/2=22, (22+26)/2=24, (24+28)/2=26
        assert_eq!(avg.temp_min().unwrap(), &[22.0, 24.0, 26.0]);
    }

    #[test]
    fn test_area_average_single_site() {
        let obs = make_obs(
            vec![1.0, 2.0, 3.0],
            Some(vec![10.0, 11.0, 12.0]),
            Some(vec![5.0, 6.0, 7.0]),
        );
        let ms = make_multi(vec![("only", obs)]);
        let avg = area_average_observed(&ms, 1).unwrap();

        assert_eq!(avg.precip(), &[1.0, 2.0, 3.0]);
        assert_eq!(avg.temp_max().unwrap(), &[10.0, 11.0, 12.0]);
        assert_eq!(avg.temp_min().unwrap(), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_area_average_no_temperature() {
        let obs_a = make_obs(vec![2.0, 4.0], None, None);
        let obs_b = make_obs(vec![6.0, 8.0], None, None);
        let ms = make_multi(vec![("a", obs_a), ("b", obs_b)]);
        let avg = area_average_observed(&ms, 1).unwrap();

        assert_eq!(avg.precip(), &[4.0, 6.0]);
        assert!(avg.temp_max().is_none());
        assert!(avg.temp_min().is_none());
    }

    #[test]
    fn test_synthetic_calendar_length() {
        let cal = compute_synthetic_calendar(3, 1).unwrap();
        assert_eq!(cal.months.len(), 3 * 365);
        assert_eq!(cal.water_years.len(), 3 * 365);
        assert_eq!(cal.days_of_year.len(), 3 * 365);
        // First month should be January (start_month=1)
        assert_eq!(cal.months[0], 1);
    }

    #[test]
    fn test_extract_site_weather() {
        // Build a small ObservedData
        let precip = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let tmax = Some(vec![25.0, 26.0, 27.0, 28.0, 29.0]);
        let tmin = Some(vec![15.0, 16.0, 17.0, 18.0, 19.0]);
        let dates: Vec<NoLeapDate> = (0..5)
            .map(|i| NoLeapDate::new(1, 1, (i as u8) + 1).unwrap())
            .collect();
        let obs = zeus_io::ObservedData::new(precip, tmax, tmin, dates, 1).unwrap();

        // Use indices [2, 0, 4] to pick days 3, 1, 5
        let indices = vec![2, 0, 4];
        let cal = SyntheticCalendar {
            months: vec![1, 1, 1],
            water_years: vec![1, 1, 1],
            days_of_year: vec![1, 2, 3],
        };
        let weather = extract_site_weather(&indices, &obs, &cal, 0, "test_site".to_string());

        assert_eq!(weather.site, "test_site");
        assert_eq!(weather.precip, vec![30.0, 10.0, 50.0]);
        assert_eq!(weather.temp_max.as_ref().unwrap(), &vec![27.0, 25.0, 29.0]);
        assert_eq!(weather.temp_min.as_ref().unwrap(), &vec![17.0, 15.0, 19.0]);
        assert_eq!(weather.months, vec![1, 1, 1]);
        assert_eq!(weather.realisation, 0);
    }
}
