use std::collections::BTreeMap;
use zeus_calendar::NoLeapDate;
use zeus_evaluate::{EvaluateConfig, MultiSiteSynthetic, evaluate};
use zeus_io::{GridMetadata, MultiSiteData, ObservedData, SyntheticWeather};

/// Helper to create a sequence of NoLeapDate values.
fn make_dates(year: i32, month: u8, day: u8, n: usize) -> Vec<NoLeapDate> {
    let mut dates = Vec::with_capacity(n);
    let mut d = NoLeapDate::new(year, month, day).unwrap();
    for _ in 0..n {
        dates.push(d);
        d = d.next();
    }
    dates
}

#[test]
fn test_evaluate_full_pipeline() {
    // Create 60 days of data starting Jan 1 2000 (covers Jan + Feb)
    let dates = make_dates(2000, 1, 1, 60);

    // Site A observed data
    let precip_a: Vec<f64> = (0..60)
        .map(|i| if i % 3 == 0 { 5.0 } else { 0.5 })
        .collect();
    let temp_max_a: Vec<f64> = (0..60).map(|i| 20.0 + (i % 10) as f64).collect();

    // Site B observed data
    let precip_b: Vec<f64> = (0..60)
        .map(|i| if i % 4 == 0 { 4.0 } else { 0.3 })
        .collect();
    let temp_max_b: Vec<f64> = (0..60).map(|i| 18.0 + (i % 8) as f64).collect();

    let obs_a = ObservedData::new(precip_a, Some(temp_max_a), None, dates.clone(), 1).unwrap();
    let obs_b = ObservedData::new(precip_b, Some(temp_max_b), None, dates.clone(), 1).unwrap();

    let mut obs_sites = BTreeMap::new();
    obs_sites.insert("site_a".to_string(), obs_a);
    obs_sites.insert("site_b".to_string(), obs_b);

    let grid = GridMetadata::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let observed = MultiSiteData::new(obs_sites, grid).unwrap();

    // Create synthetic data (2 realisations per site)
    // Extract months from dates
    let months: Vec<u8> = dates.iter().map(|d| d.month()).collect();
    let water_years = vec![2000; 60];
    let days_of_year: Vec<u16> = (1..=60).collect();

    // Realisation 0 for site A (slightly different from observed)
    let syn_precip_a0: Vec<f64> = (0..60)
        .map(|i| if i % 3 == 0 { 5.5 } else { 0.6 })
        .collect();
    let syn_temp_max_a0: Vec<f64> = (0..60).map(|i| 20.5 + (i % 10) as f64).collect();

    // Realisation 1 for site A (more different)
    let syn_precip_a1: Vec<f64> = (0..60)
        .map(|i| if i % 3 == 0 { 6.0 } else { 0.7 })
        .collect();
    let syn_temp_max_a1: Vec<f64> = (0..60).map(|i| 21.0 + (i % 10) as f64).collect();

    // Realisation 0 for site B
    let syn_precip_b0: Vec<f64> = (0..60)
        .map(|i| if i % 4 == 0 { 4.5 } else { 0.4 })
        .collect();
    let syn_temp_max_b0: Vec<f64> = (0..60).map(|i| 18.5 + (i % 8) as f64).collect();

    // Realisation 1 for site B
    let syn_precip_b1: Vec<f64> = (0..60)
        .map(|i| if i % 4 == 0 { 5.0 } else { 0.5 })
        .collect();
    let syn_temp_max_b1: Vec<f64> = (0..60).map(|i| 19.0 + (i % 8) as f64).collect();

    let syn_a0 = SyntheticWeather::new(
        "site_a",
        &syn_precip_a0,
        Some(&syn_temp_max_a0),
        None,
        &months,
        &water_years,
        &days_of_year,
        0,
    )
    .unwrap();

    let syn_a1 = SyntheticWeather::new(
        "site_a",
        &syn_precip_a1,
        Some(&syn_temp_max_a1),
        None,
        &months,
        &water_years,
        &days_of_year,
        1,
    )
    .unwrap();

    let syn_b0 = SyntheticWeather::new(
        "site_b",
        &syn_precip_b0,
        Some(&syn_temp_max_b0),
        None,
        &months,
        &water_years,
        &days_of_year,
        0,
    )
    .unwrap();

    let syn_b1 = SyntheticWeather::new(
        "site_b",
        &syn_precip_b1,
        Some(&syn_temp_max_b1),
        None,
        &months,
        &water_years,
        &days_of_year,
        1,
    )
    .unwrap();

    let mut syn_sites = BTreeMap::new();
    syn_sites.insert("site_a".to_string(), vec![syn_a0, syn_a1]);
    syn_sites.insert("site_b".to_string(), vec![syn_b0, syn_b1]);

    let synthetic = MultiSiteSynthetic::new(syn_sites).unwrap();

    // Evaluate with default config
    let config = EvaluateConfig::default();
    let json = evaluate(&observed, &synthetic, &config).unwrap();

    // Parse JSON to verify structure
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Verify config
    assert!(parsed["config"]["precip_threshold"].as_f64().is_some());
    assert_eq!(parsed["config"]["n_sites"].as_u64().unwrap(), 2);
    assert_eq!(parsed["config"]["n_realisations"].as_u64().unwrap(), 2);

    let variables = parsed["config"]["variables"].as_array().unwrap();
    assert_eq!(variables.len(), 2);
    assert_eq!(variables[0].as_str().unwrap(), "precip");
    assert_eq!(variables[1].as_str().unwrap(), "temp_max");

    // Verify scorecard
    let scorecard = parsed["scorecard"].as_array().unwrap();
    assert_eq!(scorecard.len(), 2);

    // Verify ranks are 1 and 2
    let ranks: Vec<u64> = scorecard
        .iter()
        .map(|s| s["rank"].as_u64().unwrap())
        .collect();
    assert!(ranks.contains(&1));
    assert!(ranks.contains(&2));

    // Verify all scores have non-negative raw_mae
    for score in scorecard {
        let raw_mae = score["raw_mae"].as_f64().unwrap();
        assert!(raw_mae >= 0.0);

        let normalized = score["normalized_score"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&normalized));
    }

    // Verify diagnostics structure
    assert!(parsed["diagnostics"]["timeseries"].is_object());

    let timeseries = parsed["diagnostics"]["timeseries"].as_object().unwrap();
    assert_eq!(timeseries.len(), 2); // 2 sites

    // Each site should have month entries
    for (_site, site_data) in timeseries {
        let site_obj = site_data.as_object().unwrap();
        assert!(!site_obj.is_empty()); // Should have at least one month
    }

    assert!(parsed["diagnostics"]["cross_grid_correlations"].is_array());
    assert!(parsed["diagnostics"]["inter_variable_correlations"].is_array());
    assert!(parsed["diagnostics"]["conditional_correlations"].is_array());

    // Cross-grid should have entries for each variable for the site pair
    let cross_grid = parsed["diagnostics"]["cross_grid_correlations"]
        .as_array()
        .unwrap();
    assert!(!cross_grid.is_empty());

    // Inter-variable should have entries for precip-temp_max at each site
    let inter_var = parsed["diagnostics"]["inter_variable_correlations"]
        .as_array()
        .unwrap();
    assert!(!inter_var.is_empty());

    // Conditional should have entries for temp_max under different regimes
    let conditional = parsed["diagnostics"]["conditional_correlations"]
        .as_array()
        .unwrap();
    assert!(!conditional.is_empty());
}

#[test]
fn test_evaluate_mismatched_sites_error() {
    // Create observed data with site "site_a"
    let dates = make_dates(2000, 1, 1, 30);
    let precip: Vec<f64> = vec![1.0; 30];
    let temp_max: Vec<f64> = vec![20.0; 30];

    let obs = ObservedData::new(
        precip.clone(),
        Some(temp_max.clone()),
        None,
        dates.clone(),
        1,
    )
    .unwrap();
    let mut obs_sites = BTreeMap::new();
    obs_sites.insert("site_a".to_string(), obs);

    let grid = GridMetadata::new(vec![0.0], vec![0.0]).unwrap();
    let observed = MultiSiteData::new(obs_sites, grid).unwrap();

    // Create synthetic data with site "site_b" (different site)
    let months: Vec<u8> = dates.iter().map(|d| d.month()).collect();
    let water_years = vec![2000; 30];
    let days_of_year: Vec<u16> = (1..=30).collect();

    let syn = SyntheticWeather::new(
        "site_b",
        &precip,
        Some(&temp_max),
        None,
        &months,
        &water_years,
        &days_of_year,
        0,
    )
    .unwrap();

    let mut syn_sites = BTreeMap::new();
    syn_sites.insert("site_b".to_string(), vec![syn]);

    let synthetic = MultiSiteSynthetic::new(syn_sites).unwrap();

    // Evaluate should return error
    let config = EvaluateConfig::default();
    let result = evaluate(&observed, &synthetic, &config);

    assert!(result.is_err());
    match result {
        Err(zeus_evaluate::EvaluateError::MissingSite { site, location }) => {
            // Either site_a missing from synthetic or site_b missing from observed
            assert!(
                (site == "site_a" && location == "synthetic")
                    || (site == "site_b" && location == "observed")
            );
        }
        _ => panic!("Expected MissingSite error"),
    }
}
