//! Per-(site, month, variable) summary statistics.

#![allow(dead_code)]

/// Summary statistics for a single (site, month, variable) cell.
#[derive(Debug, Clone)]
pub struct TimeseriesStats {
    pub mean: f64,
    pub sd: f64,
    pub skewness: Option<f64>,
    pub wet_days: usize,
    pub dry_days: usize,
    pub mean_wet_spell: f64,
    pub mean_dry_spell: f64,
}

/// Extract values for a specific month from a time series.
pub fn extract_monthly(data: &[f64], months: &[u8], target_month: u8) -> Vec<f64> {
    data.iter()
        .zip(months.iter())
        .filter(|&(_, m)| *m == target_month)
        .map(|(&v, _)| v)
        .collect()
}

/// Compute summary statistics for a single variable/month slice.
pub fn compute_timeseries_stats(
    data: &[f64],
    months: &[u8],
    month: u8,
    threshold: f64,
) -> TimeseriesStats {
    // Extract monthly values
    let monthly_data = extract_monthly(data, months, month);

    // Compute basic statistics
    let mean = zeus_stats::mean(&monthly_data);
    let sd = zeus_stats::sd(&monthly_data);
    let skewness = zeus_stats::skewness(&monthly_data);

    // Classify wet/dry days
    let wet_mask: Vec<bool> = monthly_data.iter().map(|&v| v >= threshold).collect();
    let wet_days = wet_mask.iter().filter(|&&w| w).count();
    let dry_days = wet_mask.iter().filter(|&&w| !w).count();

    // Compute spell lengths
    let (wet_spells, dry_spells) = zeus_stats::spell_lengths(&wet_mask);

    // Compute mean spell lengths
    let mean_wet_spell = if wet_spells.is_empty() {
        0.0
    } else {
        wet_spells.iter().sum::<usize>() as f64 / wet_spells.len() as f64
    };

    let mean_dry_spell = if dry_spells.is_empty() {
        0.0
    } else {
        dry_spells.iter().sum::<usize>() as f64 / dry_spells.len() as f64
    };

    TimeseriesStats {
        mean,
        sd,
        skewness,
        wet_days,
        dry_days,
        mean_wet_spell,
        mean_dry_spell,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_monthly() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let months = vec![1, 2, 1, 2, 1, 3];

        let result = extract_monthly(&data, &months, 1);
        assert_eq!(result, vec![1.0, 3.0, 5.0]);

        let result = extract_monthly(&data, &months, 2);
        assert_eq!(result, vec![2.0, 4.0]);

        let result = extract_monthly(&data, &months, 3);
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_extract_monthly_no_match() {
        let data = vec![1.0, 2.0, 3.0];
        let months = vec![1, 1, 1];

        let result = extract_monthly(&data, &months, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_stats_basic() {
        // Create known data: 6 values for month 1
        let data = vec![0.0, 5.0, 0.5, 10.0, 0.2, 8.0, 100.0];
        let months = vec![1, 1, 1, 1, 1, 1, 2];

        let stats = compute_timeseries_stats(&data, &months, 1, 1.0);

        // Should only use the 6 values from month 1: [0.0, 5.0, 0.5, 10.0, 0.2, 8.0]
        // Mean = (0.0 + 5.0 + 0.5 + 10.0 + 0.2 + 8.0) / 6 = 23.7 / 6 = 3.95
        assert!((stats.mean - 3.95).abs() < 0.01);

        // Wet days (>= 1.0): 5.0, 10.0, 8.0 = 3
        assert_eq!(stats.wet_days, 3);

        // Dry days (< 1.0): 0.0, 0.5, 0.2 = 3
        assert_eq!(stats.dry_days, 3);

        // SD should be > 0
        assert!(stats.sd > 0.0);
    }

    #[test]
    fn test_compute_stats_all_wet() {
        let data = vec![5.0, 10.0, 8.0, 12.0];
        let months = vec![1, 1, 1, 1];

        let stats = compute_timeseries_stats(&data, &months, 1, 1.0);

        assert_eq!(stats.wet_days, 4);
        assert_eq!(stats.dry_days, 0);

        // All wet, so one continuous spell of length 4
        assert!((stats.mean_wet_spell - 4.0).abs() < 0.01);

        // No dry spells
        assert_eq!(stats.mean_dry_spell, 0.0);
    }

    #[test]
    fn test_compute_stats_all_dry() {
        let data = vec![0.0, 0.1, 0.2, 0.3];
        let months = vec![1, 1, 1, 1];

        let stats = compute_timeseries_stats(&data, &months, 1, 1.0);

        assert_eq!(stats.wet_days, 0);
        assert_eq!(stats.dry_days, 4);

        // No wet spells
        assert_eq!(stats.mean_wet_spell, 0.0);

        // All dry, so one continuous spell of length 4
        assert!((stats.mean_dry_spell - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_spell_lengths() {
        // Pattern: wet(2), dry(3), wet(1)
        let data = vec![2.0, 3.0, 0.1, 0.2, 0.3, 5.0];
        let months = vec![1, 1, 1, 1, 1, 1];

        let stats = compute_timeseries_stats(&data, &months, 1, 1.0);

        assert_eq!(stats.wet_days, 3);
        assert_eq!(stats.dry_days, 3);

        // Wet spells: [2, 1] -> mean = 1.5
        assert!((stats.mean_wet_spell - 1.5).abs() < 0.01);

        // Dry spells: [3] -> mean = 3.0
        assert!((stats.mean_dry_spell - 3.0).abs() < 0.01);
    }
}
