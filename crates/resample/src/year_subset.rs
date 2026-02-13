//! Per-year observation subset derived from annual KNN selection.

use crate::config::ResampleConfig;
use crate::error::ResampleError;
use crate::obs_data::ObsData;
use zeus_markov::{
    MarkovConfig, MonthlyTransitions, StateThresholds, estimate_monthly_transitions,
};

/// Per-year observation subset with precomputed statistics and Markov model.
///
/// Built from the annual KNN result. Years selected multiple times
/// contribute duplicated day-level indices (with-replacement semantics).
#[allow(dead_code)]
pub(crate) struct YearSubset {
    /// Indices into ObsData arrays (may contain duplicates from with-replacement).
    subset_indices: Vec<usize>,

    /// Owned subset arrays (extracted from ObsData for cache locality).
    precip: Vec<f64>,
    temp: Vec<f64>,
    months: Vec<u8>,
    water_years: Vec<i32>,

    /// State thresholds estimated from the subset.
    thresholds: StateThresholds,

    /// Monthly transition matrices estimated from the subset.
    transitions: MonthlyTransitions,

    /// Per-month means and SDs for anomaly computation.
    monthly_mean_precip: [f64; 12],
    monthly_mean_temp: [f64; 12],
    monthly_sd_precip: [f64; 12],
    monthly_sd_temp: [f64; 12],

    /// Pre-computed KNN weights: `[precip_weight / sd_precip, temp_weight / sd_temp]`.
    knn_weights: [[f64; 2]; 12],

    /// Subset-local month-day lookup.
    /// Index = `(month - 1) * 31 + (day - 1)`, maps to indices into subset arrays.
    subset_month_day: [Vec<usize>; 372],
}

#[allow(dead_code)]
impl YearSubset {
    /// Build a per-year subset from the annual KNN result.
    ///
    /// `selected_year_indices` are indices into `obs.unique_water_years()`.
    /// May contain duplicates (with-replacement sampling).
    pub(crate) fn build(
        selected_year_indices: &[usize],
        obs: &ObsData,
        config: &ResampleConfig,
        markov_config: &MarkovConfig,
    ) -> Result<Self, ResampleError> {
        let unique_wys = obs.unique_water_years();

        // 1. Collect day-level indices from each selected year.
        //    Duplicates preserved (with-replacement semantics).
        let mut subset_indices = Vec::new();
        for &yi in selected_year_indices {
            let wy = unique_wys[yi];
            let day_indices = obs.year_day_candidates(wy);
            subset_indices.extend_from_slice(day_indices);
        }

        if subset_indices.is_empty() {
            return Err(ResampleError::EmptyData);
        }

        // 2. Extract subset arrays (owned copies for cache locality).
        let obs_precip = obs.precip();
        let obs_temp = obs.temp();
        let obs_months = obs.months();
        let obs_days = obs.days();
        let obs_wy = obs.water_years();

        let precip: Vec<f64> = subset_indices.iter().map(|&i| obs_precip[i]).collect();
        let temp: Vec<f64> = subset_indices.iter().map(|&i| obs_temp[i]).collect();
        let months: Vec<u8> = subset_indices.iter().map(|&i| obs_months[i]).collect();
        let days: Vec<u8> = subset_indices.iter().map(|&i| obs_days[i]).collect();
        let water_years: Vec<i32> = subset_indices.iter().map(|&i| obs_wy[i]).collect();

        // 3. Compute per-month means and SDs.
        let sd_floor = config.sd_floor();
        let mut monthly_mean_precip = [0.0_f64; 12];
        let mut monthly_mean_temp = [0.0_f64; 12];
        let mut monthly_sd_precip = [0.0_f64; 12];
        let mut monthly_sd_temp = [0.0_f64; 12];

        for m in 0..12u8 {
            let month_1 = m + 1;
            let p_vals: Vec<f64> = precip
                .iter()
                .zip(months.iter())
                .filter(|&(_, &mo)| mo == month_1)
                .map(|(&p, _)| p)
                .collect();
            let t_vals: Vec<f64> = temp
                .iter()
                .zip(months.iter())
                .filter(|&(_, &mo)| mo == month_1)
                .map(|(&t, _)| t)
                .collect();

            monthly_mean_precip[m as usize] = zeus_stats::mean(&p_vals);
            monthly_mean_temp[m as usize] = zeus_stats::mean(&t_vals);
            monthly_sd_precip[m as usize] = zeus_stats::sd(&p_vals).max(sd_floor);
            monthly_sd_temp[m as usize] = zeus_stats::sd(&t_vals).max(sd_floor);
        }

        // 4. Build state thresholds from the subset.
        let thresholds = StateThresholds::from_baseline(&precip, &months, markov_config)?;

        // 5. Estimate monthly transitions with water-year filtering.
        let transitions = estimate_monthly_transitions(
            &precip,
            &months,
            &thresholds,
            markov_config,
            Some(&water_years),
        )?;

        // 6. Build subset-local month-day lookup.
        let mut subset_month_day: [Vec<usize>; 372] = std::array::from_fn(|_| Vec::new());
        for (i, (&m, &d)) in months.iter().zip(days.iter()).enumerate() {
            let slot = (m as usize - 1) * 31 + (d as usize - 1);
            subset_month_day[slot].push(i);
        }

        // 7. Compute KNN weights.
        let mut knn_weights = [[0.0_f64; 2]; 12];
        for m in 0..12 {
            knn_weights[m] = [
                config.precip_weight() / monthly_sd_precip[m],
                config.temp_weight() / monthly_sd_temp[m],
            ];
        }

        Ok(Self {
            subset_indices,
            precip,
            temp,
            months,
            water_years,
            thresholds,
            transitions,
            monthly_mean_precip,
            monthly_mean_temp,
            monthly_sd_precip,
            monthly_sd_temp,
            knn_weights,
            subset_month_day,
        })
    }

    /// Returns the number of days in the subset.
    pub(crate) fn len(&self) -> usize {
        self.subset_indices.len()
    }

    /// Returns the subset-local indices for a given month-day.
    pub(crate) fn month_day_candidates(&self, month: u8, day: u8) -> &[usize] {
        let slot = (month as usize - 1) * 31 + (day as usize - 1);
        &self.subset_month_day[slot]
    }

    /// Returns all subset-local indices for a given month.
    pub(crate) fn month_candidates(&self, month: u8) -> Vec<usize> {
        self.months
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m == month)
            .map(|(i, _)| i)
            .collect()
    }

    /// Maps a subset-local index to the global ObsData index.
    pub(crate) fn to_global(&self, local: usize) -> usize {
        self.subset_indices[local]
    }

    /// Returns the subset precipitation at a local index.
    pub(crate) fn precip_at(&self, local: usize) -> f64 {
        self.precip[local]
    }

    /// Returns the subset temperature at a local index.
    pub(crate) fn temp_at(&self, local: usize) -> f64 {
        self.temp[local]
    }

    /// Returns the subset month at a local index.
    pub(crate) fn month_at(&self, local: usize) -> u8 {
        self.months[local]
    }

    /// Returns the subset water year at a local index.
    pub(crate) fn water_year_at(&self, local: usize) -> i32 {
        self.water_years[local]
    }

    /// Classifies a precipitation value using the subset thresholds.
    pub(crate) fn classify(&self, precip: f64, month: u8) -> zeus_markov::PrecipState {
        self.thresholds.classify(precip, month)
    }

    /// Returns the monthly transitions.
    pub(crate) fn transitions(&self) -> &MonthlyTransitions {
        &self.transitions
    }

    /// Returns the mean precipitation for a 1-indexed month.
    pub(crate) fn mean_precip(&self, month: u8) -> f64 {
        self.monthly_mean_precip[(month - 1) as usize]
    }

    /// Returns the mean temperature for a 1-indexed month.
    pub(crate) fn mean_temp(&self, month: u8) -> f64 {
        self.monthly_mean_temp[(month - 1) as usize]
    }

    /// Returns the KNN weights for a 1-indexed month.
    pub(crate) fn knn_weights(&self, month: u8) -> &[f64; 2] {
        &self.knn_weights[(month - 1) as usize]
    }

    /// Returns the subset precipitation slice.
    pub(crate) fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Returns the subset temperature slice.
    pub(crate) fn temp(&self) -> &[f64] {
        &self.temp
    }

    /// Returns the subset water years slice.
    pub(crate) fn water_years(&self) -> &[i32] {
        &self.water_years
    }

    /// Returns the thresholds.
    pub(crate) fn thresholds(&self) -> &StateThresholds {
        &self.thresholds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeus_markov::ThresholdSpec;

    fn make_obs() -> ObsData {
        // 3 years x 365 days for a realistic dataset
        let days_per_year = 365;
        let n_years = 3usize;
        let n = n_years * days_per_year;
        let mut precip = Vec::with_capacity(n);
        let mut temp = Vec::with_capacity(n);
        let mut months_v = Vec::with_capacity(n);
        let mut days_v = Vec::with_capacity(n);
        let mut water_years = Vec::with_capacity(n);

        for y in 0..n_years {
            let wy = 2000 + y as i32;
            let mut doy = 0;
            for m in 1..=12u8 {
                let days_in_month: u8 = match m {
                    2 => 28,
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                for d in 1..=days_in_month {
                    if doy >= days_per_year {
                        break;
                    }
                    // ~40% dry
                    let p = if (doy * 7 + y * 13) % 10 < 4 {
                        0.0
                    } else {
                        ((doy as f64) * 0.1 + (y as f64) * 5.0).max(0.1)
                    };
                    precip.push(p);
                    temp.push(15.0 + (m as f64) * 2.0 + (y as f64) * 0.5);
                    months_v.push(m);
                    days_v.push(d);
                    water_years.push(wy);
                    doy += 1;
                }
            }
        }

        ObsData::new(&precip, &temp, &months_v, &days_v, &water_years).unwrap()
    }

    #[test]
    fn build_basic() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        // Select all 3 years
        let subset = YearSubset::build(&[0, 1, 2], &obs, &config, &markov_config).unwrap();
        assert_eq!(subset.len(), 3 * 365);
    }

    #[test]
    fn duplicate_years() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        // Select year 0 twice
        let subset = YearSubset::build(&[0, 0], &obs, &config, &markov_config).unwrap();
        assert_eq!(subset.len(), 2 * 365);
    }

    #[test]
    fn month_day_lookup() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2], &obs, &config, &markov_config).unwrap();
        // Jan 1 should have 3 entries (one per year)
        let jan1 = subset.month_day_candidates(1, 1);
        assert_eq!(jan1.len(), 3);
    }

    #[test]
    fn to_global_mapping() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0], &obs, &config, &markov_config).unwrap();
        // The first day of the subset should map to index 0 in ObsData
        assert_eq!(subset.to_global(0), 0);
        assert!((subset.precip_at(0) - obs.precip()[0]).abs() < 1e-10);
    }

    #[test]
    fn knn_weights_finite() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2], &obs, &config, &markov_config).unwrap();
        for m in 1..=12u8 {
            let w = subset.knn_weights(m);
            assert!(w[0].is_finite() && w[0] > 0.0, "month {m} precip weight");
            assert!(w[1].is_finite() && w[1] > 0.0, "month {m} temp weight");
        }
    }

    #[test]
    fn classify_uses_subset_thresholds() {
        let obs = make_obs();
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2], &obs, &config, &markov_config).unwrap();
        // 0.0 should always be Dry
        assert_eq!(subset.classify(0.0, 1), zeus_markov::PrecipState::Dry);
    }
}
