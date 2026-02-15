//! Annual KNN year selection.

use crate::config::ResampleConfig;
use crate::error::ResampleError;
use crate::obs_data::ObsData;
use zeus_knn::{KnnConfig, knn_sample};

/// Selects historical years by annual KNN on observed annual precipitation.
///
/// Returns indices into `obs.unique_water_years()` / `obs.annual_precip()`.
/// May contain duplicates (with-replacement sampling).
#[tracing::instrument(skip(obs, config, rng), fields(target = sim_precip))]
#[allow(dead_code)]
pub(crate) fn select_annual_years(
    sim_precip: f64,
    obs: &ObsData,
    config: &ResampleConfig,
    rng: &mut impl rand::Rng,
) -> Result<Vec<usize>, ResampleError> {
    let n_years = obs.n_obs_years();
    let k = ((n_years as f64).sqrt().ceil() as usize).max(1);
    let knn_config = KnnConfig::new(k)
        .with_n(config.annual_knn_n())
        .with_sampling(config.sampling().clone())
        .with_epsilon(config.epsilon());

    let result = knn_sample(
        obs.annual_precip(),
        1,
        &[sim_precip],
        &[1.0],
        &knn_config,
        rng,
    )?;

    Ok(result.indices().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_obs(n_years: usize) -> ObsData {
        let days_per_year = 365;
        let n = n_years * days_per_year;
        let mut precip = Vec::with_capacity(n);
        let mut temp = Vec::with_capacity(n);
        let mut months = Vec::with_capacity(n);
        let mut days = Vec::with_capacity(n);
        let mut water_years = Vec::with_capacity(n);

        for y in 0..n_years {
            let wy = 2000 + y as i32;
            let mut doy = 0;
            for m in 1..=12u8 {
                let days_in_month = match m {
                    2 => 28,
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                for d in 1..=days_in_month {
                    if doy >= days_per_year {
                        break;
                    }
                    precip.push((y as f64) * 0.1 + (doy as f64) * 0.01);
                    temp.push(15.0 + (m as f64) * 2.0);
                    months.push(m);
                    days.push(d as u8);
                    water_years.push(wy);
                    doy += 1;
                }
            }
        }

        ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap()
    }

    #[test]
    fn returns_correct_count() {
        let obs = make_obs(10);
        let config = ResampleConfig::new().with_annual_knn_n(50);
        let mut rng = StdRng::seed_from_u64(42);
        let years = select_annual_years(500.0, &obs, &config, &mut rng).unwrap();
        assert_eq!(years.len(), 50);
    }

    #[test]
    fn indices_in_range() {
        let obs = make_obs(10);
        let config = ResampleConfig::new();
        let mut rng = StdRng::seed_from_u64(42);
        let years = select_annual_years(500.0, &obs, &config, &mut rng).unwrap();
        for &idx in &years {
            assert!(idx < obs.n_obs_years());
        }
    }

    #[test]
    fn reproducible() {
        let obs = make_obs(10);
        let config = ResampleConfig::new();
        let mut rng1 = StdRng::seed_from_u64(99);
        let y1 = select_annual_years(500.0, &obs, &config, &mut rng1).unwrap();
        let mut rng2 = StdRng::seed_from_u64(99);
        let y2 = select_annual_years(500.0, &obs, &config, &mut rng2).unwrap();
        assert_eq!(y1, y2);
    }
}
