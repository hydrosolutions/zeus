//! Multi-site synthetic data wrapper.

use crate::error::EvaluateError;
use std::collections::BTreeMap;
use zeus_io::SyntheticWeather;

/// Multi-site synthetic weather wrapper with validation.
#[derive(Debug)]
pub struct MultiSiteSynthetic<'a> {
    sites: BTreeMap<String, Vec<SyntheticWeather<'a>>>,
}

impl<'a> MultiSiteSynthetic<'a> {
    /// Create a new multi-site synthetic wrapper.
    ///
    /// # Validation
    ///
    /// - At least one site with at least one realisation
    /// - All sites must have the same number of realisations
    /// - All realisations at a site must have the same length
    pub fn new(sites: BTreeMap<String, Vec<SyntheticWeather<'a>>>) -> Result<Self, EvaluateError> {
        let mut errors = Vec::new();

        // Check non-empty
        if sites.is_empty() {
            errors.push("no sites provided".to_string());
        } else {
            // Check at least one realisation per site
            for (site, realisations) in &sites {
                if realisations.is_empty() {
                    errors.push(format!("site '{}' has no realisations", site));
                }
            }

            // Check all sites have same number of realisations
            if let Some((_, first_realisations)) = sites.iter().next() {
                let expected_count = first_realisations.len();
                for (site, realisations) in &sites {
                    if realisations.len() != expected_count {
                        errors.push(format!(
                            "site '{}' has {} realisations, expected {}",
                            site,
                            realisations.len(),
                            expected_count
                        ));
                    }
                }
            }

            // Check all realisations at each site have same length
            for (site, realisations) in &sites {
                if let Some(first) = realisations.first() {
                    let expected_len = first.len();
                    for (idx, realisation) in realisations.iter().enumerate() {
                        if realisation.len() != expected_len {
                            errors.push(format!(
                                "site '{}' realisation {} has length {}, expected {}",
                                site,
                                idx,
                                realisation.len(),
                                expected_len
                            ));
                        }
                    }
                }
            }
        }

        if !errors.is_empty() {
            return Err(EvaluateError::Validation {
                count: errors.len(),
                details: errors.join("; "),
            });
        }

        Ok(Self { sites })
    }

    /// Returns a reference to all sites.
    pub fn sites(&self) -> &BTreeMap<String, Vec<SyntheticWeather<'a>>> {
        &self.sites
    }

    /// Get realisations for a specific site.
    pub fn get(&self, key: &str) -> Option<&Vec<SyntheticWeather<'a>>> {
        self.sites.get(key)
    }

    /// Returns the number of sites.
    pub fn n_sites(&self) -> usize {
        self.sites.len()
    }

    /// Returns the number of realisations (same for all sites).
    pub fn n_realisations(&self) -> usize {
        self.sites.values().next().map(|v| v.len()).unwrap_or(0)
    }

    /// Returns an iterator over site names.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.sites.keys()
    }

    /// Returns an iterator over (site name, realisations) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<SyntheticWeather<'a>>)> {
        self.sites.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_weather(
        n_days: usize,
        realisation: u32,
    ) -> (
        Vec<f64>,
        Vec<u8>,
        Vec<i32>,
        Vec<u16>,
        SyntheticWeather<'static>,
    ) {
        let precip: Vec<f64> = (0..n_days).map(|i| i as f64).collect();
        let months: Vec<u8> = (0..n_days).map(|_| 1).collect();
        let water_years: Vec<i32> = (0..n_days).map(|_| 2000).collect();
        let days: Vec<u16> = (0..n_days).map(|i| (i + 1) as u16).collect();

        // Leak the vectors to get 'static lifetime for test purposes
        let precip_static: &'static [f64] = Box::leak(precip.into_boxed_slice());
        let months_static: &'static [u8] = Box::leak(months.into_boxed_slice());
        let water_years_static: &'static [i32] = Box::leak(water_years.into_boxed_slice());
        let days_static: &'static [u16] = Box::leak(days.into_boxed_slice());

        let sw = SyntheticWeather::new(
            precip_static,
            None,
            None,
            months_static,
            water_years_static,
            days_static,
            realisation,
        )
        .unwrap();

        (
            Vec::new(), // dummy
            Vec::new(), // dummy
            Vec::new(), // dummy
            Vec::new(), // dummy
            sw,
        )
    }

    #[test]
    fn test_valid_construction() {
        let (_, _, _, _, sw1) = make_synthetic_weather(10, 0);
        let (_, _, _, _, sw2) = make_synthetic_weather(10, 1);
        let (_, _, _, _, sw3) = make_synthetic_weather(10, 0);
        let (_, _, _, _, sw4) = make_synthetic_weather(10, 1);

        let mut sites = BTreeMap::new();
        sites.insert("site1".to_string(), vec![sw1, sw2]);
        sites.insert("site2".to_string(), vec![sw3, sw4]);

        let multi = MultiSiteSynthetic::new(sites).unwrap();

        assert_eq!(multi.n_sites(), 2);
        assert_eq!(multi.n_realisations(), 2);
        assert!(multi.get("site1").is_some());
        assert!(multi.get("site2").is_some());
        assert!(multi.get("site3").is_none());

        let keys: Vec<_> = multi.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&&"site1".to_string()));
        assert!(keys.contains(&&"site2".to_string()));

        let iter_count = multi.iter().count();
        assert_eq!(iter_count, 2);
    }

    #[test]
    fn test_empty_sites_fails() {
        let sites = BTreeMap::new();
        let result = MultiSiteSynthetic::new(sites);
        assert!(result.is_err());

        if let Err(EvaluateError::Validation { count, details }) = result {
            assert_eq!(count, 1);
            assert!(details.contains("no sites provided"));
        } else {
            panic!("expected validation error");
        }
    }

    #[test]
    fn test_mismatched_realisations_fails() {
        let (_, _, _, _, sw1) = make_synthetic_weather(10, 0);
        let (_, _, _, _, sw2) = make_synthetic_weather(10, 1);
        let (_, _, _, _, sw3) = make_synthetic_weather(10, 0);

        let mut sites = BTreeMap::new();
        sites.insert("site1".to_string(), vec![sw1, sw2]);
        sites.insert("site2".to_string(), vec![sw3]); // Only 1 realisation

        let result = MultiSiteSynthetic::new(sites);
        assert!(result.is_err());

        if let Err(EvaluateError::Validation { count, details }) = result {
            assert_eq!(count, 1);
            assert!(details.contains("site2"));
            assert!(details.contains("has 1 realisations"));
        } else {
            panic!("expected validation error");
        }
    }

    #[test]
    fn test_single_site() {
        let (_, _, _, _, sw1) = make_synthetic_weather(10, 0);
        let (_, _, _, _, sw2) = make_synthetic_weather(10, 1);

        let mut sites = BTreeMap::new();
        sites.insert("only_site".to_string(), vec![sw1, sw2]);

        let multi = MultiSiteSynthetic::new(sites).unwrap();

        assert_eq!(multi.n_sites(), 1);
        assert_eq!(multi.n_realisations(), 2);
        assert!(multi.get("only_site").is_some());
    }
}
