//! Multi-site observed data with grid metadata.

use std::collections::BTreeMap;

use crate::error::IoError;
use crate::observed::ObservedData;

// ---------------------------------------------------------------------------
// GridMetadata
// ---------------------------------------------------------------------------

/// Spatial metadata describing a collection of observation sites.
///
/// Stores per-cell longitude and latitude arrays whose lengths must be equal.
/// Construction fails with [`IoError::DimensionMismatch`] when the coordinate
/// arrays differ in length.
#[derive(Debug, Clone)]
pub struct GridMetadata {
    lons: Vec<f64>,
    lats: Vec<f64>,
}

impl GridMetadata {
    /// Create a new `GridMetadata` after validating coordinate array lengths.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::DimensionMismatch`] if `lons.len() != lats.len()`.
    pub fn new(lons: Vec<f64>, lats: Vec<f64>) -> Result<Self, IoError> {
        if lons.len() != lats.len() {
            return Err(IoError::DimensionMismatch {
                name: "latitude".into(),
                expected: lons.len(),
                got: lats.len(),
            });
        }

        Ok(Self { lons, lats })
    }

    /// Per-cell longitude values.
    pub fn lons(&self) -> &[f64] {
        &self.lons
    }

    /// Per-cell latitude values.
    pub fn lats(&self) -> &[f64] {
        &self.lats
    }

    /// Total number of grid cells.
    pub fn n_cells(&self) -> usize {
        self.lons.len()
    }
}

// ---------------------------------------------------------------------------
// MultiSiteData
// ---------------------------------------------------------------------------

/// Collection of observed data across multiple grid cells, with spatial metadata.
///
/// Each entry in the `sites` map corresponds to one grid cell. Construction
/// validates that the number of sites equals `grid.n_cells()` and that every
/// site has the same number of time steps.
#[derive(Debug, Clone)]
pub struct MultiSiteData {
    sites: BTreeMap<String, ObservedData>,
    grid: GridMetadata,
}

impl MultiSiteData {
    /// Create a new `MultiSiteData` after validating site count and timestep uniformity.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::DimensionMismatch`] if `sites.len() != grid.n_cells()`.
    ///
    /// Returns [`IoError::Validation`] if not all sites share the same number
    /// of time steps.
    pub fn new(sites: BTreeMap<String, ObservedData>, grid: GridMetadata) -> Result<Self, IoError> {
        // Check site count matches grid cell count.
        if sites.len() != grid.n_cells() {
            return Err(IoError::DimensionMismatch {
                name: "sites".into(),
                expected: grid.n_cells(),
                got: sites.len(),
            });
        }

        // Check that all sites have the same number of time steps.
        if let Some((first_key, first_obs)) = sites.iter().next() {
            let expected_len = first_obs.len();
            let mut mismatches = Vec::new();

            for (key, obs) in &sites {
                if key == first_key {
                    continue;
                }
                if obs.len() != expected_len {
                    mismatches.push(format!(
                        "site '{}' has {} timesteps (expected {} from '{}')",
                        key,
                        obs.len(),
                        expected_len,
                        first_key,
                    ));
                }
            }

            if !mismatches.is_empty() {
                return Err(IoError::Validation {
                    count: mismatches.len(),
                    details: mismatches.join("; "),
                });
            }
        }

        Ok(Self { sites, grid })
    }

    /// Look up a single site by key.
    pub fn get(&self, key: &str) -> Option<&ObservedData> {
        self.sites.get(key)
    }

    /// Reference to the full site map.
    pub fn sites(&self) -> &BTreeMap<String, ObservedData> {
        &self.sites
    }

    /// Reference to the grid metadata.
    pub fn grid(&self) -> &GridMetadata {
        &self.grid
    }

    /// Number of sites in the collection.
    pub fn n_sites(&self) -> usize {
        self.sites.len()
    }

    /// Number of time steps shared by all sites.
    ///
    /// Returns 0 when the collection is empty.
    pub fn n_timesteps(&self) -> usize {
        self.sites.values().next().map_or(0, |obs| obs.len())
    }

    /// Iterator over site keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.sites.keys()
    }

    /// Iterator over `(key, observed_data)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ObservedData)> {
        self.sites.iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zeus_calendar::NoLeapDate;

    /// Build an `ObservedData` with `n` rows of constant precipitation.
    fn make_observed(n: usize) -> ObservedData {
        let dates: Vec<NoLeapDate> = (0..n)
            .map(|i| {
                let day = (i % 28) as u8 + 1;
                let month = ((i / 28) % 12) as u8 + 1;
                NoLeapDate::new(2000, month, day).unwrap()
            })
            .collect();
        let precip = vec![1.0; n];
        ObservedData::new(precip, None, None, dates, 10).unwrap()
    }

    // -- GridMetadata -------------------------------------------------------

    #[test]
    fn grid_valid_construction() {
        let grid = GridMetadata::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        );
        assert!(grid.is_ok());

        let g = grid.unwrap();
        assert_eq!(g.n_cells(), 6);
        assert_eq!(g.lons(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(g.lats(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn grid_lon_length_mismatch() {
        let result = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            IoError::DimensionMismatch {
                name,
                expected,
                got,
            } => {
                assert_eq!(name, "latitude");
                assert_eq!(expected, 2);
                assert_eq!(got, 6);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn grid_lat_length_mismatch() {
        let result = GridMetadata::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0, 20.0, 30.0]);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            IoError::DimensionMismatch {
                name,
                expected,
                got,
            } => {
                assert_eq!(name, "latitude");
                assert_eq!(expected, 6);
                assert_eq!(got, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // -- MultiSiteData ------------------------------------------------------

    #[test]
    fn multi_site_valid_construction() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("site_a".to_string(), make_observed(30));
        sites.insert("site_b".to_string(), make_observed(30));

        let msd = MultiSiteData::new(sites, grid);
        assert!(msd.is_ok());

        let m = msd.unwrap();
        assert_eq!(m.n_sites(), 2);
        assert_eq!(m.n_timesteps(), 30);
    }

    #[test]
    fn multi_site_count_mismatch() {
        let grid = GridMetadata::new(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("site_a".to_string(), make_observed(30));
        sites.insert("site_b".to_string(), make_observed(30));
        // Only 2 sites, but grid expects 3.

        let result = MultiSiteData::new(sites, grid);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            IoError::DimensionMismatch {
                name,
                expected,
                got,
            } => {
                assert_eq!(name, "sites");
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn multi_site_non_uniform_timesteps() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("site_a".to_string(), make_observed(30));
        sites.insert("site_b".to_string(), make_observed(20)); // different length

        let result = MultiSiteData::new(sites, grid);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 1);
                assert!(details.contains("site_b"));
                assert!(details.contains("20"));
                assert!(details.contains("30"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    // -- Accessor tests -----------------------------------------------------

    #[test]
    fn multi_site_get() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("alpha".to_string(), make_observed(10));
        sites.insert("beta".to_string(), make_observed(10));

        let msd = MultiSiteData::new(sites, grid).unwrap();

        assert!(msd.get("alpha").is_some());
        assert!(msd.get("beta").is_some());
        assert!(msd.get("gamma").is_none());
    }

    #[test]
    fn multi_site_sites_accessor() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("a".to_string(), make_observed(5));
        sites.insert("b".to_string(), make_observed(5));

        let msd = MultiSiteData::new(sites, grid).unwrap();
        assert_eq!(msd.sites().len(), 2);
        assert!(msd.sites().contains_key("a"));
        assert!(msd.sites().contains_key("b"));
    }

    #[test]
    fn multi_site_grid_accessor() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("x".to_string(), make_observed(5));
        sites.insert("y".to_string(), make_observed(5));

        let msd = MultiSiteData::new(sites, grid).unwrap();
        assert_eq!(msd.grid().n_cells(), 2);
        assert_eq!(msd.grid().lons(), &[1.0, 2.0]);
        assert_eq!(msd.grid().lats(), &[10.0, 20.0]);
    }

    #[test]
    fn multi_site_keys_and_iter() {
        let grid = GridMetadata::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();

        let mut sites = BTreeMap::new();
        sites.insert("first".to_string(), make_observed(7));
        sites.insert("second".to_string(), make_observed(7));

        let msd = MultiSiteData::new(sites, grid).unwrap();

        // BTreeMap iterates in sorted order.
        let keys: Vec<&String> = msd.keys().collect();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], "first");
        assert_eq!(keys[1], "second");

        let pairs: Vec<(&String, &ObservedData)> = msd.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "first");
        assert_eq!(pairs[0].1.len(), 7);
        assert_eq!(pairs[1].0, "second");
        assert_eq!(pairs[1].1.len(), 7);
    }

    #[test]
    fn multi_site_n_timesteps_empty() {
        let grid = GridMetadata::new(vec![], vec![]).unwrap();
        let sites = BTreeMap::new();

        let msd = MultiSiteData::new(sites, grid).unwrap();
        assert_eq!(msd.n_timesteps(), 0);
        assert_eq!(msd.n_sites(), 0);
    }
}
