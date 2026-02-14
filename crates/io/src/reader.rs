//! High-level NetCDF reader configuration and orchestration.

use std::collections::BTreeMap;
use std::path::Path;

use zeus_calendar::water_year;

use crate::error::IoError;
use crate::multi_site::{GridMetadata, MultiSiteData};
use crate::netcdf_read;
use crate::observed::ObservedData;
use crate::owned_synthetic::OwnedSyntheticWeather;
use crate::parquet_read;

// ---------------------------------------------------------------------------
// ReaderConfig
// ---------------------------------------------------------------------------

/// Configuration for reading observed climate data from NetCDF files.
///
/// Use the builder methods (`with_*`) to customise variable names,
/// coordinate aliases, and water-year trimming behaviour.  The
/// [`Default`] implementation supplies CF-convention names suitable for
/// CMIP-style climate data.
#[derive(Debug, Clone)]
pub struct ReaderConfig {
    /// NetCDF variable name for precipitation.
    precip_var: String,
    /// Optional NetCDF variable name for daily maximum temperature.
    temp_max_var: Option<String>,
    /// Optional NetCDF variable name for daily minimum temperature.
    temp_min_var: Option<String>,
    /// Aliases to try when looking up longitude coordinates.
    lon_aliases: Vec<String>,
    /// Aliases to try when looking up latitude coordinates.
    lat_aliases: Vec<String>,
    /// NetCDF variable name for the time axis.
    time_var: String,
    /// First month of the water year (1 = January, 10 = October, etc.).
    start_month: u8,
    /// Whether to trim the time series to complete water years.
    trim_to_water_years: bool,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            precip_var: "pr".into(),
            temp_max_var: Some("tasmax".into()),
            temp_min_var: Some("tasmin".into()),
            lon_aliases: vec!["lon".into(), "longitude".into(), "x".into()],
            lat_aliases: vec!["lat".into(), "latitude".into(), "y".into()],
            time_var: "time".into(),
            start_month: 10,
            trim_to_water_years: true,
        }
    }
}

impl ReaderConfig {
    /// Set the precipitation variable name.
    pub fn with_precip_var(mut self, name: impl Into<String>) -> Self {
        self.precip_var = name.into();
        self
    }

    /// Set the maximum temperature variable name, or `None` to skip it.
    pub fn with_temp_max_var(mut self, name: Option<impl Into<String>>) -> Self {
        self.temp_max_var = name.map(Into::into);
        self
    }

    /// Set the minimum temperature variable name, or `None` to skip it.
    pub fn with_temp_min_var(mut self, name: Option<impl Into<String>>) -> Self {
        self.temp_min_var = name.map(Into::into);
        self
    }

    /// Set the first month of the water year.
    pub fn with_start_month(mut self, month: u8) -> Self {
        self.start_month = month;
        self
    }

    /// Enable or disable trimming to complete water years.
    pub fn with_trim_to_water_years(mut self, trim: bool) -> Self {
        self.trim_to_water_years = trim;
        self
    }

    /// Validate that the configuration is internally consistent.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if `start_month` is outside 1..=12.
    pub fn validate(&self) -> Result<(), IoError> {
        if !(1..=12).contains(&self.start_month) {
            return Err(IoError::Validation {
                count: 1,
                details: format!("start_month must be 1..=12, got {}", self.start_month),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// read_netcdf
// ---------------------------------------------------------------------------

/// Read observed climate data from a NetCDF file.
///
/// The file must contain a 3-D precipitation variable (`time x y x`), optional
/// temperature variables, and coordinate arrays.  The returned
/// [`MultiSiteData`] contains one [`ObservedData`] per grid cell, keyed as
/// `"cell_NNNN"`.
///
/// # Errors
///
/// Returns [`IoError`] on missing variables, dimension mismatches, calendar
/// conversion failures, or validation problems.
pub fn read_netcdf(path: &Path, config: &ReaderConfig) -> Result<MultiSiteData, IoError> {
    config.validate()?;

    let file = netcdf_read::open_file(path)?;

    // -- Coordinates --------------------------------------------------------

    let lon_alias_refs: Vec<&str> = config.lon_aliases.iter().map(String::as_str).collect();
    let lat_alias_refs: Vec<&str> = config.lat_aliases.iter().map(String::as_str).collect();

    let lons = netcdf_read::read_1d_f64(&file, &lon_alias_refs, path)?;
    let lats = netcdf_read::read_1d_f64(&file, &lat_alias_refs, path)?;

    // -- Time ---------------------------------------------------------------

    let time_offsets = netcdf_read::read_1d_f64(&file, &[&config.time_var], path)?;
    let (calendar, base_date) = netcdf_read::read_time_units(&file, &config.time_var, path)?;
    let dates = netcdf_read::time_offsets_to_dates(base_date, &time_offsets, &calendar)?;

    // -- 3-D data -----------------------------------------------------------

    let (precip_data, [_nt, ny, nx]) = netcdf_read::read_3d_f64(&file, &config.precip_var, path)?;

    let tmax_data = match &config.temp_max_var {
        Some(name) => Some(netcdf_read::read_3d_f64(&file, name, path)?),
        None => None,
    };

    let tmin_data = match &config.temp_min_var {
        Some(name) => Some(netcdf_read::read_3d_f64(&file, name, path)?),
        None => None,
    };

    let n_cells = nx * ny;

    // -- Determine trim range -----------------------------------------------

    let (start_idx, end_idx) = if config.trim_to_water_years {
        trim_range(&dates, config.start_month)?
    } else {
        (0, dates.len())
    };

    let trimmed_dates = &dates[start_idx..end_idx];

    // -- Reshape per cell ---------------------------------------------------

    let mut sites = BTreeMap::new();

    for c in 0..n_cells {
        let mut cell_precip = Vec::with_capacity(end_idx - start_idx);
        let mut cell_tmax: Option<Vec<f64>> = tmax_data
            .as_ref()
            .map(|_| Vec::with_capacity(end_idx - start_idx));
        let mut cell_tmin: Option<Vec<f64>> = tmin_data
            .as_ref()
            .map(|_| Vec::with_capacity(end_idx - start_idx));

        for t in start_idx..end_idx {
            let flat_idx = t * n_cells + c;
            cell_precip.push(precip_data[flat_idx]);

            if let (Some(v), Some((data, _))) = (&mut cell_tmax, &tmax_data) {
                v.push(data[flat_idx]);
            }
            if let (Some(v), Some((data, _))) = (&mut cell_tmin, &tmin_data) {
                v.push(data[flat_idx]);
            }
        }

        let obs = ObservedData::new(
            cell_precip,
            cell_tmax,
            cell_tmin,
            trimmed_dates.to_vec(),
            config.start_month,
        )?;

        let key = format!("cell_{c:04}");
        sites.insert(key, obs);
    }

    // -- Grid metadata ------------------------------------------------------

    // Expand 1-D lon/lat to per-cell arrays if they are 1-D axis arrays.
    let (cell_lons, cell_lats) = if lons.len() == nx && lats.len() == ny {
        // 1-D axis arrays: broadcast to a flat grid (row-major: y varies slowest).
        let mut flat_lons = Vec::with_capacity(n_cells);
        let mut flat_lats = Vec::with_capacity(n_cells);
        for lat in &lats {
            for lon in &lons {
                flat_lons.push(*lon);
                flat_lats.push(*lat);
            }
        }
        (flat_lons, flat_lats)
    } else {
        // Already per-cell (or some other layout); pass through directly.
        (lons, lats)
    };

    let grid = GridMetadata::new(cell_lons, cell_lats, nx, ny)?;
    MultiSiteData::new(sites, grid)
}

/// Find the first and one-past-last indices that span only complete water years.
///
/// A complete water year begins on `start_month` day 1 and ends the day
/// before the next occurrence of `start_month` day 1.
fn trim_range(
    dates: &[zeus_calendar::NoLeapDate],
    start_month: u8,
) -> Result<(usize, usize), IoError> {
    if dates.is_empty() {
        return Ok((0, 0));
    }

    // Find the first index whose month == start_month and day == 1.
    let first = dates
        .iter()
        .position(|d| d.month() == start_month && d.day() == 1)
        .unwrap_or(0);

    // Compute the end month: the last month of the water year.
    let end_month = if start_month == 1 {
        12
    } else {
        start_month - 1
    };

    // Find the last index whose month == end_month by scanning from the end.
    // We want one past the last element belonging to a complete water year.
    let last_in_end_month = dates.iter().rposition(|d| d.month() == end_month);

    let end = match last_in_end_month {
        Some(idx) if idx >= first => idx + 1,
        _ => dates.len(),
    };

    // Verify we actually have at least one complete water year.
    if end <= first {
        return Ok((0, dates.len()));
    }

    // Extra check: make sure the first/last water years are actually complete
    // by verifying the water year boundaries include a full cycle.
    let first_wy = water_year(dates[first].year(), dates[first].month(), start_month)
        .map_err(IoError::from)?;
    let last_wy = water_year(dates[end - 1].year(), dates[end - 1].month(), start_month)
        .map_err(IoError::from)?;

    if last_wy < first_wy {
        return Ok((0, dates.len()));
    }

    Ok((first, end))
}

// ---------------------------------------------------------------------------
// read_parquet
// ---------------------------------------------------------------------------

/// Read synthetic weather data from a Parquet file.
///
/// Returns one [`OwnedSyntheticWeather`] per realisation found in the file,
/// sorted by realisation index.
///
/// # Errors
///
/// Returns [`IoError::FileNotFound`] if the file does not exist, or
/// [`IoError::Parquet`] / [`IoError::Validation`] on format errors.
pub fn read_parquet(path: &Path) -> Result<Vec<OwnedSyntheticWeather>, IoError> {
    let batches = parquet_read::read_batches(path)?;
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let has_temp = parquet_read::validate_schema(&batches[0])?;
    parquet_read::group_by_realisation(&batches, has_temp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = ReaderConfig::default();
        assert_eq!(cfg.precip_var, "pr");
        assert_eq!(cfg.temp_max_var.as_deref(), Some("tasmax"));
        assert_eq!(cfg.temp_min_var.as_deref(), Some("tasmin"));
        assert_eq!(cfg.lon_aliases, vec!["lon", "longitude", "x"]);
        assert_eq!(cfg.lat_aliases, vec!["lat", "latitude", "y"]);
        assert_eq!(cfg.time_var, "time");
        assert_eq!(cfg.start_month, 10);
        assert!(cfg.trim_to_water_years);
    }

    #[test]
    fn builder_methods() {
        let cfg = ReaderConfig::default()
            .with_precip_var("precipitation")
            .with_temp_max_var(Some("tmax"))
            .with_temp_min_var(None::<String>)
            .with_start_month(1)
            .with_trim_to_water_years(false);

        assert_eq!(cfg.precip_var, "precipitation");
        assert_eq!(cfg.temp_max_var.as_deref(), Some("tmax"));
        assert!(cfg.temp_min_var.is_none());
        assert_eq!(cfg.start_month, 1);
        assert!(!cfg.trim_to_water_years);
    }

    #[test]
    fn validate_valid_start_month() {
        for m in 1..=12 {
            let cfg = ReaderConfig::default().with_start_month(m);
            assert!(cfg.validate().is_ok(), "month {m} should be valid");
        }
    }

    #[test]
    fn validate_invalid_start_month_zero() {
        let cfg = ReaderConfig::default().with_start_month(0);
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, IoError::Validation { .. }));
    }

    #[test]
    fn validate_invalid_start_month_thirteen() {
        let cfg = ReaderConfig::default().with_start_month(13);
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, IoError::Validation { .. }));
    }

    #[test]
    fn trim_range_empty() {
        let (start, end) = trim_range(&[], 10).expect("should succeed");
        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }

    #[test]
    fn trim_range_finds_boundaries() {
        // Build a date sequence spanning Sep 15 2000 to Nov 15 2001
        // (covers water year starting Oct 2000).
        let start_date = zeus_calendar::NoLeapDate::new(2000, 9, 15).unwrap();
        let mut dates = Vec::new();
        let mut d = start_date;
        for _ in 0..430 {
            dates.push(d);
            d = d.next();
        }

        let (s, e) = trim_range(&dates, 10).expect("should succeed");

        // First Oct 1 is at day index 16 (Sep has 30 days, so Sep 15 + 16 = Oct 1)
        assert_eq!(dates[s].month(), 10);
        assert_eq!(dates[s].day(), 1);

        // End should be after last September day
        assert_eq!(dates[e - 1].month(), 9);
    }
}
