//! Integration tests for NetCDF cell-skipping behaviour.
//!
//! Validates that `read_netcdf` correctly skips grid cells where all timesteps
//! contain NaN or fill-value data, while retaining cells with any valid data.

use std::path::Path;

use tempfile::tempdir;
use zeus_io::{IoError, ReaderConfig, read_netcdf};

// ---------------------------------------------------------------------------
// Helper: programmatic NetCDF fixture builder
// ---------------------------------------------------------------------------

/// Configuration for building a minimal NetCDF test fixture.
struct FixtureBuilder {
    nx: usize,
    ny: usize,
    nt: usize,
    lons: Vec<f64>,
    lats: Vec<f64>,
    /// Flat precipitation data in `[t, lat, lon]` order (length = nt * ny * nx).
    precip: Vec<f64>,
    /// Optional flat tasmax data.
    tasmax: Option<Vec<f64>>,
    /// Optional `_FillValue` for the `pr` variable.
    pr_fill_value: Option<f64>,
    /// Optional `_FillValue` for the `tasmax` variable.
    tasmax_fill_value: Option<f64>,
}

impl FixtureBuilder {
    /// Create a new builder with all-valid precip data.
    fn new(nx: usize, ny: usize, nt: usize) -> Self {
        let n_cells = nx * ny;
        let lons: Vec<f64> = (0..nx).map(|i| -120.0 + i as f64).collect();
        let lats: Vec<f64> = (0..ny).map(|i| 40.0 + i as f64).collect();
        let precip: Vec<f64> = (0..nt * n_cells).map(|i| (i % 10) as f64 * 0.1).collect();

        Self {
            nx,
            ny,
            nt,
            lons,
            lats,
            precip,
            tasmax: None,
            pr_fill_value: None,
            tasmax_fill_value: None,
        }
    }

    /// Set specific lon values.
    fn with_lons(mut self, lons: Vec<f64>) -> Self {
        assert_eq!(lons.len(), self.nx);
        self.lons = lons;
        self
    }

    /// Set specific lat values.
    fn with_lats(mut self, lats: Vec<f64>) -> Self {
        assert_eq!(lats.len(), self.ny);
        self.lats = lats;
        self
    }

    /// Replace precip data entirely.
    fn with_precip(mut self, precip: Vec<f64>) -> Self {
        assert_eq!(precip.len(), self.nt * self.nx * self.ny);
        self.precip = precip;
        self
    }

    /// Set a cell's precip data across all timesteps to a constant value.
    fn with_cell_precip_const(mut self, cell: usize, value: f64) -> Self {
        let n_cells = self.nx * self.ny;
        for t in 0..self.nt {
            self.precip[t * n_cells + cell] = value;
        }
        self
    }

    /// Set a `_FillValue` attribute on the `pr` variable.
    fn with_pr_fill_value(mut self, fv: f64) -> Self {
        self.pr_fill_value = Some(fv);
        self
    }

    /// Add a tasmax variable with all-valid data.
    fn with_tasmax(mut self) -> Self {
        let n = self.nt * self.nx * self.ny;
        self.tasmax = Some((0..n).map(|i| 20.0 + (i % 5) as f64).collect());
        self
    }

    /// Set a cell's tasmax data across all timesteps to a constant value.
    fn with_cell_tasmax_const(mut self, cell: usize, value: f64) -> Self {
        let n_cells = self.nx * self.ny;
        let data = self
            .tasmax
            .as_mut()
            .expect("call with_tasmax() before with_cell_tasmax_const()");
        for t in 0..self.nt {
            data[t * n_cells + cell] = value;
        }
        self
    }

    /// Set a `_FillValue` attribute on the `tasmax` variable.
    #[allow(dead_code)]
    fn with_tasmax_fill_value(mut self, fv: f64) -> Self {
        self.tasmax_fill_value = Some(fv);
        self
    }

    /// Write the fixture to a NetCDF file and return the path.
    fn write(&self, dir: &Path) -> std::path::PathBuf {
        let path = dir.join("test.nc");
        let mut file = netcdf::create(&path).expect("failed to create NetCDF file");

        // Dimensions.
        file.add_dimension("time", self.nt).expect("add dim time");
        file.add_dimension("lat", self.ny).expect("add dim lat");
        file.add_dimension("lon", self.nx).expect("add dim lon");

        // Coordinate variables.
        {
            let mut var = file
                .add_variable::<f64>("lon", &["lon"])
                .expect("add var lon");
            var.put_values(&self.lons, ..).expect("put lon values");
        }
        {
            let mut var = file
                .add_variable::<f64>("lat", &["lat"])
                .expect("add var lat");
            var.put_values(&self.lats, ..).expect("put lat values");
        }

        // Time variable.
        {
            let time_vals: Vec<f64> = (0..self.nt).map(|t| t as f64).collect();
            let mut var = file
                .add_variable::<f64>("time", &["time"])
                .expect("add var time");
            var.put_values(&time_vals, ..).expect("put time values");
            var.put_attribute("units", "days since 2000-01-01")
                .expect("add time units");
            var.put_attribute("calendar", "noleap")
                .expect("add time calendar");
        }

        // Precipitation variable.
        {
            let mut var = file
                .add_variable::<f64>("pr", &["time", "lat", "lon"])
                .expect("add var pr");
            if let Some(fv) = self.pr_fill_value {
                var.put_attribute("_FillValue", fv)
                    .expect("add pr _FillValue");
            }
            var.put_values(&self.precip, ..).expect("put precip values");
        }

        // Optional tasmax variable.
        if let Some(data) = &self.tasmax {
            let mut var = file
                .add_variable::<f64>("tasmax", &["time", "lat", "lon"])
                .expect("add var tasmax");
            if let Some(fv) = self.tasmax_fill_value {
                var.put_attribute("_FillValue", fv)
                    .expect("add tasmax _FillValue");
            }
            var.put_values(data, ..).expect("put tasmax values");
        }

        path
    }
}

/// Simple reader config that disables water-year trimming and temperature
/// variables so tests only exercise precipitation-based cell skipping.
fn precip_only_config() -> ReaderConfig {
    ReaderConfig::default()
        .with_trim_to_water_years(false)
        .with_temp_max_var(None::<String>)
        .with_temp_min_var(None::<String>)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn all_cells_valid() {
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10).write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(data.n_sites(), 4);
    assert_eq!(data.grid().n_cells(), 4);
}

#[test]
fn one_cell_all_nan() {
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        .with_cell_precip_const(1, f64::NAN)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(data.n_sites(), 3);
    assert_eq!(data.grid().n_cells(), 3);
}

#[test]
fn one_cell_all_fill_value() {
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        .with_pr_fill_value(-9999.0)
        .with_cell_precip_const(1, -9999.0)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(
        data.n_sites(),
        3,
        "cell with all fill-values should be skipped"
    );
    assert_eq!(data.grid().n_cells(), 3);
}

#[test]
fn boundary_cells_nan() {
    // 3x3 grid, corners (cells 0, 2, 6, 8) are all NaN.
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(3, 3, 10)
        .with_cell_precip_const(0, f64::NAN)
        .with_cell_precip_const(2, f64::NAN)
        .with_cell_precip_const(6, f64::NAN)
        .with_cell_precip_const(8, f64::NAN)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(
        data.n_sites(),
        5,
        "4 corner cells should be skipped, leaving 5"
    );
    assert_eq!(data.grid().n_cells(), 5);
}

#[test]
fn all_cells_all_nan() {
    // Every cell is all NaN -- reader should return a Validation error.
    let n_cells = 4;
    let nt = 10;
    let precip = vec![f64::NAN; nt * n_cells];

    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, nt)
        .with_precip(precip)
        .write(dir.path());

    let result = read_netcdf(&path, &precip_only_config());
    assert!(result.is_err(), "all-NaN grid should produce an error");

    let err = result.unwrap_err();
    assert!(
        matches!(err, IoError::Validation { .. }),
        "expected IoError::Validation, got {err:?}",
    );
}

#[test]
fn partial_nan_not_skipped() {
    // Cell 1 has NaN at some timesteps but not all -- should NOT be skipped.
    let dir = tempdir().unwrap();
    let nx = 2;
    let ny = 2;
    let nt = 10;
    let n_cells = nx * ny;

    let mut precip = vec![1.0; nt * n_cells];
    // Set some (but not all) timesteps of cell 1 to NaN.
    for t in 0..5 {
        precip[t * n_cells + 1] = f64::NAN;
    }
    // Timesteps 5..10 remain valid for cell 1.

    let path = FixtureBuilder::new(nx, ny, nt)
        .with_precip(precip)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(data.n_sites(), 4, "partial NaN cell should NOT be skipped");
}

#[test]
fn tmax_all_nan_skips_cell() {
    // Cell 1 has valid precip but all-NaN tasmax. With tasmax configured,
    // the cell should be skipped.
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        .with_tasmax()
        .with_cell_tasmax_const(1, f64::NAN)
        .write(dir.path());

    let config = ReaderConfig::default()
        .with_trim_to_water_years(false)
        .with_temp_min_var(None::<String>);

    let data = read_netcdf(&path, &config).unwrap();
    assert_eq!(
        data.n_sites(),
        3,
        "cell with all-NaN tasmax should be skipped"
    );
}

#[test]
fn no_fill_attr_nan_in_data() {
    // No `_FillValue` attribute, but data contains NaN. Should still skip.
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        // No pr_fill_value set, but cell 1 is all NaN.
        .with_cell_precip_const(1, f64::NAN)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(
        data.n_sites(),
        3,
        "NaN cells should be skipped even without _FillValue attribute"
    );
}

#[test]
fn fill_value_large_positive() {
    // `_FillValue=1e20`, cell 1 data is all 1e20. Should be replaced with NaN
    // and then skipped.
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        .with_pr_fill_value(1e20)
        .with_cell_precip_const(1, 1e20)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(
        data.n_sites(),
        3,
        "cell with all fill-value (1e20) should be skipped"
    );
}

#[test]
fn skipped_cells_correct_coordinates() {
    // 2x2 grid with known lon/lat. Cell 1 is all-NaN.
    // Grid layout (row-major, lat slowest):
    //   cell 0 = (lon=-100, lat=35)
    //   cell 1 = (lon=-99,  lat=35)  <-- skipped
    //   cell 2 = (lon=-100, lat=36)
    //   cell 3 = (lon=-99,  lat=36)
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(2, 2, 10)
        .with_lons(vec![-100.0, -99.0])
        .with_lats(vec![35.0, 36.0])
        .with_cell_precip_const(1, f64::NAN)
        .write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(data.n_sites(), 3);

    let grid = data.grid();
    assert_eq!(grid.n_cells(), 3);

    // Surviving cells should be 0, 2, 3 with coordinates:
    //   cell 0: lon=-100, lat=35
    //   cell 2: lon=-100, lat=36
    //   cell 3: lon=-99,  lat=36
    let lons = grid.lons();
    let lats = grid.lats();

    assert_eq!(lons[0], -100.0);
    assert_eq!(lats[0], 35.0);

    assert_eq!(lons[1], -100.0);
    assert_eq!(lats[1], 36.0);

    assert_eq!(lons[2], -99.0);
    assert_eq!(lats[2], 36.0);
}

#[test]
fn single_valid_cell() {
    // 1x1 grid with valid data should return 1 site.
    let dir = tempdir().unwrap();
    let path = FixtureBuilder::new(1, 1, 10).write(dir.path());

    let data = read_netcdf(&path, &precip_only_config()).unwrap();
    assert_eq!(data.n_sites(), 1);
    assert_eq!(data.grid().n_cells(), 1);
}
