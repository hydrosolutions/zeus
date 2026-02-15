//! Integration test: NetCDF reader config and file-not-found handling.

use std::path::Path;

use zeus_io::{IoError, ReaderConfig, read_netcdf};

#[test]
fn read_netcdf_file_not_found() {
    let path = Path::new("/tmp/zeus_test_nonexistent_file.nc");
    let config = ReaderConfig::default();

    let result = read_netcdf(path, &config);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(
        matches!(err, IoError::FileNotFound { .. }),
        "expected FileNotFound, got {err:?}",
    );
}

#[test]
fn read_netcdf_invalid_config_rejects_early() {
    let path = Path::new("/tmp/zeus_test_nonexistent_file.nc");
    let config = ReaderConfig::default().with_start_month(0);

    let result = read_netcdf(path, &config);
    assert!(result.is_err());

    // Should fail on config validation before even trying to open the file.
    let err = result.unwrap_err();
    assert!(
        matches!(err, IoError::Validation { .. }),
        "expected Validation error, got {err:?}",
    );
}
