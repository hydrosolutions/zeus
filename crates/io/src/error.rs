//! Error types for zeus-io.

use std::path::PathBuf;

/// Error type for all fallible operations in the zeus-io crate.
///
/// This enum covers I/O failures, format-specific errors from NetCDF and
/// Parquet, calendar conversion issues, validation problems, and data-model
/// mismatches encountered when reading or writing climate files.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Returned when a required file does not exist on disk.
    #[error("file not found: {}", path.display())]
    FileNotFound {
        /// Path that could not be found.
        path: PathBuf,
    },

    /// Wraps an error originating from the NetCDF library.
    #[error("netcdf error: {reason}")]
    Netcdf {
        /// Description of the underlying NetCDF failure.
        reason: String,
    },

    /// Wraps an error originating from the Parquet library.
    #[error("parquet error: {reason}")]
    Parquet {
        /// Description of the underlying Parquet failure.
        reason: String,
    },

    /// Wraps an error originating from the zeus-calendar crate.
    #[error("calendar error: {reason}")]
    Calendar {
        /// Description of the underlying calendar failure.
        reason: String,
    },

    /// Returned when one or more validation checks fail.
    #[error("{count} validation error(s): {details}")]
    Validation {
        /// Number of accumulated validation failures.
        count: usize,
        /// Human-readable summary of the failures.
        details: String,
    },

    /// Returned when a required variable is not present in a file.
    #[error("variable '{name}' not found in {}", path.display())]
    MissingVariable {
        /// Name of the missing variable.
        name: String,
        /// Path to the file that was inspected.
        path: PathBuf,
    },

    /// Returned when a dimension has an unexpected size.
    #[error("dimension '{name}' mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Name of the dimension.
        name: String,
        /// Expected size.
        expected: usize,
        /// Actual size.
        got: usize,
    },

    /// Returned when a time value cannot be parsed or is out of range.
    #[error("invalid time: {reason}")]
    InvalidTime {
        /// Description of the time parsing issue.
        reason: String,
    },
}

impl From<netcdf::Error> for IoError {
    fn from(e: netcdf::Error) -> Self {
        IoError::Netcdf {
            reason: e.to_string(),
        }
    }
}

impl From<parquet::errors::ParquetError> for IoError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        IoError::Parquet {
            reason: e.to_string(),
        }
    }
}

impl From<zeus_calendar::CalendarError> for IoError {
    fn from(e: zeus_calendar::CalendarError) -> Self {
        IoError::Calendar {
            reason: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_file_not_found() {
        let err = IoError::FileNotFound {
            path: PathBuf::from("/tmp/missing.nc"),
        };
        assert_eq!(err.to_string(), "file not found: /tmp/missing.nc");
    }

    #[test]
    fn display_netcdf() {
        let err = IoError::Netcdf {
            reason: "bad header".to_string(),
        };
        assert_eq!(err.to_string(), "netcdf error: bad header");
    }

    #[test]
    fn display_parquet() {
        let err = IoError::Parquet {
            reason: "corrupt footer".to_string(),
        };
        assert_eq!(err.to_string(), "parquet error: corrupt footer");
    }

    #[test]
    fn display_calendar() {
        let err = IoError::Calendar {
            reason: "invalid doy".to_string(),
        };
        assert_eq!(err.to_string(), "calendar error: invalid doy");
    }

    #[test]
    fn display_validation() {
        let err = IoError::Validation {
            count: 3,
            details: "temperature out of range; precip negative; wind missing".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "3 validation error(s): temperature out of range; precip negative; wind missing"
        );
    }

    #[test]
    fn display_missing_variable() {
        let err = IoError::MissingVariable {
            name: "tmax".to_string(),
            path: PathBuf::from("/data/obs.nc"),
        };
        assert_eq!(err.to_string(), "variable 'tmax' not found in /data/obs.nc");
    }

    #[test]
    fn display_dimension_mismatch() {
        let err = IoError::DimensionMismatch {
            name: "time".to_string(),
            expected: 365,
            got: 366,
        };
        assert_eq!(
            err.to_string(),
            "dimension 'time' mismatch: expected 365, got 366"
        );
    }

    #[test]
    fn display_invalid_time() {
        let err = IoError::InvalidTime {
            reason: "negative offset".to_string(),
        };
        assert_eq!(err.to_string(), "invalid time: negative offset");
    }

    #[test]
    fn from_netcdf_error() {
        let nc_err = netcdf::Error::Str("test nc error".to_string());
        let err: IoError = nc_err.into();
        assert!(matches!(err, IoError::Netcdf { .. }));
        assert!(err.to_string().contains("test nc error"));
    }

    #[test]
    fn from_parquet_error() {
        let pq_err = parquet::errors::ParquetError::General("test pq error".to_string());
        let err: IoError = pq_err.into();
        assert!(matches!(err, IoError::Parquet { .. }));
        assert!(err.to_string().contains("test pq error"));
    }

    #[test]
    fn from_calendar_error() {
        let cal_err = zeus_calendar::CalendarError::InvalidDoy { doy: 0 };
        let err: IoError = cal_err.into();
        assert!(matches!(err, IoError::Calendar { .. }));
        assert!(err.to_string().contains("calendar error"));
    }

    #[test]
    fn error_is_send_sync_and_std_error() {
        fn assert_bounds<T: Send + Sync + std::error::Error>() {}
        assert_bounds::<IoError>();
    }
}
