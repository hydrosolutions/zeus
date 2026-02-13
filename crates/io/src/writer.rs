//! High-level Parquet writer configuration and orchestration.

use std::path::Path;

use parquet::file::properties::WriterProperties;

use crate::error::IoError;
use crate::parquet_write;
use crate::synthetic::SyntheticWeather;

/// Compression algorithm for Parquet output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Compression {
    /// No compression.
    None,
    /// Snappy compression (fast, moderate ratio).
    #[default]
    Snappy,
    /// Zstd compression (slower, better ratio).
    Zstd,
}

impl Compression {
    /// Converts to the corresponding `parquet::basic::Compression` variant.
    fn to_parquet(self) -> Result<parquet::basic::Compression, IoError> {
        Ok(match self {
            Self::None => parquet::basic::Compression::UNCOMPRESSED,
            Self::Snappy => parquet::basic::Compression::SNAPPY,
            Self::Zstd => {
                let level =
                    parquet::basic::ZstdLevel::try_new(3).map_err(|e| IoError::Parquet {
                        reason: e.to_string(),
                    })?;
                parquet::basic::Compression::ZSTD(level)
            }
        })
    }
}

/// Configuration for writing synthetic weather data to Parquet.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Compression algorithm to use.
    compression: Compression,
    /// Maximum number of rows per row group.
    row_group_size: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            compression: Compression::default(),
            row_group_size: 1_000_000,
        }
    }
}

impl WriterConfig {
    /// Sets the compression algorithm.
    pub fn with_compression(mut self, comp: Compression) -> Self {
        self.compression = comp;
        self
    }

    /// Sets the maximum number of rows per row group.
    pub fn with_row_group_size(mut self, size: usize) -> Self {
        self.row_group_size = size;
        self
    }

    /// Validates this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::Validation`] if `row_group_size` is zero.
    fn validate(&self) -> Result<(), IoError> {
        if self.row_group_size == 0 {
            return Err(IoError::Validation {
                count: 1,
                details: "row_group_size must be greater than 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Write synthetic weather realisations to a Parquet file.
///
/// Each [`SyntheticWeather`] entry becomes one or more row groups in the output.
///
/// # Errors
///
/// Returns [`IoError::Validation`] if the configuration is invalid, or
/// [`IoError::Parquet`] if schema construction, batch conversion, or file I/O
/// fails.
pub fn write_parquet(
    path: &Path,
    realisations: &[SyntheticWeather<'_>],
    config: &WriterConfig,
) -> Result<(), IoError> {
    config.validate()?;

    let has_temp = realisations.first().is_some_and(|r| r.temp_max().is_some());

    let schema = parquet_write::build_schema(has_temp);

    let compression = config.compression.to_parquet()?;
    let props = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_size(config.row_group_size)
        .build();

    let batches: Vec<_> = realisations
        .iter()
        .map(|r| parquet_write::synthetic_to_record_batch(r, &schema))
        .collect::<Result<Vec<_>, _>>()?;

    parquet_write::write_batches(path, &batches, &schema, props)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = WriterConfig::default();
        assert_eq!(config.compression, Compression::Snappy);
        assert_eq!(config.row_group_size, 1_000_000);
    }

    #[test]
    fn builder_methods() {
        let config = WriterConfig::default()
            .with_compression(Compression::Zstd)
            .with_row_group_size(500);
        assert_eq!(config.compression, Compression::Zstd);
        assert_eq!(config.row_group_size, 500);
    }

    #[test]
    fn validate_ok() {
        let config = WriterConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_zero_row_group_size() {
        let config = WriterConfig::default().with_row_group_size(0);
        let err = config.validate().unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 1);
                assert!(details.contains("row_group_size"));
            }
            _ => panic!("expected Validation error"),
        }
    }

    #[test]
    fn default_compression_is_snappy() {
        assert_eq!(Compression::default(), Compression::Snappy);
    }
}
