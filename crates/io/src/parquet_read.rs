//! Low-level Parquet reading and column extraction.

use std::collections::BTreeMap;
use std::path::Path;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, Int32Type, UInt8Type, UInt16Type, UInt32Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::IoError;
use crate::owned_synthetic::OwnedSyntheticWeather;

/// Accumulator for a single realisation during grouping.
///
/// Fields: (precip, temp_max, temp_min, months, water_years, days_of_year).
type RealisationAccum = (
    Vec<f64>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Vec<u8>,
    Vec<i32>,
    Vec<u16>,
);

/// Expected column names for the precipitation-only schema.
const BASE_COLUMNS: [&str; 5] = [
    "realisation",
    "month",
    "water_year",
    "day_of_year",
    "precip",
];

/// Additional column names when temperature data is present.
const TEMP_COLUMNS: [&str; 2] = ["temp_max", "temp_min"];

/// Reads all record batches from a Parquet file.
///
/// # Errors
///
/// Returns [`IoError::FileNotFound`] if the file does not exist, or
/// [`IoError::Parquet`] if the file cannot be opened or read.
pub(crate) fn read_batches(path: &Path) -> Result<Vec<RecordBatch>, IoError> {
    if !path.exists() {
        return Err(IoError::FileNotFound {
            path: path.to_path_buf(),
        });
    }

    let file = std::fs::File::open(path).map_err(|e| IoError::Parquet {
        reason: e.to_string(),
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let batches: Vec<RecordBatch> =
        reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IoError::Parquet {
                reason: e.to_string(),
            })?;

    Ok(batches)
}

/// Validates the schema of a record batch against the expected synthetic weather
/// schema.
///
/// Returns `Ok(true)` if temperature columns are present (7 columns), or
/// `Ok(false)` if only precipitation columns are present (5 columns).
///
/// # Errors
///
/// Returns [`IoError::Validation`] if the schema does not match the expected
/// column names or count.
pub(crate) fn validate_schema(batch: &RecordBatch) -> Result<bool, IoError> {
    let num_cols = batch.num_columns();
    let has_temp = match num_cols {
        5 => false,
        7 => true,
        _ => {
            return Err(IoError::Validation {
                count: 1,
                details: format!("expected 5 or 7 columns, got {num_cols}"),
            });
        }
    };

    let schema = batch.schema();
    let mut mismatches: Vec<String> = Vec::new();

    for (i, expected_name) in BASE_COLUMNS.iter().enumerate() {
        let actual_name = schema.field(i).name();
        if actual_name != *expected_name {
            mismatches.push(format!(
                "column {i}: expected '{expected_name}', got '{actual_name}'"
            ));
        }
    }

    if has_temp {
        for (j, expected_name) in TEMP_COLUMNS.iter().enumerate() {
            let i = 5 + j;
            let actual_name = schema.field(i).name();
            if actual_name != *expected_name {
                mismatches.push(format!(
                    "column {i}: expected '{expected_name}', got '{actual_name}'"
                ));
            }
        }
    }

    if !mismatches.is_empty() {
        return Err(IoError::Validation {
            count: mismatches.len(),
            details: mismatches.join("; "),
        });
    }

    Ok(has_temp)
}

/// Groups record batches by realisation index, producing one
/// [`OwnedSyntheticWeather`] per realisation.
///
/// The returned vector is sorted by realisation index (ascending).
///
/// # Errors
///
/// Returns [`IoError::Parquet`] if column extraction fails, or
/// [`IoError::Validation`] if the resulting data is inconsistent.
pub(crate) fn group_by_realisation(
    batches: &[RecordBatch],
    has_temp: bool,
) -> Result<Vec<OwnedSyntheticWeather>, IoError> {
    let mut groups: BTreeMap<u32, RealisationAccum> = BTreeMap::new();

    for batch in batches {
        let realisation_col = batch.column(0).as_primitive::<UInt32Type>();
        let month_col = batch.column(1).as_primitive::<UInt8Type>();
        let water_year_col = batch.column(2).as_primitive::<Int32Type>();
        let day_of_year_col = batch.column(3).as_primitive::<UInt16Type>();
        let precip_col = batch.column(4).as_primitive::<Float64Type>();

        let temp_max_col = if has_temp {
            Some(batch.column(5).as_primitive::<Float64Type>())
        } else {
            None
        };

        let temp_min_col = if has_temp {
            Some(batch.column(6).as_primitive::<Float64Type>())
        } else {
            None
        };

        for row in 0..batch.num_rows() {
            let r = realisation_col.value(row);
            let entry = groups.entry(r).or_insert_with(|| {
                (
                    Vec::new(),
                    if has_temp { Some(Vec::new()) } else { None },
                    if has_temp { Some(Vec::new()) } else { None },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            });

            entry.0.push(precip_col.value(row));
            if let (Some(tmax_vec), Some(tmax_col)) = (&mut entry.1, temp_max_col) {
                tmax_vec.push(tmax_col.value(row));
            }
            if let (Some(tmin_vec), Some(tmin_col)) = (&mut entry.2, temp_min_col) {
                tmin_vec.push(tmin_col.value(row));
            }
            entry.3.push(month_col.value(row));
            entry.4.push(water_year_col.value(row));
            entry.5.push(day_of_year_col.value(row));
        }
    }

    groups
        .into_iter()
        .map(
            |(r, (precip, temp_max, temp_min, months, water_years, days_of_year))| {
                OwnedSyntheticWeather::new(
                    precip,
                    temp_max,
                    temp_min,
                    months,
                    water_years,
                    days_of_year,
                    r,
                )
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array, UInt8Array, UInt16Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::parquet_write;

    /// Helper: build a record batch using the existing `parquet_write` module.
    fn make_batch(has_temp: bool, realisation: u32, n: usize) -> RecordBatch {
        let schema = parquet_write::build_schema(has_temp);
        let precip: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let months: Vec<u8> = (0..n).map(|i| (i % 12) as u8 + 1).collect();
        let water_years: Vec<i32> = vec![2020; n];
        let days_of_year: Vec<u16> = (0..n).map(|i| i as u16 + 1).collect();
        let tmax = vec![25.0; n];
        let tmin = vec![10.0; n];

        let sw = if has_temp {
            crate::synthetic::SyntheticWeather::new(
                &precip,
                Some(&tmax),
                Some(&tmin),
                &months,
                &water_years,
                &days_of_year,
                realisation,
            )
            .unwrap()
        } else {
            crate::synthetic::SyntheticWeather::new(
                &precip,
                None,
                None,
                &months,
                &water_years,
                &days_of_year,
                realisation,
            )
            .unwrap()
        };

        parquet_write::synthetic_to_record_batch(&sw, &schema).unwrap()
    }

    #[test]
    fn validate_schema_5_columns() {
        let batch = make_batch(false, 0, 3);
        let has_temp = validate_schema(&batch).unwrap();
        assert!(!has_temp);
    }

    #[test]
    fn validate_schema_7_columns() {
        let batch = make_batch(true, 0, 3);
        let has_temp = validate_schema(&batch).unwrap();
        assert!(has_temp);
    }

    #[test]
    fn validate_schema_wrong_column_count() {
        // Build a batch with 4 columns.
        let schema = Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::UInt32, false),
            Field::new("c", DataType::UInt32, false),
            Field::new("d", DataType::UInt32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt32Array::from(vec![1])),
                Arc::new(UInt32Array::from(vec![2])),
                Arc::new(UInt32Array::from(vec![3])),
                Arc::new(UInt32Array::from(vec![4])),
            ],
        )
        .unwrap();

        let result = validate_schema(&batch);
        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::Validation { details, .. } => {
                assert!(details.contains("expected 5 or 7 columns"));
            }
            _ => panic!("expected Validation error"),
        }
    }

    #[test]
    fn validate_schema_wrong_column_name() {
        // Build a 5-column batch with a wrong name.
        let schema = Schema::new(vec![
            Field::new("wrong_name", DataType::UInt32, false),
            Field::new("month", DataType::UInt8, false),
            Field::new("water_year", DataType::Int32, false),
            Field::new("day_of_year", DataType::UInt16, false),
            Field::new("precip", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt32Array::from(vec![0u32])),
                Arc::new(UInt8Array::from(vec![1u8])),
                Arc::new(Int32Array::from(vec![2020])),
                Arc::new(UInt16Array::from(vec![1u16])),
                Arc::new(Float64Array::from(vec![0.5])),
            ],
        )
        .unwrap();

        let result = validate_schema(&batch);
        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::Validation { details, .. } => {
                assert!(details.contains("realisation"));
                assert!(details.contains("wrong_name"));
            }
            _ => panic!("expected Validation error"),
        }
    }

    #[test]
    fn group_by_realisation_single() {
        let batch = make_batch(true, 0, 3);
        let result = group_by_realisation(&[batch], true).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].realisation(), 0);
        assert_eq!(result[0].len(), 3);
        assert!(result[0].temp_max().is_some());
    }

    #[test]
    fn group_by_realisation_multiple() {
        let b0 = make_batch(false, 0, 2);
        let b1 = make_batch(false, 1, 3);
        let result = group_by_realisation(&[b0, b1], false).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].realisation(), 0);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].realisation(), 1);
        assert_eq!(result[1].len(), 3);
        assert!(result[0].temp_max().is_none());
        assert!(result[1].temp_max().is_none());
    }

    #[test]
    fn group_by_realisation_empty() {
        let result = group_by_realisation(&[], false).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_batches_file_not_found() {
        let result = read_batches(Path::new("/nonexistent/path/file.parquet"));
        assert!(result.is_err());
        match result.unwrap_err() {
            IoError::FileNotFound { path } => {
                assert_eq!(path.to_str().unwrap(), "/nonexistent/path/file.parquet");
            }
            _ => panic!("expected FileNotFound error"),
        }
    }
}
