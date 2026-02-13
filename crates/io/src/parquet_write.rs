//! Low-level Parquet column building.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, RecordBatch, UInt8Array, UInt16Array, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::error::IoError;
use crate::synthetic::SyntheticWeather;

/// Builds the Arrow schema for synthetic weather output.
///
/// Always includes `realisation`, `month`, `water_year`, `day_of_year`, and
/// `precip` columns. When `has_temp` is true, `temp_max` and `temp_min` are
/// appended.
pub(crate) fn build_schema(has_temp: bool) -> Schema {
    let mut fields = vec![
        Field::new("realisation", DataType::UInt32, false),
        Field::new("month", DataType::UInt8, false),
        Field::new("water_year", DataType::Int32, false),
        Field::new("day_of_year", DataType::UInt16, false),
        Field::new("precip", DataType::Float64, false),
    ];

    if has_temp {
        fields.push(Field::new("temp_max", DataType::Float64, false));
        fields.push(Field::new("temp_min", DataType::Float64, false));
    }

    Schema::new(fields)
}

/// Converts a single [`SyntheticWeather`] realisation into an Arrow
/// [`RecordBatch`].
///
/// The batch schema must match the one returned by [`build_schema`] for the
/// same `has_temp` value; otherwise the call will fail.
pub(crate) fn synthetic_to_record_batch(
    weather: &SyntheticWeather<'_>,
    schema: &Schema,
) -> Result<RecordBatch, IoError> {
    let n = weather.len();

    let realisation_col: ArrayRef = Arc::new(UInt32Array::from(vec![weather.realisation(); n]));
    let month_col: ArrayRef = Arc::new(UInt8Array::from(weather.months().to_vec()));
    let water_year_col: ArrayRef = Arc::new(Int32Array::from(weather.water_years().to_vec()));
    let day_of_year_col: ArrayRef = Arc::new(UInt16Array::from(weather.days_of_year().to_vec()));
    let precip_col: ArrayRef = Arc::new(Float64Array::from(weather.precip().to_vec()));

    let mut columns: Vec<ArrayRef> = vec![
        realisation_col,
        month_col,
        water_year_col,
        day_of_year_col,
        precip_col,
    ];

    // If the schema includes temperature columns, append them.
    if schema.fields().len() > 5 {
        if let Some(tmax) = weather.temp_max() {
            columns.push(Arc::new(Float64Array::from(tmax.to_vec())));
        }
        if let Some(tmin) = weather.temp_min() {
            columns.push(Arc::new(Float64Array::from(tmin.to_vec())));
        }
    }

    RecordBatch::try_new(Arc::new(schema.clone()), columns).map_err(|e| IoError::Parquet {
        reason: e.to_string(),
    })
}

/// Writes a sequence of [`RecordBatch`]es to a Parquet file at `path`.
///
/// # Errors
///
/// Returns [`IoError::Parquet`] if file creation, batch writing, or file
/// finalisation fails.
pub(crate) fn write_batches(
    path: &Path,
    batches: &[RecordBatch],
    schema: &Schema,
    props: WriterProperties,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path).map_err(|e| IoError::Parquet {
        reason: e.to_string(),
    })?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;

    for batch in batches {
        writer.write(batch)?;
    }

    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_without_temp() {
        let schema = build_schema(false);
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "realisation");
        assert_eq!(schema.field(1).name(), "month");
        assert_eq!(schema.field(2).name(), "water_year");
        assert_eq!(schema.field(3).name(), "day_of_year");
        assert_eq!(schema.field(4).name(), "precip");
    }

    #[test]
    fn schema_with_temp() {
        let schema = build_schema(true);
        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(5).name(), "temp_max");
        assert_eq!(schema.field(6).name(), "temp_min");
    }

    #[test]
    fn record_batch_without_temp() {
        let precip = [1.0, 2.0, 3.0];
        let months = [1u8, 2, 3];
        let water_years = [2020i32, 2020, 2020];
        let days_of_year = [1u16, 32, 60];

        let sw =
            SyntheticWeather::new(&precip, None, None, &months, &water_years, &days_of_year, 0)
                .unwrap();

        let schema = build_schema(false);
        let batch = synthetic_to_record_batch(&sw, &schema).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn record_batch_with_temp() {
        let precip = [1.0, 2.0];
        let tmax = [25.0, 26.0];
        let tmin = [10.0, 11.0];
        let months = [6u8, 7];
        let water_years = [2021i32, 2021];
        let days_of_year = [152u16, 182];

        let sw = SyntheticWeather::new(
            &precip,
            Some(&tmax),
            Some(&tmin),
            &months,
            &water_years,
            &days_of_year,
            1,
        )
        .unwrap();

        let schema = build_schema(true);
        let batch = synthetic_to_record_batch(&sw, &schema).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 7);
    }
}
