//! Integration test: round-trip synthetic weather through Parquet.

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, Int32Type, UInt8Type, UInt16Type, UInt32Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use zeus_io::{Compression, SyntheticWeather, WriterConfig, write_parquet};

/// Helper: build a small `SyntheticWeather` that lives long enough for the
/// test by keeping owned buffers in scope.
struct Fixture {
    precip: Vec<f64>,
    temp_max: Vec<f64>,
    temp_min: Vec<f64>,
    months: Vec<u8>,
    water_years: Vec<i32>,
    days_of_year: Vec<u16>,
}

impl Fixture {
    fn new(n: usize, realisation: u32) -> Self {
        Self {
            precip: (0..n)
                .map(|i| i as f64 * 0.5 + realisation as f64)
                .collect(),
            temp_max: vec![25.0; n],
            temp_min: vec![10.0; n],
            months: (0..n).map(|i| (i % 12) as u8 + 1).collect(),
            water_years: vec![2020; n],
            days_of_year: (0..n).map(|i| i as u16 + 1).collect(),
        }
    }

    fn as_weather(&self, realisation: u32) -> SyntheticWeather<'_> {
        SyntheticWeather::new(
            "test_site",
            &self.precip,
            Some(&self.temp_max),
            Some(&self.temp_min),
            &self.months,
            &self.water_years,
            &self.days_of_year,
            realisation,
        )
        .expect("fixture is valid")
    }
}

#[test]
fn round_trip_parquet_with_temp() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("output.parquet");

    let f0 = Fixture::new(5, 0);
    let f1 = Fixture::new(5, 1);
    let realisations = [f0.as_weather(0), f1.as_weather(1)];

    let config = WriterConfig::default().with_compression(Compression::Snappy);
    write_parquet(&path, &realisations, &config).expect("write succeeds");

    // Read back and verify.
    let file = std::fs::File::open(&path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("reader builder");
    let reader = builder.build().expect("build reader");

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().expect("read batches");
    assert!(!batches.is_empty());

    // Total rows = 2 realisations * 5 rows each = 10
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10);

    // Check schema has 8 columns (with site and temp).
    assert_eq!(batches[0].num_columns(), 8);

    // Verify column names.
    let schema = batches[0].schema();
    assert_eq!(schema.field(0).name(), "site");
    assert_eq!(schema.field(1).name(), "realisation");
    assert_eq!(schema.field(2).name(), "month");
    assert_eq!(schema.field(3).name(), "water_year");
    assert_eq!(schema.field(4).name(), "day_of_year");
    assert_eq!(schema.field(5).name(), "precip");
    assert_eq!(schema.field(6).name(), "temp_max");
    assert_eq!(schema.field(7).name(), "temp_min");

    // Verify first batch values.
    let batch = &batches[0];
    let site_col = batch.column(0).as_string::<i32>();
    assert_eq!(site_col.value(0), "test_site");

    let realisation_col = batch.column(1).as_primitive::<UInt32Type>();
    assert_eq!(realisation_col.value(0), 0);

    let precip_col = batch.column(5).as_primitive::<Float64Type>();
    assert!((precip_col.value(0) - 0.0).abs() < f64::EPSILON);

    let temp_max_col = batch.column(6).as_primitive::<Float64Type>();
    assert!((temp_max_col.value(0) - 25.0).abs() < f64::EPSILON);
}

#[test]
fn round_trip_parquet_without_temp() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("no_temp.parquet");

    let precip = vec![1.0, 2.0, 3.0];
    let months = vec![1u8, 2, 3];
    let water_years = vec![2021i32, 2021, 2021];
    let days_of_year = vec![1u16, 32, 60];

    let sw = SyntheticWeather::new(
        "test_site",
        &precip,
        None,
        None,
        &months,
        &water_years,
        &days_of_year,
        0,
    )
    .expect("valid");

    let config = WriterConfig::default();
    write_parquet(&path, &[sw], &config).expect("write succeeds");

    let file = std::fs::File::open(&path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("reader builder");
    let reader = builder.build().expect("build reader");

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().expect("read batches");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
    assert_eq!(batches[0].num_columns(), 6);
}

#[test]
fn round_trip_parquet_zstd_compression() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("zstd.parquet");

    let precip = vec![0.5; 10];
    let months = vec![6u8; 10];
    let water_years = vec![2022i32; 10];
    let days_of_year: Vec<u16> = (152..162).collect();

    let sw = SyntheticWeather::new(
        "test_site",
        &precip,
        None,
        None,
        &months,
        &water_years,
        &days_of_year,
        42,
    )
    .expect("valid");

    let config = WriterConfig::default().with_compression(Compression::Zstd);
    write_parquet(&path, &[sw], &config).expect("write with zstd succeeds");

    // Verify we can read it back.
    let file = std::fs::File::open(&path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("reader builder");
    let reader = builder.build().expect("build reader");

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().expect("read batches");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10);

    // Check realisation value (column 1, after site at column 0).
    let batch = &batches[0];
    let r_col = batch.column(1).as_primitive::<UInt32Type>();
    assert_eq!(r_col.value(0), 42);
}

#[test]
fn round_trip_verifies_values() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("verify.parquet");

    let precip = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    let temp_max = vec![30.0, 31.0, 32.0, 33.0, 34.0];
    let temp_min = vec![5.0, 6.0, 7.0, 8.0, 9.0];
    let months = vec![10u8, 10, 10, 11, 11];
    let water_years = vec![2001i32, 2001, 2001, 2001, 2001];
    let days_of_year = vec![274u16, 275, 276, 305, 306];

    let sw = SyntheticWeather::new(
        "test_site",
        &precip,
        Some(&temp_max),
        Some(&temp_min),
        &months,
        &water_years,
        &days_of_year,
        7,
    )
    .expect("valid");

    write_parquet(&path, &[sw], &WriterConfig::default()).expect("write");

    let file = std::fs::File::open(&path).expect("open");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("builder");
    let reader = builder.build().expect("build");

    let batch: RecordBatch = reader
        .collect::<Result<Vec<_>, _>>()
        .expect("read")
        .into_iter()
        .next()
        .expect("at least one batch");

    // Verify site column values.
    let site_col = batch.column(0).as_string::<i32>();
    for i in 0..5 {
        assert_eq!(site_col.value(i), "test_site");
    }

    // Verify every column value (indices shifted +1 for site column).
    let r = batch.column(1).as_primitive::<UInt32Type>();
    let m = batch.column(2).as_primitive::<UInt8Type>();
    let wy = batch.column(3).as_primitive::<Int32Type>();
    let doy = batch.column(4).as_primitive::<UInt16Type>();
    let p = batch.column(5).as_primitive::<Float64Type>();
    let tmax = batch.column(6).as_primitive::<Float64Type>();
    let tmin = batch.column(7).as_primitive::<Float64Type>();

    for i in 0..5 {
        assert_eq!(r.value(i), 7);
        assert_eq!(m.value(i), months[i]);
        assert_eq!(wy.value(i), water_years[i]);
        assert_eq!(doy.value(i), days_of_year[i]);
        assert!((p.value(i) - precip[i]).abs() < f64::EPSILON);
        assert!((tmax.value(i) - temp_max[i]).abs() < f64::EPSILON);
        assert!((tmin.value(i) - temp_min[i]).abs() < f64::EPSILON);
    }
}
