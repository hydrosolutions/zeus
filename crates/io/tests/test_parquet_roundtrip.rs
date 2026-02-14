//! Integration tests: round-trip synthetic weather through Parquet read/write.

use zeus_io::{
    Compression, IoError, OwnedSyntheticWeather, SyntheticWeather, WriterConfig, read_parquet,
    write_parquet,
};

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
            temp_max: (0..n).map(|i| 25.0 + i as f64).collect(),
            temp_min: (0..n).map(|i| 10.0 + i as f64).collect(),
            months: (0..n).map(|i| (i % 12) as u8 + 1).collect(),
            water_years: vec![2020; n],
            days_of_year: (0..n).map(|i| i as u16 + 1).collect(),
        }
    }

    fn as_weather_with_temp(&self, realisation: u32) -> SyntheticWeather<'_> {
        SyntheticWeather::new(
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

    fn as_weather_no_temp(&self, realisation: u32) -> SyntheticWeather<'_> {
        SyntheticWeather::new(
            &self.precip,
            None,
            None,
            &self.months,
            &self.water_years,
            &self.days_of_year,
            realisation,
        )
        .expect("fixture is valid")
    }
}

#[test]
fn round_trip_with_temp() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("with_temp.parquet");

    let f0 = Fixture::new(5, 0);
    let f1 = Fixture::new(5, 1);
    let realisations = [f0.as_weather_with_temp(0), f1.as_weather_with_temp(1)];

    let config = WriterConfig::default().with_compression(Compression::Snappy);
    write_parquet(&path, &realisations, &config).expect("write succeeds");

    let result = read_parquet(&path).expect("read succeeds");
    assert_eq!(result.len(), 2);

    // Verify realisation 0.
    let r0 = &result[0];
    assert_eq!(r0.realisation(), 0);
    assert_eq!(r0.len(), 5);
    assert_eq!(r0.precip(), f0.precip.as_slice());
    assert_eq!(r0.temp_max(), Some(f0.temp_max.as_slice()));
    assert_eq!(r0.temp_min(), Some(f0.temp_min.as_slice()));
    assert_eq!(r0.months(), f0.months.as_slice());
    assert_eq!(r0.water_years(), f0.water_years.as_slice());
    assert_eq!(r0.days_of_year(), f0.days_of_year.as_slice());

    // Verify realisation 1.
    let r1 = &result[1];
    assert_eq!(r1.realisation(), 1);
    assert_eq!(r1.len(), 5);
    assert_eq!(r1.precip(), f1.precip.as_slice());
    assert_eq!(r1.temp_max(), Some(f1.temp_max.as_slice()));
    assert_eq!(r1.temp_min(), Some(f1.temp_min.as_slice()));
    assert_eq!(r1.months(), f1.months.as_slice());
    assert_eq!(r1.water_years(), f1.water_years.as_slice());
    assert_eq!(r1.days_of_year(), f1.days_of_year.as_slice());
}

#[test]
fn round_trip_without_temp() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("no_temp.parquet");

    let f0 = Fixture::new(4, 0);
    let realisations = [f0.as_weather_no_temp(0)];

    let config = WriterConfig::default();
    write_parquet(&path, &realisations, &config).expect("write succeeds");

    let result = read_parquet(&path).expect("read succeeds");
    assert_eq!(result.len(), 1);

    let r0 = &result[0];
    assert_eq!(r0.realisation(), 0);
    assert_eq!(r0.len(), 4);
    assert_eq!(r0.precip(), f0.precip.as_slice());
    assert!(r0.temp_max().is_none());
    assert!(r0.temp_min().is_none());
    assert_eq!(r0.months(), f0.months.as_slice());
    assert_eq!(r0.water_years(), f0.water_years.as_slice());
    assert_eq!(r0.days_of_year(), f0.days_of_year.as_slice());
}

#[test]
fn read_parquet_file_not_found() {
    let result = read_parquet(std::path::Path::new(
        "/tmp/nonexistent_zeus_test_file.parquet",
    ));
    assert!(result.is_err());
    match result.unwrap_err() {
        IoError::FileNotFound { path } => {
            assert!(
                path.to_str()
                    .unwrap()
                    .contains("nonexistent_zeus_test_file")
            );
        }
        other => panic!("expected FileNotFound, got: {other}"),
    }
}

#[test]
fn round_trip_as_view_bridge() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("view_bridge.parquet");

    let f0 = Fixture::new(3, 0);
    let realisations = [f0.as_weather_with_temp(0)];

    write_parquet(&path, &realisations, &WriterConfig::default()).expect("write succeeds");

    let owned: Vec<OwnedSyntheticWeather> = read_parquet(&path).expect("read succeeds");
    assert_eq!(owned.len(), 1);

    // Convert to a borrowed SyntheticWeather view.
    let view: SyntheticWeather<'_> = owned[0].as_view().expect("as_view succeeds");

    assert_eq!(view.precip(), owned[0].precip());
    assert_eq!(view.temp_max(), owned[0].temp_max());
    assert_eq!(view.temp_min(), owned[0].temp_min());
    assert_eq!(view.months(), owned[0].months());
    assert_eq!(view.water_years(), owned[0].water_years());
    assert_eq!(view.days_of_year(), owned[0].days_of_year());
    assert_eq!(view.realisation(), owned[0].realisation());
    assert_eq!(view.len(), owned[0].len());
}
