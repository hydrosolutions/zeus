//! Low-level NetCDF extraction helpers.

use std::path::Path;

use chrono::{Datelike, NaiveDate};
use netcdf::AttributeValue;
use zeus_calendar::NoLeapDate;

use crate::error::IoError;

/// Open a NetCDF file at `path`, returning [`IoError::FileNotFound`] if the
/// path does not exist on disk.
pub(crate) fn open_file(path: &Path) -> Result<netcdf::File, IoError> {
    if !path.exists() {
        return Err(IoError::FileNotFound {
            path: path.to_path_buf(),
        });
    }
    Ok(netcdf::open(path)?)
}

/// Read a 1-D `f64` variable, trying each alias in order.
///
/// Returns the data from the first alias that matches. If none match,
/// returns [`IoError::MissingVariable`] with the first alias as the name.
pub(crate) fn read_1d_f64(
    file: &netcdf::File,
    aliases: &[&str],
    path: &Path,
) -> Result<Vec<f64>, IoError> {
    for &alias in aliases {
        if let Some(var) = file.variable(alias) {
            return Ok(var.get_values::<f64, _>(..)?);
        }
    }

    let name = aliases.first().copied().unwrap_or("unknown");
    Err(IoError::MissingVariable {
        name: name.to_string(),
        path: path.to_path_buf(),
    })
}

/// Read a 3-D `f64` variable and return the flattened data together with
/// the shape `[nt, ny, nx]` derived from the variable's dimensions.
pub(crate) fn read_3d_f64(
    file: &netcdf::File,
    var_name: &str,
    path: &Path,
) -> Result<(Vec<f64>, [usize; 3]), IoError> {
    let var = file
        .variable(var_name)
        .ok_or_else(|| IoError::MissingVariable {
            name: var_name.to_string(),
            path: path.to_path_buf(),
        })?;

    let dims = var.dimensions();
    if dims.len() != 3 {
        return Err(IoError::DimensionMismatch {
            name: format!("{var_name} dimensions"),
            expected: 3,
            got: dims.len(),
        });
    }

    let nt = dims[0].len();
    let ny = dims[1].len();
    let nx = dims[2].len();

    let mut data = var.get_values::<f64, _>(..)?;
    if let Some(fv) = read_fill_value(&var)
        && !fv.is_nan()
    {
        replace_fill_with_nan(&mut data, fv);
    }
    Ok((data, [nt, ny, nx]))
}

/// Read the `units` and optional `calendar` attributes from a time variable.
///
/// Parses CF-convention strings like `"days since YYYY-MM-DD"` or
/// `"days since YYYY-MM-DD HH:MM:SS"` and returns the calendar name
/// (defaulting to `"noleap"`) together with the parsed base date.
pub(crate) fn read_time_units(
    file: &netcdf::File,
    time_var: &str,
    path: &Path,
) -> Result<(String, NaiveDate), IoError> {
    let var = file
        .variable(time_var)
        .ok_or_else(|| IoError::MissingVariable {
            name: time_var.to_string(),
            path: path.to_path_buf(),
        })?;

    // Read the "units" attribute.
    let units_str: String = var
        .attribute_value("units")
        .ok_or_else(|| IoError::InvalidTime {
            reason: format!("time variable '{time_var}' has no 'units' attribute"),
        })?
        .map_err(|e| IoError::InvalidTime {
            reason: format!("failed to read 'units' attribute: {e}"),
        })?
        .try_into()
        .map_err(|e: netcdf::Error| IoError::InvalidTime {
            reason: format!("'units' attribute is not a string: {e}"),
        })?;

    // Expected format: "days since YYYY-MM-DD" or "days since YYYY-MM-DD HH:MM:SS"
    let parts: Vec<&str> = units_str.splitn(3, ' ').collect();
    if parts.len() < 3 || parts[1] != "since" {
        return Err(IoError::InvalidTime {
            reason: format!("unexpected time units format: '{units_str}'"),
        });
    }

    // Take only the date portion (first 10 characters of parts[2]).
    let date_str = if parts[2].len() >= 10 {
        &parts[2][..10]
    } else {
        parts[2]
    };

    let base_date =
        NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|e| IoError::InvalidTime {
            reason: format!("failed to parse base date '{date_str}': {e}"),
        })?;

    // Read the optional "calendar" attribute, defaulting to "noleap".
    // CF convention defaults to "standard" (Gregorian), but Zeus only
    // supports noleap calendars, so we default to "noleap" here.
    let calendar = var
        .attribute_value("calendar")
        .and_then(|res| res.ok())
        .and_then(|av| match av {
            AttributeValue::Str(s) => Some(s),
            _ => None,
        })
        .unwrap_or_else(|| "noleap".to_string());

    Ok((calendar, base_date))
}

/// Read the `_FillValue` or `missing_value` attribute from a variable, promoting
/// to `f64`. Returns `None` if neither attribute exists.
fn read_fill_value(var: &netcdf::Variable) -> Option<f64> {
    for attr_name in ["_FillValue", "missing_value"] {
        if let Some(val) = var.attribute_value(attr_name).and_then(|res| res.ok()) {
            match val {
                AttributeValue::Double(v) => return Some(v),
                AttributeValue::Float(v) => return Some(v as f64),
                AttributeValue::Short(v) => return Some(v as f64),
                AttributeValue::Int(v) => return Some(v as f64),
                AttributeValue::Longlong(v) => return Some(v as f64),
                _ => {}
            }
        }
    }
    None
}

/// Replace every element in `data` whose bit-pattern matches `fill_value` with
/// [`f64::NAN`]. This is a no-op if `fill_value` is already NaN.
pub(crate) fn replace_fill_with_nan(data: &mut [f64], fill_value: f64) {
    if fill_value.is_nan() {
        return;
    }
    let fill_bits = fill_value.to_bits();
    for val in data.iter_mut() {
        if val.to_bits() == fill_bits {
            *val = f64::NAN;
        }
    }
}

/// Convert floating-point day offsets from a base date into [`NoLeapDate`]s.
///
/// Each offset is truncated to an integer number of days and added using
/// pure 365-day no-leap arithmetic via [`NoLeapDate::add_days`].
///
/// # Supported calendars
///
/// `"noleap"`, `"no_leap"`, `"365_day"` (case-insensitive).
pub(crate) fn time_offsets_to_dates(
    base_date: NaiveDate,
    offsets: &[f64],
    calendar: &str,
) -> Result<Vec<NoLeapDate>, IoError> {
    match calendar.to_lowercase().as_str() {
        "noleap" | "no_leap" | "365_day" => {}
        other => {
            return Err(IoError::Calendar {
                reason: format!("unsupported calendar: '{other}'"),
            });
        }
    }

    let base = NoLeapDate::new(
        base_date.year(),
        base_date.month() as u8,
        base_date.day() as u8,
    )
    .map_err(|e| IoError::Calendar {
        reason: format!("failed to convert base date: {e}"),
    })?;

    offsets
        .iter()
        .map(|&offset| {
            let days = offset as i64;
            Ok(base.add_days(days))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_to_dates_basic() {
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let offsets = vec![0.0, 1.0, 10.0, 365.0];

        let dates = time_offsets_to_dates(base, &offsets, "noleap").expect("conversion succeeds");

        assert_eq!(dates.len(), 4);

        // Day 0 => 2000-01-01
        assert_eq!(dates[0].year(), 2000);
        assert_eq!(dates[0].month(), 1);
        assert_eq!(dates[0].day(), 1);

        // Day 1 => 2000-01-02
        assert_eq!(dates[1].year(), 2000);
        assert_eq!(dates[1].month(), 1);
        assert_eq!(dates[1].day(), 2);

        // Day 10 => 2000-01-11
        assert_eq!(dates[2].year(), 2000);
        assert_eq!(dates[2].month(), 1);
        assert_eq!(dates[2].day(), 11);

        // Day 365 => 2001-01-01 in noleap (365-day years)
        assert_eq!(dates[3].year(), 2001);
        assert_eq!(dates[3].month(), 1);
        assert_eq!(dates[3].day(), 1);
    }

    #[test]
    fn offsets_to_dates_fractional_truncated() {
        let base = NaiveDate::from_ymd_opt(2001, 6, 15).expect("valid date");
        let offsets = vec![0.5, 1.9, 2.0];

        let dates = time_offsets_to_dates(base, &offsets, "noleap").expect("conversion succeeds");

        // Fractional days are truncated to integer: 0.5 => 0, 1.9 => 1
        assert_eq!(dates[0].year(), 2001);
        assert_eq!(dates[0].month(), 6);
        assert_eq!(dates[0].day(), 15);

        assert_eq!(dates[1].year(), 2001);
        assert_eq!(dates[1].month(), 6);
        assert_eq!(dates[1].day(), 16);

        assert_eq!(dates[2].year(), 2001);
        assert_eq!(dates[2].month(), 6);
        assert_eq!(dates[2].day(), 17);
    }

    #[test]
    fn offsets_to_dates_empty() {
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let dates = time_offsets_to_dates(base, &[], "noleap").expect("conversion succeeds");
        assert!(dates.is_empty());
    }

    #[test]
    fn offsets_to_dates_feb29_becomes_mar1() {
        // In noleap, offset 59 from Jan 1 is DOY 60 = Mar 1 (Feb has only 28 days).
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let offsets = vec![59.0];

        let dates = time_offsets_to_dates(base, &offsets, "noleap").expect("conversion succeeds");

        assert_eq!(dates[0].year(), 2000);
        assert_eq!(dates[0].month(), 3);
        assert_eq!(dates[0].day(), 1);
    }

    #[test]
    fn offsets_to_dates_multi_year() {
        let base = NaiveDate::from_ymd_opt(1984, 1, 1).expect("valid date");
        let offsets = vec![0.0, 365.0, 730.0, 1095.0, 1460.0];

        let dates = time_offsets_to_dates(base, &offsets, "noleap").expect("conversion succeeds");

        for (i, date) in dates.iter().enumerate() {
            assert_eq!(date.year(), 1984 + i as i32, "year mismatch at index {i}");
            assert_eq!(date.month(), 1, "month mismatch at index {i}");
            assert_eq!(date.day(), 1, "day mismatch at index {i}");
        }
    }

    #[test]
    fn offsets_to_dates_calendar_variants() {
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let offsets = vec![10.0];

        for cal in ["noleap", "365_day", "no_leap"] {
            let dates = time_offsets_to_dates(base, &offsets, cal).expect("conversion succeeds");
            assert_eq!(dates[0].month(), 1);
            assert_eq!(dates[0].day(), 11);
        }
    }

    #[test]
    fn offsets_to_dates_unsupported_calendar() {
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let result = time_offsets_to_dates(base, &[0.0], "standard");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, IoError::Calendar { .. }),
            "expected IoError::Calendar, got: {err:?}"
        );
    }

    // --- replace_fill_with_nan tests ---

    #[test]
    fn fill_nan_basic_replacement() {
        let fill = -9999.0;
        let mut data = vec![1.0, -9999.0, 3.0, -9999.0, 5.0];
        replace_fill_with_nan(&mut data, fill);
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_nan());
        assert_eq!(data[2], 3.0);
        assert!(data[3].is_nan());
        assert_eq!(data[4], 5.0);
    }

    #[test]
    fn fill_nan_no_matches() {
        let fill = -9999.0;
        let mut data = vec![1.0, 2.0, 3.0];
        replace_fill_with_nan(&mut data, fill);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn fill_nan_all_matches() {
        let fill = -9999.0;
        let mut data = vec![-9999.0, -9999.0, -9999.0];
        replace_fill_with_nan(&mut data, fill);
        for val in &data {
            assert!(val.is_nan(), "expected NaN, got {val}");
        }
    }

    #[test]
    fn fill_nan_already_nan() {
        // When fill_value is NaN the function is a no-op.
        let mut data = vec![1.0, f64::NAN, 3.0];
        replace_fill_with_nan(&mut data, f64::NAN);
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_nan()); // untouched
        assert_eq!(data[2], 3.0);
    }

    #[test]
    fn fill_nan_large_fill_value() {
        let fill = 1e20;
        let mut data = vec![0.5, 1e20, -1e20, 1e20];
        replace_fill_with_nan(&mut data, fill);
        assert_eq!(data[0], 0.5);
        assert!(data[1].is_nan());
        assert_eq!(data[2], -1e20);
        assert!(data[3].is_nan());
    }

    #[test]
    fn fill_nan_empty_slice() {
        let mut data: Vec<f64> = vec![];
        replace_fill_with_nan(&mut data, -9999.0);
        assert!(data.is_empty());
    }
}
