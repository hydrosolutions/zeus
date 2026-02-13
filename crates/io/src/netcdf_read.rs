//! Low-level NetCDF extraction helpers.

use std::path::Path;

use chrono::NaiveDate;
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

    let data = var.get_values::<f64, _>(..)?;
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

/// Convert floating-point day offsets from a base date into [`NoLeapDate`]s.
///
/// Each offset is rounded to an integer number of days, added to `base_date`
/// using chrono arithmetic, and then converted to a `NoLeapDate`.
pub(crate) fn time_offsets_to_dates(
    base_date: NaiveDate,
    offsets: &[f64],
    _calendar: &str,
) -> Result<Vec<NoLeapDate>, IoError> {
    offsets
        .iter()
        .map(|&offset| {
            let days = offset as i64;
            let greg = base_date
                .checked_add_signed(chrono::TimeDelta::days(days))
                .ok_or_else(|| IoError::InvalidTime {
                    reason: format!("date overflow adding {days} days to {base_date}"),
                })?;

            let year =
                greg.format("%Y")
                    .to_string()
                    .parse::<i32>()
                    .map_err(|e| IoError::InvalidTime {
                        reason: format!("failed to parse year: {e}"),
                    })?;
            let month =
                greg.format("%m")
                    .to_string()
                    .parse::<u8>()
                    .map_err(|e| IoError::InvalidTime {
                        reason: format!("failed to parse month: {e}"),
                    })?;
            let day =
                greg.format("%d")
                    .to_string()
                    .parse::<u8>()
                    .map_err(|e| IoError::InvalidTime {
                        reason: format!("failed to parse day: {e}"),
                    })?;

            NoLeapDate::new(year, month, day).map_err(IoError::from)
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

        // Day 365 => 2000-12-31 in gregorian, which is also valid in noleap
        // (chrono gives 2000-12-31 since 2000 is a leap year in gregorian;
        //  but 365 days from Jan 1 in a leap year is Dec 31)
        assert_eq!(dates[3].year(), 2000);
        assert_eq!(dates[3].month(), 12);
        assert_eq!(dates[3].day(), 31);
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
    fn offsets_to_dates_feb29_rejected() {
        // In gregorian 2000 is a leap year, so day 59 from Jan 1 is Feb 29.
        // NoLeapDate should reject this.
        let base = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        let offsets = vec![59.0]; // Feb 29 in leap year

        let result = time_offsets_to_dates(base, &offsets, "noleap");
        assert!(result.is_err());
    }
}
