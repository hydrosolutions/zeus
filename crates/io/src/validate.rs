//! Accumulated validation utilities.
//!
//! Provides [`ValidationCollector`] for gathering multiple validation errors
//! into a single [`IoError::Validation`], plus standalone helpers that check
//! common invariants on climate data arrays.

use crate::error::IoError;

// ---------------------------------------------------------------------------
// ValidationCollector
// ---------------------------------------------------------------------------

/// Accumulates validation errors and converts them into a single
/// [`IoError::Validation`].
///
/// Create a collector, push zero or more error messages, then call
/// [`finish`](Self::finish) to obtain `Ok(())` when everything is valid or a
/// single `Err` that summarises every violation.
pub(crate) struct ValidationCollector {
    errors: Vec<String>,
}

impl ValidationCollector {
    /// Create an empty collector.
    pub(crate) fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Record one validation error.
    pub(crate) fn push(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    /// Returns `true` when no errors have been recorded.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the number of recorded errors.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.errors.len()
    }

    /// Consume the collector and return `Ok(())` if no errors were recorded,
    /// or `Err(IoError::Validation { count, details })` otherwise.
    ///
    /// The `details` string joins all messages with `"; "`.
    pub(crate) fn finish(self) -> Result<(), IoError> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(IoError::Validation {
                count: self.errors.len(),
                details: self.errors.join("; "),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone validation helpers
// ---------------------------------------------------------------------------

/// Check that all present array lengths match `precip_len`.
///
/// `temp_max_len` and `temp_min_len` are optional because not every dataset
/// includes temperature columns. `dates_len` is always required.
pub(crate) fn validate_lengths(
    precip_len: usize,
    temp_max_len: Option<usize>,
    temp_min_len: Option<usize>,
    dates_len: usize,
) -> ValidationCollector {
    let mut c = ValidationCollector::new();

    if let Some(len) = temp_max_len
        && len != precip_len
    {
        c.push(format!(
            "temp_max length {len} != precip length {precip_len}"
        ));
    }

    if let Some(len) = temp_min_len
        && len != precip_len
    {
        c.push(format!(
            "temp_min length {len} != precip length {precip_len}"
        ));
    }

    if dates_len != precip_len {
        c.push(format!(
            "dates length {dates_len} != precip length {precip_len}"
        ));
    }

    c
}

/// Check that every precipitation value is non-negative.
///
/// Records one message per offending index.
pub(crate) fn validate_precip_non_negative(precip: &[f64]) -> ValidationCollector {
    let mut c = ValidationCollector::new();

    for (i, &val) in precip.iter().enumerate() {
        if val < 0.0 {
            c.push(format!("negative precipitation at index {i}: {val}"));
        }
    }

    c
}

/// Check that `temp_min[i] <= temp_max[i]` for every index.
///
/// Assumes both slices have the same length (length validation is handled
/// separately by [`validate_lengths`]).
pub(crate) fn validate_temp_ordering(temp_max: &[f64], temp_min: &[f64]) -> ValidationCollector {
    let mut c = ValidationCollector::new();

    for (i, (&tmax, &tmin)) in temp_max.iter().zip(temp_min.iter()).enumerate() {
        if tmin > tmax {
            c.push(format!(
                "temp_min ({tmin}) > temp_max ({tmax}) at index {i}"
            ));
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ValidationCollector -------------------------------------------------

    #[test]
    fn collector_empty_is_ok() {
        let c = ValidationCollector::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert!(c.finish().is_ok());
    }

    #[test]
    fn collector_non_empty_is_err_with_correct_count() {
        let mut c = ValidationCollector::new();
        c.push("error one");
        c.push("error two");
        assert!(!c.is_empty());
        assert_eq!(c.len(), 2);

        let err = c.finish().unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 2);
                assert!(details.contains("error one"));
                assert!(details.contains("error two"));
                assert!(details.contains("; "));
            }
            other => panic!("expected IoError::Validation, got {other:?}"),
        }
    }

    // -- validate_lengths ----------------------------------------------------

    #[test]
    fn lengths_all_match_is_empty() {
        let c = validate_lengths(100, Some(100), Some(100), 100);
        assert!(c.is_empty());
        assert!(c.finish().is_ok());
    }

    #[test]
    fn lengths_none_temps_match_is_empty() {
        let c = validate_lengths(100, None, None, 100);
        assert!(c.is_empty());
        assert!(c.finish().is_ok());
    }

    #[test]
    fn lengths_mismatches_produce_errors() {
        let c = validate_lengths(200, Some(100), Some(150), 180);
        assert_eq!(c.len(), 3);

        let err = c.finish().unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 3);
                assert!(details.contains("temp_max length 100 != precip length 200"));
                assert!(details.contains("temp_min length 150 != precip length 200"));
                assert!(details.contains("dates length 180 != precip length 200"));
            }
            other => panic!("expected IoError::Validation, got {other:?}"),
        }
    }

    // -- validate_precip_non_negative ----------------------------------------

    #[test]
    fn precip_all_positive_is_empty() {
        let data = vec![0.0, 1.5, 3.0, 0.0];
        let c = validate_precip_non_negative(&data);
        assert!(c.is_empty());
        assert!(c.finish().is_ok());
    }

    #[test]
    fn precip_negatives_produce_errors() {
        let data = vec![1.0, -0.5, 3.0, -1.2];
        let c = validate_precip_non_negative(&data);
        assert_eq!(c.len(), 2);

        let err = c.finish().unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 2);
                assert!(details.contains("negative precipitation at index 1"));
                assert!(details.contains("negative precipitation at index 3"));
            }
            other => panic!("expected IoError::Validation, got {other:?}"),
        }
    }

    // -- validate_temp_ordering ----------------------------------------------

    #[test]
    fn temps_ordered_is_empty() {
        let maxs = vec![30.0, 25.0, 20.0];
        let mins = vec![15.0, 10.0, 5.0];
        let c = validate_temp_ordering(&maxs, &mins);
        assert!(c.is_empty());
        assert!(c.finish().is_ok());
    }

    #[test]
    fn temps_equal_is_empty() {
        let maxs = vec![20.0, 20.0];
        let mins = vec![20.0, 20.0];
        let c = validate_temp_ordering(&maxs, &mins);
        assert!(c.is_empty());
        assert!(c.finish().is_ok());
    }

    #[test]
    fn temps_violations_produce_errors() {
        let maxs = vec![30.0, 20.0, 25.0];
        let mins = vec![15.0, 25.0, 30.0];
        let c = validate_temp_ordering(&maxs, &mins);
        assert_eq!(c.len(), 2);

        let err = c.finish().unwrap_err();
        match err {
            IoError::Validation { count, details } => {
                assert_eq!(count, 2);
                assert!(details.contains("temp_min (25) > temp_max (20) at index 1"));
                assert!(details.contains("temp_min (30) > temp_max (25) at index 2"));
            }
            other => panic!("expected IoError::Validation, got {other:?}"),
        }
    }
}
