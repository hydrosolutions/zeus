//! Validated time-series wrapper.

use crate::error::WaveletError;

/// A validated time series of finite `f64` values.
///
/// Wraps a `Vec<f64>` and guarantees:
/// - length >= 2
/// - all values are finite (no NaN or infinity)
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::TimeSeries;
///
/// let ts = TimeSeries::new(vec![1.0, 2.0, 3.0])?;
/// assert_eq!(ts.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct TimeSeries {
    data: Vec<f64>,
}

impl TimeSeries {
    /// Creates a new `TimeSeries` after validating the data.
    ///
    /// # Errors
    ///
    /// | Variant | Trigger |
    /// |---------|---------|
    /// | [`WaveletError::SeriesTooShort`] | `data.len() < 2` |
    /// | [`WaveletError::NonFiniteData`] | any element is NaN or infinite |
    pub fn new(data: Vec<f64>) -> Result<Self, WaveletError> {
        if data.len() < 2 {
            return Err(WaveletError::SeriesTooShort {
                len: data.len(),
                min: 2,
            });
        }
        if !data.iter().all(|v| v.is_finite()) {
            return Err(WaveletError::NonFiniteData);
        }
        Ok(Self { data })
    }

    /// Returns the time series as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns the number of observations.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the series is empty.
    ///
    /// Note: a valid `TimeSeries` is never empty (minimum length is 2).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl AsRef<[f64]> for TimeSeries {
    fn as_ref(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_series() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
        assert_eq!(ts.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_minimum_length() {
        let ts = TimeSeries::new(vec![1.0, 2.0]).unwrap();
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn new_too_short_empty() {
        let err = TimeSeries::new(vec![]).unwrap_err();
        assert!(matches!(
            err,
            WaveletError::SeriesTooShort { len: 0, min: 2 }
        ));
    }

    #[test]
    fn new_too_short_one() {
        let err = TimeSeries::new(vec![1.0]).unwrap_err();
        assert!(matches!(
            err,
            WaveletError::SeriesTooShort { len: 1, min: 2 }
        ));
    }

    #[test]
    fn new_nan_rejected() {
        let err = TimeSeries::new(vec![1.0, f64::NAN, 3.0]).unwrap_err();
        assert!(matches!(err, WaveletError::NonFiniteData));
    }

    #[test]
    fn new_infinity_rejected() {
        let err = TimeSeries::new(vec![1.0, f64::INFINITY]).unwrap_err();
        assert!(matches!(err, WaveletError::NonFiniteData));
    }

    #[test]
    fn new_neg_infinity_rejected() {
        let err = TimeSeries::new(vec![f64::NEG_INFINITY, 1.0]).unwrap_err();
        assert!(matches!(err, WaveletError::NonFiniteData));
    }

    #[test]
    fn as_ref_trait() {
        let ts = TimeSeries::new(vec![1.0, 2.0]).unwrap();
        let slice: &[f64] = ts.as_ref();
        assert_eq!(slice, &[1.0, 2.0]);
    }

    #[test]
    fn series_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<TimeSeries>();
    }

    #[test]
    fn series_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<TimeSeries>();
    }
}
