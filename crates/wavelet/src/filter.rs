//! Wavelet filter definitions.

use crate::error::WaveletError;

/// Supported wavelet filters for MODWT decomposition.
///
/// Six filters are currently supported, spanning Haar, Daubechies (D),
/// and Least Asymmetric (LA) families.
///
/// # Example
///
/// ```ignore
/// use zeus_wavelet::WaveletFilter;
///
/// let filter = WaveletFilter::La8;
/// assert_eq!(filter.length(), 8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaveletFilter {
    /// Haar wavelet (length 2).
    Haar,
    /// Daubechies D4 wavelet (length 4).
    D4,
    /// Daubechies D6 wavelet (length 6).
    D6,
    /// Daubechies D8 wavelet (length 8).
    D8,
    /// Least Asymmetric LA(8) wavelet (length 8).
    La8,
    /// Least Asymmetric LA(16) wavelet (length 16).
    La16,
}

impl Default for WaveletFilter {
    /// Returns `WaveletFilter::La8` as the default filter.
    fn default() -> Self {
        Self::La8
    }
}

impl WaveletFilter {
    /// Returns the filter length (number of coefficients).
    pub fn length(&self) -> usize {
        match self {
            Self::Haar => 2,
            Self::D4 => 4,
            Self::D6 => 6,
            Self::D8 => 8,
            Self::La8 => 8,
            Self::La16 => 16,
        }
    }

    /// Returns the scaling (father wavelet) coefficients.
    pub fn scaling_coeffs(&self) -> &[f64] {
        let _ = self;
        todo!()
    }

    /// Returns the wavelet (mother wavelet) coefficients.
    ///
    /// Derived from the scaling coefficients via the quadrature mirror
    /// filter relationship.
    pub fn wavelet_coeffs(&self) -> Vec<f64> {
        let _ = self;
        todo!()
    }

    /// Parses a wavelet filter from a case-insensitive name string.
    ///
    /// # Supported Names
    ///
    /// | Input | Filter |
    /// |-------|--------|
    /// | `"haar"` | [`WaveletFilter::Haar`] |
    /// | `"d4"` | [`WaveletFilter::D4`] |
    /// | `"d6"` | [`WaveletFilter::D6`] |
    /// | `"d8"` | [`WaveletFilter::D8`] |
    /// | `"la8"` | [`WaveletFilter::La8`] |
    /// | `"la16"` | [`WaveletFilter::La16`] |
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::UnsupportedFilter`] if the name is not recognized.
    pub fn from_name(name: &str) -> Result<Self, WaveletError> {
        match name.to_lowercase().as_str() {
            "haar" => Ok(Self::Haar),
            "d4" => Ok(Self::D4),
            "d6" => Ok(Self::D6),
            "d8" => Ok(Self::D8),
            "la8" => Ok(Self::La8),
            "la16" => Ok(Self::La16),
            _ => Err(WaveletError::UnsupportedFilter(name.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_lengths() {
        assert_eq!(WaveletFilter::Haar.length(), 2);
        assert_eq!(WaveletFilter::D4.length(), 4);
        assert_eq!(WaveletFilter::D6.length(), 6);
        assert_eq!(WaveletFilter::D8.length(), 8);
        assert_eq!(WaveletFilter::La8.length(), 8);
        assert_eq!(WaveletFilter::La16.length(), 16);
    }

    #[test]
    fn filter_default_is_la8() {
        assert_eq!(WaveletFilter::default(), WaveletFilter::La8);
    }

    #[test]
    fn from_name_valid() {
        assert_eq!(
            WaveletFilter::from_name("haar").unwrap(),
            WaveletFilter::Haar
        );
        assert_eq!(WaveletFilter::from_name("D4").unwrap(), WaveletFilter::D4);
        assert_eq!(WaveletFilter::from_name("d6").unwrap(), WaveletFilter::D6);
        assert_eq!(WaveletFilter::from_name("D8").unwrap(), WaveletFilter::D8);
        assert_eq!(WaveletFilter::from_name("La8").unwrap(), WaveletFilter::La8);
        assert_eq!(
            WaveletFilter::from_name("LA16").unwrap(),
            WaveletFilter::La16
        );
    }

    #[test]
    fn from_name_invalid() {
        let err = WaveletFilter::from_name("coif4").unwrap_err();
        assert!(matches!(err, WaveletError::UnsupportedFilter(ref s) if s == "coif4"));
    }

    #[test]
    fn filter_is_copy() {
        let a = WaveletFilter::La8;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn filter_is_send_and_sync() {
        fn assert_impl<T: Send + Sync>() {}
        assert_impl::<WaveletFilter>();
    }
}
