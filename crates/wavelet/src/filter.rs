//! Wavelet filter definitions.

use crate::error::WaveletError;

/// Haar scaling coefficients (Percival & Walden, 2000; waveslim R package).
const HAAR_SCALING: [f64; 2] = [
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];

/// Daubechies D4 scaling coefficients (Percival & Walden, 2000; waveslim R package).
const D4_SCALING: [f64; 4] = [
    0.4829629131445341,
    0.8365163037378077,
    0.2241438680420134,
    -0.1294095225512603,
];

/// Daubechies D6 scaling coefficients (Percival & Walden, 2000; waveslim R package).
const D6_SCALING: [f64; 6] = [
    0.3326705529500827,
    0.8068915093110928,
    0.4598775021184915,
    -0.1350110200102546,
    -0.0854412738820267,
    0.0352262918857096,
];

/// Daubechies D8 scaling coefficients (Percival & Walden, 2000; waveslim R package).
const D8_SCALING: [f64; 8] = [
    0.2303778133074431,
    0.7148465705484058,
    0.6308807679358788,
    -0.0279837694166834,
    -0.1870348117179132,
    0.0308413818353661,
    0.0328830116666778,
    -0.0105974017850021,
];

/// Least Asymmetric LA(8) scaling coefficients (Percival & Walden, 2000; waveslim R package).
const LA8_SCALING: [f64; 8] = [
    -0.07576571478935668,
    -0.02963552764596039,
    0.4976186676325629,
    0.803738751805386,
    0.29785779560560505,
    -0.09921954357695636,
    -0.01260396726226383,
    0.03222310060407815,
];

/// Least Asymmetric LA(16) scaling coefficients (Percival & Walden, 2000; waveslim R package).
const LA16_SCALING: [f64; 16] = [
    -0.0033824159513594,
    -0.0005421323316355,
    0.0316950878103452,
    0.0076074873252848,
    -0.1432942383510542,
    -0.0612733590679088,
    0.4813596512592012,
    0.7771857516997478,
    0.3644418948359564,
    -0.0519458381078751,
    -0.0272190299168137,
    0.0491371796734768,
    0.0038087520140601,
    -0.0149522583367926,
    -0.0003029205145516,
    0.0018899503329007,
];

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
        match self {
            Self::Haar => &HAAR_SCALING,
            Self::D4 => &D4_SCALING,
            Self::D6 => &D6_SCALING,
            Self::D8 => &D8_SCALING,
            Self::La8 => &LA8_SCALING,
            Self::La16 => &LA16_SCALING,
        }
    }

    /// Returns the wavelet (mother wavelet) coefficients.
    ///
    /// Derived from the scaling coefficients via the quadrature mirror
    /// filter relationship.
    pub fn wavelet_coeffs(&self) -> Vec<f64> {
        let g = self.scaling_coeffs();
        let l = g.len();
        (0..l)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 } * g[l - 1 - k])
            .collect()
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

    #[test]
    fn scaling_coeffs_lengths() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            assert_eq!(
                filter.scaling_coeffs().len(),
                filter.length(),
                "scaling_coeffs length mismatch for {:?}",
                filter
            );
        }
    }

    #[test]
    fn scaling_coeffs_sum_to_sqrt2() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let sum: f64 = filter.scaling_coeffs().iter().sum();
            assert!(
                (sum - std::f64::consts::SQRT_2).abs() < 1e-10,
                "scaling coeffs sum for {:?}: {} (expected sqrt(2))",
                filter,
                sum
            );
        }
    }

    #[test]
    fn scaling_coeffs_unit_energy() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let energy: f64 = filter.scaling_coeffs().iter().map(|c| c * c).sum();
            assert!(
                (energy - 1.0).abs() < 1e-10,
                "scaling coeffs energy for {:?}: {} (expected 1.0)",
                filter,
                energy
            );
        }
    }

    #[test]
    fn haar_scaling_coeffs_exact() {
        let coeffs = WaveletFilter::Haar.scaling_coeffs();
        assert_eq!(coeffs[0], std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(coeffs[1], std::f64::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn wavelet_coeffs_sum_to_zero() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let sum: f64 = filter.wavelet_coeffs().iter().sum();
            assert!(
                sum.abs() < 1e-10,
                "wavelet coeffs sum for {:?}: {} (expected ~0)",
                filter,
                sum
            );
        }
    }

    #[test]
    fn wavelet_coeffs_correct_length() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            assert_eq!(
                filter.wavelet_coeffs().len(),
                filter.length(),
                "wavelet_coeffs length mismatch for {:?}",
                filter
            );
        }
    }

    #[test]
    fn wavelet_coeffs_unit_energy() {
        for filter in [
            WaveletFilter::Haar,
            WaveletFilter::D4,
            WaveletFilter::D6,
            WaveletFilter::D8,
            WaveletFilter::La8,
            WaveletFilter::La16,
        ] {
            let energy: f64 = filter.wavelet_coeffs().iter().map(|c| c * c).sum();
            assert!(
                (energy - 1.0).abs() < 1e-12,
                "wavelet coeffs energy for {:?}: {} (expected 1.0)",
                filter,
                energy
            );
        }
    }

    #[test]
    fn haar_wavelet_coeffs_known() {
        let coeffs = WaveletFilter::Haar.wavelet_coeffs();
        assert_eq!(coeffs[0], std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(coeffs[1], -std::f64::consts::FRAC_1_SQRT_2);
    }
}
