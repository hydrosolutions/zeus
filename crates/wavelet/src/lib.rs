//! # zeus-wavelet
//!
//! Wavelet transforms for weather signal decomposition and reconstruction.
//!
//! Provides two complementary wavelet analysis paths:
//!
//! - **MODWT / MRA** --- discrete multiresolution decomposition for additive
//!   signal separation.
//! - **CWT / Significance** --- continuous wavelet transform for time-frequency
//!   power analysis with statistical significance testing.
//!
//! Both paths combine via [`analyze_additive()`] for a single-call pipeline.
//!
//! ## Analysis Pipelines
//!
//! ```mermaid
//! graph LR
//!     A["TimeSeries::new(data)?"] -->|"validate"| B["TimeSeries"]
//!     B -->|"modwt(&ts, &config)?"| C["ModwtCoeffs"]
//!     C -->|"imodwt(&coeffs)?"| B
//!     B -->|"mra(&ts, &config)?"| D["Mra"]
//!     D --> E[".detail(level)"]
//!     D --> F[".smooth()"]
//!     D --> G[".variance_fractions()"]
//!     D --> H[".to_matrix()"]
//!     B -->|"cwt_morlet(&ts, &config)?"| I["CwtResult"]
//!     I --> J[".power()"]
//!     I --> K[".global_wavelet_spectrum()"]
//!     I -->|"test_significance(&ts, &cwt, &config)?"| L["GwsResult"]
//!     L --> M[".significant_periods()"]
//!     B -->|"analyze_additive(&ts, &config)?"| N["WaveletAdditive"]
//!     N --> O[".mra()"]
//!     N --> P[".significant_levels()"]
//!     N --> Q[".gws_result()"]
//! ```
//!
//! ## Supported Filters
//!
//! | Filter | Length | Family |
//! |--------|--------|--------|
//! | [`WaveletFilter::Haar`] | 2 | Haar |
//! | [`WaveletFilter::D4`] | 4 | Daubechies |
//! | [`WaveletFilter::D6`] | 6 | Daubechies |
//! | [`WaveletFilter::D8`] | 8 | Daubechies |
//! | [`WaveletFilter::La8`] | 8 | Least Asymmetric |
//! | [`WaveletFilter::La16`] | 16 | Least Asymmetric |
//!
//! ## Quick Start
//!
//! ```ignore
//! use zeus_wavelet::{TimeSeries, WaveletFilter, MraConfig, mra};
//!
//! let ts = TimeSeries::new(data)?;
//! let config = MraConfig::new(WaveletFilter::La8);
//! let result = mra(&ts, &config)?;
//!
//! for component in result.components() {
//!     println!("len = {}", component.len());
//! }
//! ```

mod additive;
mod cwt;
mod error;
mod filter;
mod modwt;
mod mra;
mod series;
mod significance;

// --- Re-export Complex for CwtResult consumers ---
pub use num_complex::Complex;

// --- Error ---
pub use error::WaveletError;

// --- Core types ---
pub use filter::WaveletFilter;
pub use series::TimeSeries;

// --- MODWT / MRA ---
pub use modwt::{ModwtCoeffs, ModwtConfig, imodwt, max_modwt_level, modwt};
pub use mra::{Mra, MraConfig, mra, select_levels};

// --- CWT ---
pub use cwt::{CwtConfig, CwtResult, cwt_morlet};

// --- Significance testing ---
pub use significance::{GwsResult, NoiseModel, SignificanceConfig, test_significance};

// --- Additive decomposition pipeline ---
pub use additive::{AdditiveConfig, WaveletAdditive, analyze_additive};
