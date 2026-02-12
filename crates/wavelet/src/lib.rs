//! # zeus-wavelet
//!
//! Wavelet transforms for weather signal decomposition and reconstruction.
//!
//! ## Analysis Pipeline
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

mod error;
mod filter;
mod modwt;
mod mra;
mod series;

pub use error::WaveletError;
pub use filter::WaveletFilter;
pub use modwt::{ModwtCoeffs, ModwtConfig, imodwt, max_modwt_level, modwt};
pub use mra::{Mra, MraConfig, mra, select_levels};
pub use series::TimeSeries;
