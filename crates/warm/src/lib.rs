//! # zeus-warm
//!
//! Wavelet-based ARMA Recombination Method (WARM) for stochastic weather
//! generation.
//!
//! ## Simulation Pipeline
//!
//! ```mermaid
//! graph LR
//!     A["observed data"] -->|"MRA decomposition"| B["MRA components"]
//!     B -->|"fit ARMA per component"| C["ARMA models"]
//!     C -->|"simulate_warm()"| D["WarmResult"]
//!     D -->|"filter_warm_pool()"| E["FilteredPool"]
//!     E --> F[".selected() — indices"]
//!     E --> G[".scores() — quality"]
//! ```
//!
//! ## Quick Start
//!
//! ```ignore
//! use zeus_wavelet::{WaveletFilter, MraConfig};
//! use zeus_warm::{WarmConfig, simulate_warm, FilterBounds, filter_warm_pool};
//!
//! let mra_config = MraConfig::new(WaveletFilter::La8);
//! let config = WarmConfig::new(mra_config, 1000, 30);
//! let result = simulate_warm(&observed, &config)?;
//! let pool = filter_warm_pool(&observed, &result, &FilterBounds::default())?;
//! ```

mod bootstrap;
mod error;
mod spectral;
mod warm;
mod warm_filter;

pub use error::WarmError;
pub use warm::{WarmConfig, WarmResult, simulate_warm};
pub use warm_filter::{FilterBounds, FilteredPool, filter_warm_pool};
