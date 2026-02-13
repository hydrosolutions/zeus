//! Daily disaggregation for the Zeus weather generator.
//!
//! This crate selects historical daily observation indices using a three-tier
//! hybrid system: annual KNN, Markov-conditioned state sequences, and daily
//! KNN on weather anomalies.
//!
//! # Pipeline
//!
//! ```text
//!  ┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
//!  │  Annual KNN   │────▶│  Year Subset   │────▶│   Daily Loop     │
//!  │  (year select) │     │  (stats/markov) │     │  (KNN + fallback)│
//!  └──────────────┘     └────────────────┘     └──────────────────┘
//! ```
//!
//! # Quick start
//!
//! ```ignore
//! use zeus_resample::{ObsData, ResampleConfig, resample_dates};
//! use zeus_markov::MarkovConfig;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let obs = ObsData::new(&precip, &temp, &months, &days, &water_years)?;
//! let config = ResampleConfig::new().with_year_start_month(10);
//! let markov_config = MarkovConfig::new();
//! let mut rng = StdRng::seed_from_u64(42);
//! let indices = resample_dates(&sim_precip, &obs, &markov_config, &config, &mut rng)?;
//! ```

mod annual;
mod config;
mod daily;
mod error;
mod obs_data;
mod resample;
mod result;
mod year_subset;

pub use config::ResampleConfig;
pub use error::ResampleError;
pub use obs_data::ObsData;
pub use resample::{resample_dates, resample_year};
pub use result::ResampleResult;
pub use zeus_knn::Sampling;
pub use zeus_markov::{MarkovConfig, PrecipState};
