//! Three-state Markov chain for daily precipitation occurrence.
//!
//! This crate models the wet/dry/extreme occurrence process using a
//! first-order, three-state Markov chain with monthly varying transition
//! probabilities.
//!
//! # Pipeline
//!
//! ```text
//!  ┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
//!  │  threshold    │────▶│  transition    │────▶│    simulate      │
//!  │  (classify)   │     │  (estimate P)  │     │  (draw states)   │
//!  └──────────────┘     └────────────────┘     └──────────────────┘
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use zeus_markov::{MarkovConfig, PrecipState, ThresholdSpec};
//!
//! // Build a configuration
//! let config = MarkovConfig::new()
//!     .with_wet_spec(ThresholdSpec::Fixed(0.3))
//!     .with_extreme_spec(ThresholdSpec::Quantile(0.8))
//!     .with_dirichlet_alpha(0.5);
//!
//! assert!(config.validate().is_ok());
//! ```

pub mod config;
pub mod error;
pub mod simulate;
pub mod state;
pub mod threshold;
pub mod transition;

pub use config::{MarkovConfig, ThresholdSpec};
pub use error::MarkovError;
pub use simulate::{simulate_states, simulate_states_into};
pub use state::PrecipState;
pub use threshold::StateThresholds;
pub use transition::{MonthlyTransitions, TransitionMatrix, estimate_monthly_transitions};
