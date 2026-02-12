//! # zeus-arma
//!
//! ARMA(p,q) model fitting and simulation via state-space
//! maximum-likelihood (Kalman filter).
//!
//! ## Typestate Workflow
//!
//! ```mermaid
//! graph LR
//!     A["ArmaSpec::new(p, q)"] -->|".fit(&data)?"| B["ArmaFit"]
//!     B --> C[".ar() — AR coefficients"]
//!     B --> D[".ma() — MA coefficients"]
//!     B --> E[".sigma2() — innovation variance"]
//!     B --> F[".aic() — Akaike Information Criterion"]
//!     B --> G[".simulate(n, n_sim, &mut rng)"]
//!     H["select_best_aic(&data, max_p, max_q)?"] -->|"grid search"| B
//! ```
//!
//! ## Two Usage Paths
//!
//! **Direct fit** (known orders):
//! ```ignore
//! let fit = ArmaSpec::new(2, 1).fit(&data)?;
//! ```
//!
//! **AIC grid search** (unknown orders):
//! ```ignore
//! let fit = select_best_aic(&data, 2, 2)?;
//! ```
//!
//! ## Mathematical Glossary
//!
//! | Symbol | Accessor | Meaning |
//! |--------|----------|---------|
//! | phi | [`ArmaFit::ar()`] | AR coefficients: weights on past observations |
//! | theta | [`ArmaFit::ma()`] | MA coefficients: weights on past forecast errors |
//! | sigma2 | [`ArmaFit::sigma2()`] | Innovation (white-noise) variance |
//! | AIC | [`ArmaFit::aic()`] | Akaike Information Criterion (lower = better) |

mod error;
mod fit;
mod selection;
mod spec;

pub(crate) mod kalman;
pub(crate) mod optimizer;
pub(crate) mod params;
pub(crate) mod state_space;

pub use error::ArmaError;
pub use fit::ArmaFit;
pub use selection::select_best_aic;
pub use spec::ArmaSpec;
