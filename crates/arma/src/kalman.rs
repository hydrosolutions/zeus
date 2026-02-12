//! Kalman filter for ARMA likelihood evaluation.
//!
//! Implements a univariate Kalman filter operating on the state-space
//! representation from [`crate::state_space`]. Used internally by
//! [`ArmaSpec::fit()`](crate::ArmaSpec::fit) to evaluate the exact
//! Gaussian log-likelihood via prediction error decomposition.
//!
//! **Not part of the public API.**
