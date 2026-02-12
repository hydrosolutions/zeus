//! ARMA state-space representation.
//!
//! Converts ARMA(p,q) coefficients into state-space form:
//!
//! ```text
//! x[t+1] = T * x[t] + R * e[t]     (state transition)
//! y[t]   = Z' * x[t]                (observation)
//! ```
//!
//! where `T` is the transition matrix, `R` the noise-input vector,
//! `Z` the observation vector, and `e[t] ~ N(0, sigma2)`.
//!
//! **Not part of the public API.**
