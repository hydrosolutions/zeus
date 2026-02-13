//! # zeus-calendar
//!
//! Pure date arithmetic for the 365-day no-leap calendar.
//!
//! ## Architecture
//!
//! ```mermaid
//! graph LR
//!     A["Doy (1..=365)"] -->|".month_day()"| B["(month, day)"]
//!     B -->|"Doy::from_month_day()"| A
//!     A -->|"NoLeapDate::from_year_doy()"| C["NoLeapDate"]
//!     C -->|".next()"| C
//!     C -->|"noleap_sequence()"| D["Vec of NoLeapDate"]
//!     C -->|".month()"| E["water_year()"]
//!     F["base indices"] -->|"expand_indices()"| G["expanded indices"]
//! ```
//!
//! ## Quick Start
//!
//! ```ignore
//! use zeus_calendar::{Doy, NoLeapDate, noleap_sequence, water_year, expand_indices};
//!
//! // Day-of-year conversions
//! let doy = Doy::from_month_day(3, 15).unwrap(); // Mar 15 → DOY 74
//! assert_eq!(doy.get(), 74);
//!
//! // Date construction and sequencing
//! let start = NoLeapDate::new(2000, 1, 1).unwrap();
//! let dates = noleap_sequence(start, 365);
//!
//! // Water year
//! let wy = water_year(2000, 10, 10).unwrap(); // Oct start → WY 2001
//!
//! // Index expansion
//! let expanded = expand_indices(&[100], &[-3, -2, -1, 0, 1, 2, 3], 365);
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `doy` | Day-of-year newtype and conversion tables |
//! | `date` | No-leap date with year context |
//! | `sequence` | Date sequence generation |
//! | `water_year` | Water year computation |
//! | `expand` | Index expansion for candidate day selection |
//! | `error` | Error types |

mod date;
mod doy;
mod error;
mod expand;
mod sequence;
mod water_year;

pub use date::NoLeapDate;
pub use doy::Doy;
pub use error::CalendarError;
pub use expand::expand_indices;
pub use sequence::noleap_sequence;
pub use water_year::water_year;
