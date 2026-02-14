//! # zeus-io
//!
//! Read observed climate data from NetCDF files and write synthetic weather
//! output to Parquet. Bridges external file formats into Zeus's internal
//! `&[f64]` slice-based APIs.

mod error;
mod multi_site;
mod netcdf_read;
mod observed;
mod owned_synthetic;
mod parquet_read;
mod parquet_write;
mod reader;
mod synthetic;
mod validate;
mod writer;

pub use error::IoError;
pub use multi_site::{GridMetadata, MultiSiteData};
pub use observed::ObservedData;
pub use owned_synthetic::OwnedSyntheticWeather;
pub use reader::{ReaderConfig, read_netcdf, read_parquet};
pub use synthetic::SyntheticWeather;
pub use writer::{Compression, WriterConfig, write_parquet};
