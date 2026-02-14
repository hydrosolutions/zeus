use std::path::Path;

use anyhow::{Context, Result, bail};
use tracing::info;

use zeus_io::read_netcdf;

use crate::cli::Cli;
use crate::config::ZeusConfig;
use crate::convert;

/// Run the standalone evaluation pipeline.
///
/// For V1, this command is a placeholder — inline evaluation in `zeus generate`
/// is the primary path. A future version will add `read_parquet()` to zeus-io
/// to support this command fully.
pub fn run(_cli: &Cli, config: &ZeusConfig, synthetic: &Path) -> Result<()> {
    let input =
        config.io.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no input path: set [io].input in config or use --input")
        })?;

    let reader_cfg = convert::build_reader_config(&config.io)?;

    info!(path = %input.display(), "reading observed data");
    let _multi_site = read_netcdf(input, &reader_cfg)
        .with_context(|| format!("failed to read NetCDF: {}", input.display()))?;

    bail!(
        "parquet reader not yet implemented — use inline evaluation from `zeus generate`\n\
         Synthetic file was: {}",
        synthetic.display()
    );
}
