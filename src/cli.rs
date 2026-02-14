use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Zeus stochastic multisite weather generator.
#[derive(Parser)]
#[command(
    name = "zeus",
    version,
    about = "Stochastic multisite weather generator"
)]
pub struct Cli {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "zeus.toml")]
    pub config: PathBuf,

    /// Override input NetCDF path from config.
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Override output Parquet path from config.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Override global RNG seed from config.
    #[arg(short, long)]
    pub seed: Option<u64>,

    /// Increase verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Run the full generation pipeline.
    Generate,
    /// Evaluate synthetic output against observed data.
    Evaluate {
        /// Path to synthetic Parquet file.
        #[arg(short = 's', long)]
        synthetic: PathBuf,
    },
}
