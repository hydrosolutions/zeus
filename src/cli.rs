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
    /// Increase verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Run the full generation pipeline.
    Generate(GenerateArgs),
    /// Apply climate perturbations to existing synthetic data.
    Perturb(PerturbArgs),
    /// Evaluate synthetic output against observed data.
    Evaluate(EvaluateArgs),
}

/// Arguments for the `generate` subcommand.
#[derive(clap::Args)]
pub struct GenerateArgs {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "zeus.toml")]
    pub config: PathBuf,

    /// Override output Parquet path from config.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Override global RNG seed from config.
    #[arg(short, long)]
    pub seed: Option<u64>,
}

/// Arguments for the `perturb` subcommand.
#[derive(clap::Args)]
pub struct PerturbArgs {
    /// Path to input synthetic Parquet file.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Path for perturbed output Parquet file.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Path to perturbation TOML config file.
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Uniform temperature delta (degrees) applied to all months.
    #[arg(long = "temp-delta", visible_alias = "dt")]
    pub temp_delta: Option<f64>,

    /// Uniform precipitation scaling factor applied to all months.
    #[arg(long = "precip-factor", visible_alias = "dp")]
    pub precip_factor: Option<f64>,
}

/// Arguments for the `evaluate` subcommand.
#[derive(clap::Args)]
pub struct EvaluateArgs {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "zeus.toml")]
    pub config: PathBuf,

    /// Path to synthetic Parquet file to evaluate.
    #[arg(long)]
    pub synthetic: PathBuf,

    /// Path for diagnostics JSON output.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}
