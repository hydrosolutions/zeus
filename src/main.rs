mod cli;
mod config;
mod convert;
mod evaluate_cmd;
mod generate;
mod logging;

use std::process;

use anyhow::{Context, Result};
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::config::ZeusConfig;

fn main() {
    let cli = Cli::parse();
    logging::init(cli.verbose);

    if let Err(e) = run(&cli) {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<()> {
    let toml_str = std::fs::read_to_string(&cli.config)
        .with_context(|| format!("failed to read config file: {}", cli.config.display()))?;
    let mut config: ZeusConfig =
        toml::from_str(&toml_str).context("failed to parse TOML config")?;

    // CLI overrides
    if let Some(ref input) = cli.input {
        config.io.input = Some(input.clone());
    }
    if let Some(ref output) = cli.output {
        config.io.output = Some(output.clone());
    }
    if let Some(seed) = cli.seed {
        config.seed = Some(seed);
    }

    match &cli.command {
        Command::Generate => generate::run(cli, &config),
        Command::Evaluate { synthetic } => evaluate_cmd::run(cli, &config, synthetic),
    }
}
