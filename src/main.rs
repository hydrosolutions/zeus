mod cli;
mod config;
mod convert;
mod evaluate_cmd;
mod generate;
mod logging;
mod perturb_cmd;

use std::process;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};

fn main() {
    let cli = Cli::parse();
    logging::init(cli.verbose);

    if let Err(e) = run(cli.command) {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Generate(args) => generate::run(args),
        Command::Perturb(args) => perturb_cmd::run(args),
        Command::Evaluate(args) => evaluate_cmd::run(args),
    }
}
