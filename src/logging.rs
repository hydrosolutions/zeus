use tracing_subscriber::EnvFilter;

/// Initialize tracing based on CLI verbosity level.
///
/// Mapping:
/// - 0 (none) -> zeus=warn
/// - 1 (-v)   -> zeus=info
/// - 2 (-vv)  -> zeus=debug
/// - 3+ (-vvv)-> zeus=trace
///
/// `RUST_LOG` env var overrides the CLI flag if set.
pub fn init(verbosity: u8) {
    let default_filter = match verbosity {
        0 => "zeus=warn",
        1 => "zeus=info",
        2 => "zeus=debug",
        _ => "zeus=trace",
    };

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    tracing_subscriber::fmt().with_env_filter(filter).init();
}
