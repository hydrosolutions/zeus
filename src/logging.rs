use tracing_subscriber::EnvFilter;

/// All workspace crate targets that should receive log output.
const CRATE_TARGETS: &[&str] = &[
    "zeus",
    "zeus_arma",
    "zeus_calendar",
    "zeus_evaluate",
    "zeus_io",
    "zeus_knn",
    "zeus_markov",
    "zeus_perturb",
    "zeus_pet",
    "zeus_quantile_map",
    "zeus_resample",
    "zeus_stats",
    "zeus_warm",
    "zeus_wavelet",
];

/// Initialize tracing based on CLI verbosity level.
///
/// Mapping:
/// - 0 (none) -> warn
/// - 1 (-v)   -> info
/// - 2 (-vv)  -> debug
/// - 3+ (-vvv)-> trace
///
/// `RUST_LOG` env var overrides the CLI flag if set.
pub fn init(verbosity: u8) {
    let level = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let default_filter: String = CRATE_TARGETS
        .iter()
        .map(|t| format!("{t}={level}"))
        .collect::<Vec<_>>()
        .join(",");

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    tracing_subscriber::fmt().with_env_filter(filter).init();
}
