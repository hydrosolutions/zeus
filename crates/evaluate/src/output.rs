//! JSON output structures for evaluation results.

use crate::error::EvaluateError;
use serde::Serialize;
use std::collections::BTreeMap;

/// Top-level evaluation output.
#[derive(Debug, Serialize)]
pub struct EvaluationOutput {
    /// Configuration summary.
    pub config: ConfigSummary,
    /// Per-realisation scores, ranked.
    pub scorecard: Vec<RealisationScore>,
    /// Detailed diagnostic comparisons.
    pub diagnostics: Diagnostics,
}

/// Summary of the configuration used.
#[derive(Debug, Serialize)]
pub struct ConfigSummary {
    pub precip_threshold: f64,
    pub n_sites: usize,
    pub n_realisations: usize,
    pub variables: Vec<String>,
}

/// Score for a single realisation.
#[derive(Debug, Clone, Serialize)]
pub struct RealisationScore {
    pub realisation: u32,
    pub raw_mae: f64,
    pub normalized_score: f64,
    pub rank: usize,
}

/// Detailed diagnostic data.
#[derive(Debug, Serialize)]
pub struct Diagnostics {
    /// site -> month -> variable -> comparison
    pub timeseries: BTreeMap<String, BTreeMap<String, BTreeMap<String, TimeseriesComparison>>>,
    pub cross_grid_correlations: Vec<CrossGridEntry>,
    pub inter_variable_correlations: Vec<InterVarEntry>,
    pub conditional_correlations: Vec<ConditionalEntry>,
}

/// Comparison of observed vs simulated timeseries stats.
#[derive(Debug, Clone, Serialize)]
pub struct TimeseriesComparison {
    pub observed: TimeseriesStatsOutput,
    pub simulated_mean: TimeseriesStatsOutput,
}

/// Summary statistics for a time series.
#[derive(Debug, Clone, Serialize)]
pub struct TimeseriesStatsOutput {
    pub mean: f64,
    pub sd: f64,
    pub skewness: Option<f64>,
    pub wet_days: usize,
    pub dry_days: usize,
    pub mean_wet_spell: f64,
    pub mean_dry_spell: f64,
}

/// Cross-grid correlation entry.
#[derive(Debug, Clone, Serialize)]
pub struct CrossGridEntry {
    pub site_a: String,
    pub site_b: String,
    pub variable: String,
    pub observed_correlation: Option<f64>,
    pub simulated_correlation: Option<f64>,
}

/// Inter-variable correlation entry.
#[derive(Debug, Clone, Serialize)]
pub struct InterVarEntry {
    pub site: String,
    pub variable_a: String,
    pub variable_b: String,
    pub observed_correlation: Option<f64>,
    pub simulated_correlation: Option<f64>,
}

/// Conditional correlation entry.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionalEntry {
    pub site: String,
    pub variable: String,
    pub regime: String,
    pub observed_correlation: Option<f64>,
    pub simulated_correlation: Option<f64>,
}

/// Serialize evaluation output to a JSON string.
pub fn to_json(output: &EvaluationOutput) -> Result<String, EvaluateError> {
    serde_json::to_string_pretty(output).map_err(|e| EvaluateError::Serialization {
        reason: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_json_roundtrip() {
        let output = EvaluationOutput {
            config: ConfigSummary {
                precip_threshold: 0.01,
                n_sites: 2,
                n_realisations: 10,
                variables: vec!["precip".to_string(), "temp_max".to_string()],
            },
            scorecard: vec![RealisationScore {
                realisation: 0,
                raw_mae: 1.5,
                normalized_score: 0.85,
                rank: 1,
            }],
            diagnostics: Diagnostics {
                timeseries: BTreeMap::new(),
                cross_grid_correlations: vec![],
                inter_variable_correlations: vec![],
                conditional_correlations: vec![],
            },
        };

        let json = to_json(&output).unwrap();
        assert!(json.contains("\"precip_threshold\": 0.01"));
        assert!(json.contains("\"n_sites\": 2"));
        assert!(json.contains("\"n_realisations\": 10"));
        assert!(json.contains("\"scorecard\""));
        assert!(json.contains("\"diagnostics\""));
    }

    #[test]
    fn test_config_summary_serializes() {
        let config = ConfigSummary {
            precip_threshold: 1.0,
            n_sites: 5,
            n_realisations: 100,
            variables: vec!["var1".to_string(), "var2".to_string()],
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"precip_threshold\":1.0"));
        assert!(json.contains("\"n_sites\":5"));
        assert!(json.contains("\"n_realisations\":100"));
        assert!(json.contains("\"var1\""));
        assert!(json.contains("\"var2\""));
    }

    #[test]
    fn test_realisation_score_serializes() {
        let score = RealisationScore {
            realisation: 42,
            raw_mae: 2.5,
            normalized_score: 0.75,
            rank: 3,
        };

        let json = serde_json::to_string(&score).unwrap();
        assert!(json.contains("\"realisation\":42"));
        assert!(json.contains("\"raw_mae\":2.5"));
        assert!(json.contains("\"normalized_score\":0.75"));
        assert!(json.contains("\"rank\":3"));
    }
}
