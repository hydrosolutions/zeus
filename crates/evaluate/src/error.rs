//! Evaluation error types.

/// Errors that can occur during evaluation.
#[derive(Debug, thiserror::Error)]
pub enum EvaluateError {
    /// One or more validation checks failed.
    #[error("{count} validation error(s): {details}")]
    Validation { count: usize, details: String },

    /// A required site was not found.
    #[error("site '{site}' not found in {location}")]
    MissingSite { site: String, location: String },

    /// JSON serialization failed.
    #[error("serialization error: {reason}")]
    Serialization { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_display() {
        let err = EvaluateError::Validation {
            count: 2,
            details: "missing data".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("2 validation error(s)"));
        assert!(msg.contains("missing data"));
    }

    #[test]
    fn test_missing_site_display() {
        let err = EvaluateError::MissingSite {
            site: "site1".to_string(),
            location: "observed".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("site 'site1'"));
        assert!(msg.contains("not found in observed"));
    }

    #[test]
    fn test_serialization_display() {
        let err = EvaluateError::Serialization {
            reason: "invalid JSON".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("serialization error"));
        assert!(msg.contains("invalid JSON"));
    }
}
