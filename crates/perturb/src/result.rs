//! Result types for the perturbation pipeline.

/// Records which pipeline modules were applied.
#[derive(Debug, Clone)]
pub struct ModulesApplied {
    /// Whether temperature adjustment was applied.
    pub temperature: bool,
    /// Whether quantile mapping was applied.
    pub quantile_map: bool,
    /// Whether occurrence adjustment was applied.
    pub occurrence: bool,
    /// Whether safety rails were applied.
    pub safety_rails: bool,
}

/// Result of the temperature adjustment step.
#[derive(Debug, Clone)]
pub struct TempResult {
    /// Adjusted mean temperature.
    temp: Vec<f64>,
    /// Adjusted minimum temperature, if provided.
    temp_min: Option<Vec<f64>>,
    /// Adjusted maximum temperature, if provided.
    temp_max: Option<Vec<f64>>,
}

impl TempResult {
    /// Creates a new temperature result.
    pub fn new(temp: Vec<f64>, temp_min: Option<Vec<f64>>, temp_max: Option<Vec<f64>>) -> Self {
        Self {
            temp,
            temp_min,
            temp_max,
        }
    }

    /// Returns the adjusted mean temperature.
    pub fn temp(&self) -> &[f64] {
        &self.temp
    }

    /// Consumes self and returns the owned mean temperature vector.
    pub fn into_temp(self) -> Vec<f64> {
        self.temp
    }

    /// Returns the adjusted minimum temperature, if available.
    pub fn temp_min(&self) -> Option<&[f64]> {
        self.temp_min.as_deref()
    }

    /// Returns the adjusted maximum temperature, if available.
    pub fn temp_max(&self) -> Option<&[f64]> {
        self.temp_max.as_deref()
    }
}

/// Result of the occurrence adjustment step.
#[derive(Debug, Clone)]
pub struct OccurrenceResult {
    /// Adjusted precipitation after occurrence changes.
    precip: Vec<f64>,
    /// Number of wet days added across all months/years.
    days_added: usize,
    /// Number of wet days removed across all months/years.
    days_removed: usize,
}

impl OccurrenceResult {
    /// Creates a new occurrence result.
    pub fn new(precip: Vec<f64>, days_added: usize, days_removed: usize) -> Self {
        Self {
            precip,
            days_added,
            days_removed,
        }
    }

    /// Returns the adjusted precipitation.
    pub fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Consumes self and returns the owned precipitation vector.
    pub fn into_precip(self) -> Vec<f64> {
        self.precip
    }

    /// Returns the number of wet days added.
    pub fn days_added(&self) -> usize {
        self.days_added
    }

    /// Returns the number of wet days removed.
    pub fn days_removed(&self) -> usize {
        self.days_removed
    }
}

/// The complete output of the perturbation pipeline.
#[derive(Debug, Clone)]
pub struct PerturbResult {
    /// Final adjusted precipitation.
    precip: Vec<f64>,
    /// Adjusted mean temperature.
    temp: Vec<f64>,
    /// Adjusted minimum temperature, if provided.
    temp_min: Option<Vec<f64>>,
    /// Adjusted maximum temperature, if provided.
    temp_max: Option<Vec<f64>>,
    /// Which modules were applied.
    modules_applied: ModulesApplied,
}

impl PerturbResult {
    /// Creates a new perturbation result.
    pub fn new(
        precip: Vec<f64>,
        temp: Vec<f64>,
        temp_min: Option<Vec<f64>>,
        temp_max: Option<Vec<f64>>,
        modules_applied: ModulesApplied,
    ) -> Self {
        Self {
            precip,
            temp,
            temp_min,
            temp_max,
            modules_applied,
        }
    }

    /// Returns the final adjusted precipitation.
    pub fn precip(&self) -> &[f64] {
        &self.precip
    }

    /// Consumes self and returns the owned precipitation vector.
    pub fn into_precip(self) -> Vec<f64> {
        self.precip
    }

    /// Returns the adjusted mean temperature.
    pub fn temp(&self) -> &[f64] {
        &self.temp
    }

    /// Returns the adjusted minimum temperature, if available.
    pub fn temp_min(&self) -> Option<&[f64]> {
        self.temp_min.as_deref()
    }

    /// Returns the adjusted maximum temperature, if available.
    pub fn temp_max(&self) -> Option<&[f64]> {
        self.temp_max.as_deref()
    }

    /// Returns information about which modules were applied.
    pub fn modules_applied(&self) -> &ModulesApplied {
        &self.modules_applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_result_accessors() {
        let tr = TempResult::new(vec![1.0, 2.0], Some(vec![0.5, 1.5]), Some(vec![1.5, 2.5]));
        assert_eq!(tr.temp(), &[1.0, 2.0]);
        assert_eq!(tr.temp_min(), Some([0.5, 1.5].as_slice()));
        assert_eq!(tr.temp_max(), Some([1.5, 2.5].as_slice()));
    }

    #[test]
    fn temp_result_no_minmax() {
        let tr = TempResult::new(vec![1.0], None, None);
        assert!(tr.temp_min().is_none());
        assert!(tr.temp_max().is_none());
    }

    #[test]
    fn temp_result_into_temp() {
        let tr = TempResult::new(vec![1.0, 2.0, 3.0], None, None);
        let owned = tr.into_temp();
        assert_eq!(owned, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn occurrence_result_accessors() {
        let or = OccurrenceResult::new(vec![0.0, 5.0, 0.0], 1, 2);
        assert_eq!(or.precip(), &[0.0, 5.0, 0.0]);
        assert_eq!(or.days_added(), 1);
        assert_eq!(or.days_removed(), 2);
    }

    #[test]
    fn occurrence_result_into_precip() {
        let or = OccurrenceResult::new(vec![1.0, 2.0], 0, 0);
        let owned = or.into_precip();
        assert_eq!(owned, vec![1.0, 2.0]);
    }

    #[test]
    fn perturb_result_accessors() {
        let pr = PerturbResult::new(
            vec![0.0, 5.0],
            vec![20.0, 22.0],
            Some(vec![15.0, 17.0]),
            Some(vec![25.0, 27.0]),
            ModulesApplied {
                temperature: true,
                quantile_map: false,
                occurrence: true,
                safety_rails: true,
            },
        );
        assert_eq!(pr.precip(), &[0.0, 5.0]);
        assert_eq!(pr.temp(), &[20.0, 22.0]);
        assert_eq!(pr.temp_min(), Some([15.0, 17.0].as_slice()));
        assert_eq!(pr.temp_max(), Some([25.0, 27.0].as_slice()));
        assert!(pr.modules_applied().temperature);
        assert!(!pr.modules_applied().quantile_map);
        assert!(pr.modules_applied().occurrence);
        assert!(pr.modules_applied().safety_rails);
    }

    #[test]
    fn perturb_result_into_precip() {
        let pr = PerturbResult::new(
            vec![1.0, 2.0, 3.0],
            vec![20.0, 21.0, 22.0],
            None,
            None,
            ModulesApplied {
                temperature: false,
                quantile_map: false,
                occurrence: false,
                safety_rails: false,
            },
        );
        let owned = pr.into_precip();
        assert_eq!(owned, vec![1.0, 2.0, 3.0]);
    }
}
