//! Output type for KNN sampling queries.

/// Result of a KNN sampling query.
///
/// Contains the sampled candidate indices and the Euclidean distances
/// of the k nearest neighbors (sorted ascending).
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// Sampled candidate indices (length = config.n).
    /// These index into the original candidate rows `[0..n_candidates)`.
    indices: Vec<usize>,
    /// Euclidean distances of the k nearest neighbors (sorted ascending).
    /// Length = `min(config.k, n_candidates)`.
    nn_distances: Vec<f64>,
}

impl KnnResult {
    /// Creates a new `KnnResult`.
    pub(crate) fn new(indices: Vec<usize>, nn_distances: Vec<f64>) -> Self {
        Self {
            indices,
            nn_distances,
        }
    }

    /// Returns the sampled candidate indices.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the Euclidean distances of the k nearest neighbors.
    pub fn nn_distances(&self) -> &[f64] {
        &self.nn_distances
    }

    /// Returns the first sampled index.
    ///
    /// This is a convenience method for the common case where `n = 1`.
    pub fn index(&self) -> usize {
        self.indices[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessors() {
        let result = KnnResult::new(vec![42], vec![1.0, 2.5, 3.7]);
        assert_eq!(result.indices(), &[42]);
        assert_eq!(result.nn_distances(), &[1.0, 2.5, 3.7]);
        assert_eq!(result.index(), 42);
    }

    #[test]
    fn test_multiple_indices() {
        let result = KnnResult::new(vec![10, 20, 30], vec![0.5, 1.2]);
        assert_eq!(result.indices().len(), 3);
        assert_eq!(result.indices(), &[10, 20, 30]);
        assert_eq!(result.index(), 10);
    }
}
