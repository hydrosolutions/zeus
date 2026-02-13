//! Index expansion for candidate day selection.

/// Expands a set of base indices by a set of offsets, clipping to `[0, n_max)`.
///
/// Computes the Cartesian sum `{b + o | b in base, o in offsets}`, clips each
/// result to the range `[0, n_max)`, then deduplicates and sorts the output.
///
/// This is used to expand candidate days by +/-k windows around each base index.
///
/// # Example
///
/// ```ignore
/// let indices = expand_indices(&[10, 20], &[-1, 0, 1], 365);
/// assert_eq!(indices, vec![9, 10, 11, 19, 20, 21]);
/// ```
pub fn expand_indices(base: &[usize], offsets: &[i32], n_max: usize) -> Vec<usize> {
    if n_max == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(base.len() * offsets.len());
    for &b in base {
        for &o in offsets {
            let idx = b as isize + o as isize;
            if idx >= 0 {
                let idx = idx as usize;
                if idx < n_max {
                    result.push(idx);
                }
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let indices = expand_indices(&[10, 20], &[-1, 0, 1], 365);
        assert_eq!(indices, vec![9, 10, 11, 19, 20, 21]);
    }

    #[test]
    fn clip_lower() {
        let indices = expand_indices(&[0], &[-5, -1, 0, 1], 365);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn clip_upper() {
        let indices = expand_indices(&[363, 364], &[0, 1, 2], 365);
        // 363+0=363, 363+1=364, 363+2=365 (clipped)
        // 364+0=364, 364+1=365 (clipped), 364+2=366 (clipped)
        assert_eq!(indices, vec![363, 364]);
    }

    #[test]
    fn dedup() {
        let indices = expand_indices(&[5, 6], &[-1, 0, 1], 365);
        // 5-1=4, 5+0=5, 5+1=6, 6-1=5, 6+0=6, 6+1=7
        // After dedup: [4, 5, 6, 7]
        assert_eq!(indices, vec![4, 5, 6, 7]);
    }

    #[test]
    fn sorted_output() {
        let indices = expand_indices(&[20, 10, 30], &[2, -2, 0], 365);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
    }

    #[test]
    fn empty_base() {
        let indices = expand_indices(&[], &[-1, 0, 1], 365);
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn empty_offsets() {
        let indices = expand_indices(&[10], &[], 365);
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn overlapping_windows() {
        let indices = expand_indices(&[10, 12], &[-2, -1, 0, 1, 2], 365);
        // 10: 8,9,10,11,12
        // 12: 10,11,12,13,14
        // Union: 8,9,10,11,12,13,14
        assert_eq!(indices, vec![8, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn plus_minus_3_realistic() {
        let indices = expand_indices(&[100], &[-3, -2, -1, 0, 1, 2, 3], 365);
        let expected: Vec<usize> = (97..=103).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn plus_minus_30_realistic() {
        let offsets: Vec<i32> = (-30..=30).collect();
        let indices = expand_indices(&[5], &offsets, 365);
        // 5 + (-30) = -25 (clipped) ... 5 + (-6) = -1 (clipped)
        // 5 + (-5) = 0 ... 5 + 30 = 35
        let expected: Vec<usize> = (0..=35).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn successor_constraint_pattern() {
        // Show that a caller can exclude an index by filtering the result.
        let indices = expand_indices(&[10, 20], &[-1, 0, 1], 365);
        let excluded = 10usize;
        let filtered: Vec<usize> = indices.into_iter().filter(|&i| i != excluded).collect();
        assert_eq!(filtered, vec![9, 11, 19, 20, 21]);
    }

    #[test]
    fn n_max_zero() {
        let indices = expand_indices(&[0], &[0], 0);
        assert_eq!(indices, vec![]);
    }
}
