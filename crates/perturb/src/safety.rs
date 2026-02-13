//! Safety rails for post-processing precipitation values.
//!
//! Cleans up non-finite values, enforces non-negativity, and clamps
//! values to a configurable floor and cap.

/// Applies safety rails to a precipitation series in-place.
///
/// For each value:
/// 1. NaN → 0.0
/// 2. Infinity (positive or negative) → 0.0
/// 3. Negative → `floor` (typically 0.0)
/// 4. Clamp to `[floor, cap]`
///
/// # Arguments
///
/// * `precip` — Mutable precipitation values to clean.
/// * `floor` — Minimum allowed value (typically 0.0).
/// * `cap` — Maximum allowed value (typically 500.0).
pub fn apply_safety_rails(precip: &mut [f64], floor: f64, cap: f64) {
    for val in precip.iter_mut() {
        if !val.is_finite() {
            *val = 0.0;
        }
        if *val < floor {
            *val = floor;
        }
        if *val > cap {
            *val = cap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_replaced_with_zero() {
        let mut data = vec![f64::NAN, 1.0, f64::NAN];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 0.0);
    }

    #[test]
    fn positive_infinity_replaced() {
        let mut data = vec![f64::INFINITY, 5.0];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 5.0);
    }

    #[test]
    fn negative_infinity_replaced() {
        let mut data = vec![f64::NEG_INFINITY, 3.0];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 3.0);
    }

    #[test]
    fn negative_values_clamped_to_floor() {
        let mut data = vec![-5.0, -0.1, 0.0, 10.0];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 10.0);
    }

    #[test]
    fn values_above_cap_clamped() {
        let mut data = vec![100.0, 500.0, 501.0, 1000.0];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data[0], 100.0);
        assert_eq!(data[1], 500.0);
        assert_eq!(data[2], 500.0);
        assert_eq!(data[3], 500.0);
    }

    #[test]
    fn custom_floor() {
        let mut data = vec![0.5, 1.0, 5.0];
        apply_safety_rails(&mut data, 1.0, 500.0);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 5.0);
    }

    #[test]
    fn custom_cap() {
        let mut data = vec![50.0, 100.0, 150.0];
        apply_safety_rails(&mut data, 0.0, 100.0);
        assert_eq!(data[0], 50.0);
        assert_eq!(data[1], 100.0);
        assert_eq!(data[2], 100.0);
    }

    #[test]
    fn clean_data_unchanged() {
        let original = vec![0.0, 1.5, 10.0, 50.0, 100.0, 499.0];
        let mut data = original.clone();
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data, original);
    }

    #[test]
    fn idempotent() {
        let mut data = vec![f64::NAN, -5.0, 600.0, f64::INFINITY, 10.0];
        apply_safety_rails(&mut data, 0.0, 500.0);
        let first_pass = data.clone();
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert_eq!(data, first_pass);
    }

    #[test]
    fn empty_input() {
        let mut data: Vec<f64> = vec![];
        apply_safety_rails(&mut data, 0.0, 500.0);
        assert!(data.is_empty());
    }
}
