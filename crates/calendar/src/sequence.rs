//! Date sequence generation for the no-leap calendar.

use crate::date::NoLeapDate;

/// Generates a contiguous sequence of no-leap dates.
///
/// Starting from `start`, produces exactly `n_days` consecutive dates by
/// repeatedly advancing to the next day. Year boundaries are handled
/// automatically (Dec 31 wraps to Jan 1 of the following year).
///
/// # Example
///
/// ```ignore
/// let start = NoLeapDate::new(2000, 12, 30).unwrap();
/// let dates = noleap_sequence(start, 4);
/// assert_eq!(dates.len(), 4);
/// // Dec 30, Dec 31, Jan 1 (2001), Jan 2 (2001)
/// ```
pub fn noleap_sequence(start: NoLeapDate, n_days: usize) -> Vec<NoLeapDate> {
    let mut dates = Vec::with_capacity(n_days);
    if n_days == 0 {
        return dates;
    }
    dates.push(start);
    let mut current = start;
    for _ in 1..n_days {
        current = current.next();
        dates.push(current);
    }
    dates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let start = NoLeapDate::new(2000, 1, 1).unwrap();
        let dates = noleap_sequence(start, 0);
        assert!(dates.is_empty());
    }

    #[test]
    fn single() {
        let start = NoLeapDate::new(2000, 6, 15).unwrap();
        let dates = noleap_sequence(start, 1);
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0], start);
    }

    #[test]
    fn full_year() {
        let start = NoLeapDate::new(2000, 1, 1).unwrap();
        let dates = noleap_sequence(start, 365);
        assert_eq!(dates.len(), 365);
        assert_eq!(dates[0], NoLeapDate::new(2000, 1, 1).unwrap());
        let last = dates.last().unwrap();
        assert_eq!(last.year(), 2000);
        assert_eq!(last.month(), 12);
        assert_eq!(last.day(), 31);
    }

    #[test]
    fn multi_year() {
        let start = NoLeapDate::new(2000, 1, 1).unwrap();
        let dates = noleap_sequence(start, 730);
        assert_eq!(dates.len(), 730);
        // Day 366 (index 365) should be Jan 1 of the next year.
        let day_366 = dates[365];
        assert_eq!(day_366.year(), 2001);
        assert_eq!(day_366.month(), 1);
        assert_eq!(day_366.day(), 1);
    }

    #[test]
    fn mid_year_start() {
        let start = NoLeapDate::new(2000, 7, 15).unwrap();
        let dates = noleap_sequence(start, 10);
        assert_eq!(dates.len(), 10);
        assert_eq!(dates[0], NoLeapDate::new(2000, 7, 15).unwrap());
        assert_eq!(
            *dates.last().unwrap(),
            NoLeapDate::new(2000, 7, 24).unwrap()
        );
    }

    #[test]
    fn year_transition() {
        let start = NoLeapDate::new(2000, 12, 30).unwrap();
        let dates = noleap_sequence(start, 4);
        assert_eq!(dates.len(), 4);
        assert_eq!(dates[0], NoLeapDate::new(2000, 12, 30).unwrap());
        assert_eq!(dates[1], NoLeapDate::new(2000, 12, 31).unwrap());
        assert_eq!(dates[2], NoLeapDate::new(2001, 1, 1).unwrap());
        assert_eq!(dates[3], NoLeapDate::new(2001, 1, 2).unwrap());
    }

    #[test]
    fn feb_28_to_mar_1() {
        let start = NoLeapDate::new(2000, 2, 27).unwrap();
        let dates = noleap_sequence(start, 3);
        assert_eq!(dates.len(), 3);
        assert_eq!(dates[0], NoLeapDate::new(2000, 2, 27).unwrap());
        assert_eq!(dates[1], NoLeapDate::new(2000, 2, 28).unwrap());
        assert_eq!(dates[2], NoLeapDate::new(2000, 3, 1).unwrap());
    }

    #[test]
    fn negative_year() {
        let start = NoLeapDate::new(-1, 12, 31).unwrap();
        let dates = noleap_sequence(start, 2);
        assert_eq!(dates.len(), 2);
        assert_eq!(dates[0], NoLeapDate::new(-1, 12, 31).unwrap());
        assert_eq!(dates[1], NoLeapDate::new(0, 1, 1).unwrap());
    }
}
