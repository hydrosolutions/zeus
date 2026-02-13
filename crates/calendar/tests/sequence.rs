use zeus_calendar::{NoLeapDate, noleap_sequence};

#[test]
fn full_year_month_boundaries() {
    let start = NoLeapDate::new(2000, 1, 1).unwrap();
    let dates = noleap_sequence(start, 365);
    assert_eq!(dates.len(), 365);

    // Index 0: Jan 1
    assert_eq!(dates[0].year(), 2000);
    assert_eq!(dates[0].month(), 1);
    assert_eq!(dates[0].day(), 1);

    // Index 30: Jan 31
    assert_eq!(dates[30].year(), 2000);
    assert_eq!(dates[30].month(), 1);
    assert_eq!(dates[30].day(), 31);

    // Index 31: Feb 1
    assert_eq!(dates[31].year(), 2000);
    assert_eq!(dates[31].month(), 2);
    assert_eq!(dates[31].day(), 1);

    // Index 58: Feb 28
    assert_eq!(dates[58].year(), 2000);
    assert_eq!(dates[58].month(), 2);
    assert_eq!(dates[58].day(), 28);

    // Index 59: Mar 1
    assert_eq!(dates[59].year(), 2000);
    assert_eq!(dates[59].month(), 3);
    assert_eq!(dates[59].day(), 1);

    // Index 364: Dec 31
    assert_eq!(dates[364].year(), 2000);
    assert_eq!(dates[364].month(), 12);
    assert_eq!(dates[364].day(), 31);
}

#[test]
fn multi_year_transitions() {
    let start = NoLeapDate::new(2000, 1, 1).unwrap();
    let dates = noleap_sequence(start, 730);
    assert_eq!(dates.len(), 730);

    // Index 364: Dec 31, 2000
    assert_eq!(dates[364].year(), 2000);
    assert_eq!(dates[364].month(), 12);
    assert_eq!(dates[364].day(), 31);

    // Index 365: Jan 1, 2001
    assert_eq!(dates[365].year(), 2001);
    assert_eq!(dates[365].month(), 1);
    assert_eq!(dates[365].day(), 1);

    // Index 729: Dec 31, 2001
    assert_eq!(dates[729].year(), 2001);
    assert_eq!(dates[729].month(), 12);
    assert_eq!(dates[729].day(), 31);
}

#[test]
fn length_always_matches() {
    let start = NoLeapDate::new(2000, 1, 1).unwrap();
    for n_days in [0, 1, 100, 365, 730, 1000] {
        let dates = noleap_sequence(start, n_days);
        assert_eq!(
            dates.len(),
            n_days,
            "expected length {n_days}, got {}",
            dates.len()
        );
    }
}
