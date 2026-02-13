use zeus_calendar::{CalendarError, Doy, NoLeapDate};

#[test]
fn doy_roundtrip_all_365() {
    for d in 1..=365u16 {
        let doy = Doy::new(d).unwrap();
        let (m, day) = doy.month_day();
        let back = Doy::from_month_day(m, day).unwrap();
        assert_eq!(
            back.get(),
            d,
            "roundtrip failed for doy {d}: month_day=({m}, {day})"
        );
    }
}

#[test]
fn noleap_date_from_year_doy_roundtrip() {
    for d in 1..=365u16 {
        let doy = Doy::new(d).unwrap();
        let date = NoLeapDate::from_year_doy(2000, doy);
        assert_eq!(
            date.doy().get(),
            d,
            "doy mismatch for d={d}: got {}",
            date.doy().get()
        );
        let (expected_m, expected_day) = doy.month_day();
        assert_eq!(
            date.month(),
            expected_m,
            "month mismatch for d={d}: expected {expected_m}, got {}",
            date.month()
        );
        assert_eq!(
            date.day(),
            expected_day,
            "day mismatch for d={d}: expected {expected_day}, got {}",
            date.day()
        );
    }
}

#[test]
fn noleap_date_new_preserves_doy() {
    let cases: &[(u8, u8, u16)] = &[
        (1, 1, 1),     // Jan 1
        (2, 28, 59),   // Feb 28
        (3, 1, 60),    // Mar 1
        (7, 4, 185),   // Jul 4
        (12, 31, 365), // Dec 31
    ];
    for &(month, day, expected_doy) in cases {
        let date = NoLeapDate::new(2000, month, day).unwrap();
        assert_eq!(
            date.doy().get(),
            expected_doy,
            "NoLeapDate::new(2000, {month}, {day}).doy() = {}, expected {expected_doy}",
            date.doy().get()
        );
    }
}

#[test]
fn feb_29_rejected() {
    // Doy::from_month_day(2, 29) should fail with InvalidDay
    let err_doy = Doy::from_month_day(2, 29).unwrap_err();
    assert_eq!(
        err_doy,
        CalendarError::InvalidDay {
            day: 29,
            month: 2,
            max_day: 28,
        }
    );

    // NoLeapDate::new(2000, 2, 29) should fail with InvalidDay
    let err_date = NoLeapDate::new(2000, 2, 29).unwrap_err();
    assert_eq!(
        err_date,
        CalendarError::InvalidDay {
            day: 29,
            month: 2,
            max_day: 28,
        }
    );
}
