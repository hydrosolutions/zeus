use zeus_calendar::water_year;

#[test]
fn all_months_october_start() {
    // With start_month=10 (October), months 1-9 belong to the current year,
    // months 10-12 roll into the next water year.
    for month in 1..=9_u8 {
        assert_eq!(
            water_year(2000, month, 10).unwrap(),
            2000,
            "month {month} with October start should return 2000"
        );
    }
    for month in 10..=12_u8 {
        assert_eq!(
            water_year(2000, month, 10).unwrap(),
            2001,
            "month {month} with October start should return 2001"
        );
    }
}

#[test]
fn all_start_months_fixed_date() {
    // For month=6 (June), year=2000, test every start_month.
    let year = 2000;
    let month = 6_u8;

    // start_month=1 is the special case: always returns year.
    assert_eq!(water_year(year, month, 1).unwrap(), 2000);

    // start_month 2..=6: month (6) >= start_month, so water year = year + 1.
    for sm in 2..=6_u8 {
        assert_eq!(
            water_year(year, month, sm).unwrap(),
            2001,
            "start_month={sm}: month 6 >= {sm}, should return 2001"
        );
    }

    // start_month 7..=12: month (6) < start_month, so water year = year.
    for sm in 7..=12_u8 {
        assert_eq!(
            water_year(year, month, sm).unwrap(),
            2000,
            "start_month={sm}: month 6 < {sm}, should return 2000"
        );
    }
}

#[test]
fn negative_years() {
    // year=-100, month=10, start_month=10 -> month >= start -> year + 1 = -99
    assert_eq!(water_year(-100, 10, 10).unwrap(), -99);

    // year=-1, month=1, start_month=10 -> month < start -> year = -1
    assert_eq!(water_year(-1, 1, 10).unwrap(), -1);

    // year=0, month=12, start_month=10 -> month >= start -> year + 1 = 1
    assert_eq!(water_year(0, 12, 10).unwrap(), 1);
}
