use zeus_calendar::expand_indices;

#[test]
fn plus_minus_3_window() {
    let result = expand_indices(&[100, 200], &[-3, -2, -1, 0, 1, 2, 3], 365);
    assert_eq!(
        result,
        vec![
            97, 98, 99, 100, 101, 102, 103, 197, 198, 199, 200, 201, 202, 203
        ]
    );
}

#[test]
fn plus_minus_30_boundary_clipping() {
    let offsets: Vec<i32> = (-30..=30).collect();
    let result = expand_indices(&[10], &offsets, 365);

    // 10 + (-30) = -20 (clipped) ... 10 + (-11) = -1 (clipped)
    // 10 + (-10) = 0 ... 10 + 30 = 40
    // So result should be 0..=40
    assert_eq!(
        *result.first().unwrap(),
        0,
        "first element should be 0 (clipped)"
    );
    assert_eq!(*result.last().unwrap(), 40, "last element should be 40");
    assert_eq!(result.len(), 41, "should have 41 elements (0 through 40)");
}

#[test]
fn successor_constraint_scenario() {
    // Expand base=[50] with offsets=[-3..=3], n_max=365
    let offsets: Vec<i32> = (-3..=3).collect();
    let result = expand_indices(&[50], &offsets, 365);

    // Should produce [47, 48, 49, 50, 51, 52, 53]
    let expected: Vec<usize> = (47..=53).collect();
    assert_eq!(result, expected);

    // Demonstrate the successor constraint pattern: filter out index 51
    let mut filtered = result;
    filtered.retain(|&i| i != 51);
    assert_eq!(filtered, vec![47, 48, 49, 50, 52, 53]);
}

#[test]
fn large_base_many_offsets() {
    let base: Vec<usize> = (0..100).collect();
    let offsets: Vec<i32> = (-5..=5).collect();
    let result = expand_indices(&base, &offsets, 365);

    // Verify sorted
    let mut sorted = result.clone();
    sorted.sort_unstable();
    assert_eq!(result, sorted, "result must be sorted");

    // Verify deduplicated
    let mut deduped = result.clone();
    deduped.dedup();
    assert_eq!(result, deduped, "result must be deduplicated");

    // First element: min(base) + min(offset) = 0 + (-5) = -5, clipped to 0
    assert_eq!(*result.first().unwrap(), 0);

    // Last element: max(base) + max(offset) = 99 + 5 = 104
    assert_eq!(*result.last().unwrap(), 104);

    // Length: continuous range 0..=104 = 105 elements
    assert_eq!(result.len(), 105);
}

#[test]
fn empty_edge_cases() {
    // Empty base
    assert_eq!(expand_indices(&[], &[0], 365), vec![]);

    // Empty offsets
    assert_eq!(expand_indices(&[0], &[], 365), vec![]);

    // n_max = 0 means no valid indices exist
    assert_eq!(expand_indices(&[0], &[0], 0), vec![]);
}
