//! Integration tests for the k_lall_sharma heuristic.

use zeus_knn::k_lall_sharma;

#[test]
fn known_values() {
    assert_eq!(k_lall_sharma(1), 1);
    assert_eq!(k_lall_sharma(4), 2);
    assert_eq!(k_lall_sharma(9), 3);
    assert_eq!(k_lall_sharma(16), 4);
    assert_eq!(k_lall_sharma(25), 5);
    assert_eq!(k_lall_sharma(100), 10);
}

#[test]
fn zero_candidates() {
    // Edge case: 0 candidates -> max(floor(0), 1) = 1
    assert_eq!(k_lall_sharma(0), 1);
}

#[test]
fn large_values() {
    assert_eq!(k_lall_sharma(10000), 100);
    assert_eq!(k_lall_sharma(1_000_000), 1000);
}

#[test]
fn non_perfect_squares() {
    // floor(sqrt(50)) = 7
    assert_eq!(k_lall_sharma(50), 7);
    // floor(sqrt(2)) = 1
    assert_eq!(k_lall_sharma(2), 1);
    // floor(sqrt(3)) = 1
    assert_eq!(k_lall_sharma(3), 1);
    // floor(sqrt(5)) = 2
    assert_eq!(k_lall_sharma(5), 2);
    // floor(sqrt(99)) = 9
    assert_eq!(k_lall_sharma(99), 9);
}
