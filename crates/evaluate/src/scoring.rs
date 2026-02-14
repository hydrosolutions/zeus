//! MAE-based scoring and ranking of realisations.

use crate::output::RealisationScore;
use crate::timeseries::TimeseriesStats;

/// Flatten timeseries stats into a vector of comparable values.
///
/// Produces: [mean, sd, skewness_or_0, wet_days, dry_days, mean_wet_spell, mean_dry_spell]
pub fn flatten_stats(stats: &TimeseriesStats) -> Vec<f64> {
    vec![
        stats.mean,
        stats.sd,
        stats.skewness.unwrap_or(0.0),
        stats.wet_days as f64,
        stats.dry_days as f64,
        stats.mean_wet_spell,
        stats.mean_dry_spell,
    ]
}

/// Compute per-realisation MAE scores from observed and simulated timeseries stats.
///
/// `obs_values` and `sim_values[r]` are parallel vectors of stat values
/// (one f64 per (site, month, variable, stat_name) cell).
/// `sim_values` has one entry per realisation.
pub fn score_realisations(
    obs_values: &[f64],
    sim_values: &[Vec<f64>],
    realisation_ids: &[u32],
) -> Vec<RealisationScore> {
    assert_eq!(sim_values.len(), realisation_ids.len());

    // Compute raw MAE for each realisation
    let raw_maes: Vec<(u32, f64)> = realisation_ids
        .iter()
        .zip(sim_values.iter())
        .map(|(&id, sim)| {
            assert_eq!(obs_values.len(), sim.len());
            let mae: f64 = obs_values
                .iter()
                .zip(sim.iter())
                .map(|(o, s)| (o - s).abs())
                .sum::<f64>()
                / obs_values.len() as f64;
            (id, mae)
        })
        .collect();

    // Find min and max for normalization
    let min_mae = raw_maes
        .iter()
        .map(|(_, mae)| mae)
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_mae = raw_maes
        .iter()
        .map(|(_, mae)| mae)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // Normalize
    let range = max_mae - min_mae;
    let normalized: Vec<(u32, f64, f64)> = raw_maes
        .iter()
        .map(|&(id, raw)| {
            let norm = if range.abs() < 1e-10 {
                0.0
            } else {
                (raw - min_mae) / range
            };
            (id, raw, norm)
        })
        .collect();

    // Sort by raw MAE ascending for ranking
    let mut sorted = normalized;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Assign ranks (handle ties)
    let mut scores: Vec<RealisationScore> = Vec::with_capacity(sorted.len());
    let mut current_rank = 1;
    for i in 0..sorted.len() {
        // Check if this is a tie with the previous entry
        if i > 0 && (sorted[i].1 - sorted[i - 1].1).abs() < 1e-10 {
            // Same MAE, same rank
            let prev_rank = scores[i - 1].rank;
            scores.push(RealisationScore {
                realisation: sorted[i].0,
                raw_mae: sorted[i].1,
                normalized_score: sorted[i].2,
                rank: prev_rank,
            });
        } else {
            scores.push(RealisationScore {
                realisation: sorted[i].0,
                raw_mae: sorted[i].1,
                normalized_score: sorted[i].2,
                rank: current_rank,
            });
        }
        current_rank += 1;
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_stats() {
        let stats = TimeseriesStats {
            mean: 10.0,
            sd: 2.0,
            skewness: Some(0.5),
            wet_days: 15,
            dry_days: 5,
            mean_wet_spell: 3.0,
            mean_dry_spell: 1.0,
        };

        let flat = flatten_stats(&stats);
        assert_eq!(flat.len(), 7);
        assert_eq!(flat[0], 10.0); // mean
        assert_eq!(flat[1], 2.0); // sd
        assert_eq!(flat[2], 0.5); // skewness
        assert_eq!(flat[3], 15.0); // wet_days
        assert_eq!(flat[4], 5.0); // dry_days
        assert_eq!(flat[5], 3.0); // mean_wet_spell
        assert_eq!(flat[6], 1.0); // mean_dry_spell
    }

    #[test]
    fn test_flatten_stats_none_skewness() {
        let stats = TimeseriesStats {
            mean: 5.0,
            sd: 1.0,
            skewness: None,
            wet_days: 10,
            dry_days: 10,
            mean_wet_spell: 2.0,
            mean_dry_spell: 2.0,
        };

        let flat = flatten_stats(&stats);
        assert_eq!(flat[2], 0.0); // None skewness becomes 0.0
    }

    #[test]
    fn test_score_realisations_basic() {
        // 3 realisations with known MAEs
        let obs = vec![1.0, 2.0, 3.0];
        let sim = vec![
            vec![1.5, 2.0, 3.0], // MAE = (0.5 + 0.0 + 0.0) / 3 = 0.1667
            vec![1.0, 2.0, 3.0], // MAE = (0.0 + 0.0 + 0.0) / 3 = 0.0
            vec![2.0, 3.0, 4.0], // MAE = (1.0 + 1.0 + 1.0) / 3 = 1.0
        ];
        let ids = vec![10, 20, 30];

        let scores = score_realisations(&obs, &sim, &ids);

        assert_eq!(scores.len(), 3);

        // Best (lowest MAE) should be rank 1
        let rank1 = scores.iter().find(|s| s.rank == 1).unwrap();
        assert_eq!(rank1.realisation, 20);
        assert!((rank1.raw_mae - 0.0).abs() < 1e-6);

        // Worst (highest MAE) should be rank 3
        let rank3 = scores.iter().find(|s| s.rank == 3).unwrap();
        assert_eq!(rank3.realisation, 30);
        assert!((rank3.raw_mae - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_realisations_normalization() {
        let obs = vec![0.0, 0.0];
        let sim = vec![
            vec![1.0, 1.0], // MAE = 1.0
            vec![3.0, 3.0], // MAE = 3.0
        ];
        let ids = vec![0, 1];

        let scores = score_realisations(&obs, &sim, &ids);

        // Min MAE should get normalized score 0.0
        let best = scores.iter().find(|s| s.rank == 1).unwrap();
        assert!((best.normalized_score - 0.0).abs() < 1e-6);

        // Max MAE should get normalized score 1.0
        let worst = scores.iter().find(|s| s.rank == 2).unwrap();
        assert!((worst.normalized_score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_realisations_tied() {
        let obs = vec![1.0, 2.0];
        let sim = vec![
            vec![1.5, 2.5], // MAE = 0.5
            vec![1.5, 2.5], // MAE = 0.5 (tied)
            vec![2.0, 3.0], // MAE = 1.0
        ];
        let ids = vec![0, 1, 2];

        let scores = score_realisations(&obs, &sim, &ids);

        // Both tied realisations should have rank 1
        let tied_count = scores.iter().filter(|s| s.rank == 1).count();
        assert_eq!(tied_count, 2);

        // The worse one should have rank 3
        let worse = scores.iter().find(|s| s.rank == 3).unwrap();
        assert_eq!(worse.realisation, 2);
    }

    #[test]
    fn test_score_realisations_single() {
        let obs = vec![1.0, 2.0, 3.0];
        let sim = vec![vec![1.5, 2.5, 3.5]];
        let ids = vec![42];

        let scores = score_realisations(&obs, &sim, &ids);

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].realisation, 42);
        assert_eq!(scores[0].rank, 1);
        assert!((scores[0].normalized_score - 0.0).abs() < 1e-6); // Single realisation gets 0.0
    }

    #[test]
    fn test_score_realisations_identical() {
        let obs = vec![1.0, 2.0];
        let sim = vec![
            vec![2.0, 3.0], // MAE = 1.0
            vec![2.0, 3.0], // MAE = 1.0
            vec![2.0, 3.0], // MAE = 1.0
        ];
        let ids = vec![0, 1, 2];

        let scores = score_realisations(&obs, &sim, &ids);

        // All should have rank 1 and normalized score 0.0 (no variation)
        for score in &scores {
            assert_eq!(score.rank, 1);
            assert!((score.normalized_score - 0.0).abs() < 1e-6);
        }
    }
}
