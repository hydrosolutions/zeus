//! Daily resampling loop with multi-level fallback cascade.

use crate::config::ResampleConfig;
use crate::error::ResampleError;
use crate::obs_data::ObsData;
use crate::result::ResampleResult;
use crate::year_subset::YearSubset;
use tracing::{debug, trace_span};
use zeus_calendar::expand_indices;
use zeus_knn::{KnnConfig, KnnScratch, k_lall_sharma, knn_sample_with_scratch};
use zeus_markov::{PrecipState, simulate_states};

/// Resamples 365 days given a pre-built year subset.
///
/// Implements the full daily loop with 6-level fallback cascade:
/// 1. Carry-forward: no month candidates -> copy previous day
/// 2. Empty window: expand_indices returns empty -> uniform sample day0
/// 3. Narrow +/-3: transition match -> KNN -> assign day1 successor
/// 4. Wide +/-30: transition match -> KNN -> assign day1 successor
/// 5. Relaxed: any same-month day, no transition -> uniform sample day0
/// 6. Calendar-year safeguard: pre/post-KNN water-year filtering
#[tracing::instrument(skip(subset, obs, sim_months, sim_days, config, rng), fields(n_days = sim_months.len()))]
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn resample_days(
    subset: &YearSubset,
    obs: &ObsData,
    sim_months: &[u8],
    sim_days: &[u8],
    initial_state: PrecipState,
    prev_obs_idx: Option<usize>,
    config: &ResampleConfig,
    rng: &mut impl rand::Rng,
) -> Result<ResampleResult, ResampleError> {
    use rand::seq::IndexedRandom;

    let n_days = sim_months.len(); // should be 365
    let mut output = vec![0usize; n_days];
    let calendar_year_mode = config.year_start_month() == 1;
    let subset_len = subset.len();

    // Build offset arrays for narrow and wide window.
    let narrow_w = config.narrow_window() as i32;
    let narrow_offsets: Vec<i32> = (-narrow_w..=narrow_w).collect();
    let wide_w = config.wide_window() as i32;
    let wide_offsets: Vec<i32> = (-wide_w..=wide_w).collect();

    // n_max for expand_indices: subset_len - 1 ensures valid day1 successors.
    let n_max = if subset_len > 1 { subset_len - 1 } else { 1 };

    // === Day 1: Uniform random selection ===
    let mut day1_candidates: Vec<usize> = subset
        .month_day_candidates(sim_months[0], sim_days[0])
        .to_vec();
    if day1_candidates.is_empty() {
        day1_candidates = subset.month_candidates(sim_months[0]);
    }
    if day1_candidates.is_empty() {
        day1_candidates = (0..subset_len).collect();
    }

    // Calendar-year safeguard for day 1 (year > 1 only).
    if calendar_year_mode && let Some(prev_idx) = prev_obs_idx {
        let prev_wy = obs.water_years()[prev_idx];
        let filtered: Vec<usize> = day1_candidates
            .iter()
            .copied()
            .filter(|&c| subset.water_year_at(c) == prev_wy)
            .collect();
        if !filtered.is_empty() {
            day1_candidates = filtered;
        }
    }

    let &day1_local = day1_candidates
        .choose(rng)
        .ok_or(ResampleError::NoCandidates {
            day: 0,
            month: sim_months[0],
            year: None,
        })?;

    let day1_global = subset.to_global(day1_local);
    output[0] = day1_global;

    let mut prev_precip = obs.precip()[day1_global];
    let mut prev_temp = obs.temp()[day1_global];

    // Classify day 1 state from observed value.
    let day1_state = subset.classify(prev_precip, sim_months[0]);

    // === Generate Markov states for days 2-365 ===
    let markov_states_rest = if n_days > 1 {
        simulate_states(subset.transitions(), &sim_months[1..], day1_state, rng)
    } else {
        Vec::new()
    };

    // Full state array: day1_state + markov_states_rest
    let mut states = Vec::with_capacity(n_days);
    states.push(day1_state);
    states.extend_from_slice(&markov_states_rest);

    // === Allocate KNN scratch ===
    let mut scratch = KnnScratch::new(subset_len);

    // === Days 2..365 ===
    for d in 1..n_days {
        let _day = trace_span!("day", d, month = sim_months[d]).entered();
        let month = sim_months[d];
        let day = sim_days[d];

        // Get base candidates for this month-day.
        let base_md = subset.month_day_candidates(month, day);
        let base_candidates: Vec<usize> = if base_md.is_empty() {
            let mc = subset.month_candidates(month);
            if mc.is_empty() {
                // L1 Carry-forward fallback: no month candidates at all.
                debug!(day = d, level = 1, "carry-forward: no candidates for month");
                output[d] = output[d - 1];
                prev_precip = obs.precip()[output[d]];
                prev_temp = obs.temp()[output[d]];
                continue;
            }
            mc
        } else {
            base_md.to_vec()
        };

        // Expand +/-narrow window.
        let expanded_narrow = expand_indices(&base_candidates, &narrow_offsets, n_max);

        if expanded_narrow.is_empty() {
            // L2 Empty window fallback: uniform sample day0, assign day0 values.
            debug!(
                day = d,
                level = 2,
                "empty window: uniform from base candidates"
            );
            let &chosen = base_candidates
                .choose(rng)
                .expect("base_candidates non-empty");
            let global_idx = subset.to_global(chosen);
            output[d] = global_idx;
            prev_precip = obs.precip()[global_idx];
            prev_temp = obs.temp()[global_idx];
            continue;
        }

        // L3 Narrow window with transition match.
        let state_from = states[d - 1];
        let state_to = states[d];
        let mut matches = match_transitions(&expanded_narrow, state_from, state_to, subset);

        if matches.is_empty() {
            // L4 Wide window with transition match.
            debug!(day = d, level = 4, "narrow window miss: trying wide window");
            let expanded_wide = expand_indices(&base_candidates, &wide_offsets, n_max);
            matches = match_transitions(&expanded_wide, state_from, state_to, subset);
        }

        if matches.is_empty() {
            // L5 Relaxed fallback: any same-month day with valid successor.
            debug!(day = d, level = 5, "wide window miss: relaxed fallback");
            let mut fb: Vec<usize> = (0..subset_len.saturating_sub(1))
                .filter(|&i| subset.month_at(i) == month)
                .collect();

            // In non-calendar-year mode, also require same water year for successor.
            if !calendar_year_mode {
                fb.retain(|&i| subset.water_year_at(i) == subset.water_year_at(i + 1));
            }

            if fb.is_empty() {
                // Fallback to ALL subset indices with valid successor.
                fb = (0..subset_len.saturating_sub(1)).collect();
                if !calendar_year_mode {
                    fb.retain(|&i| subset.water_year_at(i) == subset.water_year_at(i + 1));
                }
            }

            if fb.is_empty() {
                // Ultimate fallback.
                fb = (0..subset_len).collect();
            }

            // Assign day0 values (NOT day1).
            let &chosen = fb.choose(rng).expect("fb non-empty");
            let global_idx = subset.to_global(chosen);
            output[d] = global_idx;
            prev_precip = obs.precip()[global_idx];
            prev_temp = obs.temp()[global_idx];
            continue;
        }

        // Pre-KNN calendar-year filter.
        let mut day0_indices = matches;
        if calendar_year_mode {
            let filtered: Vec<usize> = day0_indices
                .iter()
                .copied()
                .filter(|&i| subset.water_year_at(i) == subset.water_year_at(i + 1))
                .collect();
            if !filtered.is_empty() {
                day0_indices = filtered;
            }
            // If empty, keep original (drops calendar-year constraint).
        }

        // Build 2-D candidate matrix for KNN: (precip_anom, temp_anom)
        let mean_p = subset.mean_precip(month);
        let mean_t = subset.mean_temp(month);
        let n_matches = day0_indices.len();

        let mut candidates_flat = Vec::with_capacity(n_matches * 2);
        for &i in &day0_indices {
            candidates_flat.push(subset.precip_at(i) - mean_p);
            candidates_flat.push(subset.temp_at(i) - mean_t);
        }

        let target = [prev_precip - mean_p, prev_temp - mean_t];
        let weights = subset.knn_weights(month);

        let k_daily = k_lall_sharma(n_matches).max(1);
        let knn_config = KnnConfig::new(k_daily)
            .with_sampling(config.sampling().clone())
            .with_epsilon(config.epsilon());

        let knn_result = knn_sample_with_scratch(
            &candidates_flat,
            2,
            &target,
            weights,
            &knn_config,
            rng,
            &mut scratch,
        )?;

        let selected_local_idx = knn_result.indices()[0]; // index into day0_indices
        let day0_subset_idx = day0_indices[selected_local_idx];

        // Post-KNN calendar-year filter.
        let mut final_day1_idx = day0_subset_idx + 1;
        if calendar_year_mode && d > 0 {
            let prev_obs_wy = obs.water_years()[output[d - 1]];
            let day1_global = subset.to_global(final_day1_idx);
            let day1_wy = obs.water_years()[day1_global];
            if day1_wy != prev_obs_wy {
                // Try to find a valid day1 in same water year from the matched set.
                let valid: Vec<usize> = day0_indices
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let g = subset.to_global(i + 1);
                        obs.water_years()[g] == prev_obs_wy
                    })
                    .collect();
                if !valid.is_empty() {
                    let &fallback = valid.choose(rng).expect("valid non-empty");
                    final_day1_idx = fallback + 1;
                }
                // If valid is empty, keep original (drops calendar-year constraint).
            }
        }

        // Assign day1 successor values (normal KNN path).
        let global_idx = subset.to_global(final_day1_idx);
        output[d] = global_idx;
        prev_precip = obs.precip()[global_idx];
        prev_temp = obs.temp()[global_idx];
    }

    let final_state = *states.last().unwrap_or(&initial_state);
    let last_obs_idx = *output.last().unwrap_or(&0);

    Ok(ResampleResult::new(output, final_state, last_obs_idx))
}

/// Filters expanded indices to those matching a specific state transition.
///
/// Returns subset indices where:
/// - `classify(precip[pos], month_at(pos)) == state_from`
/// - `classify(precip[pos+1], month_at(pos+1)) == state_to`
#[allow(dead_code)]
fn match_transitions(
    expanded: &[usize],
    state_from: PrecipState,
    state_to: PrecipState,
    subset: &YearSubset,
) -> Vec<usize> {
    expanded
        .iter()
        .copied()
        .filter(|&pos| {
            let from_month = subset.month_at(pos);
            let to_month = subset.month_at(pos + 1);
            subset.classify(subset.precip_at(pos), from_month) == state_from
                && subset.classify(subset.precip_at(pos + 1), to_month) == state_to
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs_data::ObsData;
    use crate::year_subset::YearSubset;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use zeus_markov::{MarkovConfig, ThresholdSpec};

    /// Helper: create a multi-year observation dataset.
    fn make_obs(n_years: usize) -> ObsData {
        let days_per_year = 365;
        let n = n_years * days_per_year;
        let mut precip = Vec::with_capacity(n);
        let mut temp = Vec::with_capacity(n);
        let mut months = Vec::with_capacity(n);
        let mut days = Vec::with_capacity(n);
        let mut water_years = Vec::with_capacity(n);

        for y in 0..n_years {
            let wy = 2000 + y as i32;
            let mut doy = 0;
            for m in 1..=12u8 {
                let dim: u8 = match m {
                    2 => 28,
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                for d in 1..=dim {
                    if doy >= days_per_year {
                        break;
                    }
                    let p = if (doy * 7 + y * 13) % 10 < 4 {
                        0.0
                    } else {
                        ((doy as f64) * 0.1 + (y as f64) * 3.0).max(0.1)
                    };
                    precip.push(p);
                    temp.push(15.0 + (m as f64) * 2.0 + (y as f64) * 0.5);
                    months.push(m);
                    days.push(d);
                    water_years.push(wy);
                    doy += 1;
                }
            }
        }

        ObsData::new(&precip, &temp, &months, &days, &water_years).unwrap()
    }

    fn make_sim_calendar(start_month: u8) -> (Vec<u8>, Vec<u8>) {
        let start = zeus_calendar::NoLeapDate::new(2000, start_month, 1).unwrap();
        let dates = zeus_calendar::noleap_sequence(start, 365);
        let sim_months: Vec<u8> = dates.iter().map(|d| d.month()).collect();
        let sim_days: Vec<u8> = dates.iter().map(|d| d.day()).collect();
        (sim_months, sim_days)
    }

    #[test]
    fn basic_365_indices() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2, 3, 4], &obs, &config, &markov_config).unwrap();
        let (sim_months, sim_days) = make_sim_calendar(1);
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_days(
            &subset,
            &obs,
            &sim_months,
            &sim_days,
            PrecipState::Dry,
            None,
            &config,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.indices().len(), 365);
        // All indices must be valid.
        for &idx in result.indices() {
            assert!(idx < obs.len(), "index {idx} out of bounds");
        }
    }

    #[test]
    fn reproducible() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2, 3, 4], &obs, &config, &markov_config).unwrap();
        let (sm, sd) = make_sim_calendar(1);

        let mut rng1 = StdRng::seed_from_u64(99);
        let r1 = resample_days(
            &subset,
            &obs,
            &sm,
            &sd,
            PrecipState::Dry,
            None,
            &config,
            &mut rng1,
        )
        .unwrap();

        let mut rng2 = StdRng::seed_from_u64(99);
        let r2 = resample_days(
            &subset,
            &obs,
            &sm,
            &sd,
            PrecipState::Dry,
            None,
            &config,
            &mut rng2,
        )
        .unwrap();

        assert_eq!(r1.indices(), r2.indices());
    }

    #[test]
    fn output_data_finite() {
        let obs = make_obs(5);
        let config = ResampleConfig::new();
        let markov_config = MarkovConfig::new()
            .with_wet_spec(ThresholdSpec::Fixed(0.3))
            .with_extreme_spec(ThresholdSpec::Quantile(0.8));

        let subset = YearSubset::build(&[0, 1, 2, 3, 4], &obs, &config, &markov_config).unwrap();
        let (sm, sd) = make_sim_calendar(1);
        let mut rng = StdRng::seed_from_u64(42);

        let result = resample_days(
            &subset,
            &obs,
            &sm,
            &sd,
            PrecipState::Dry,
            None,
            &config,
            &mut rng,
        )
        .unwrap();

        for &idx in result.indices() {
            assert!(obs.precip()[idx].is_finite());
            assert!(obs.temp()[idx].is_finite());
            assert!(obs.precip()[idx] >= 0.0);
        }
    }
}
