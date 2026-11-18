---
title: View decoding performance across-time metrics in pyPhoPlaceCellAnalysis/pyPhoCoreHelpers
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/2a3822c4-fcac-4233-b121-585394d17c6b
url: https://phohale.sourcegraph.app/deepsearch/378
created: 2026-06-19T17:02:15Z
updated: 2026-06-19T17:06:26Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/CommanderPho/pyPhoCoreHelpers What decoding performance across-time metrics do I have?

Now I have a comprehensive picture. Let me compile the full answer.

Here is a summary of all the decoding performance "across-time" metrics you have implemented in [pyPhoPlaceCellAnalysis](r/github.com/CommanderPho/pyPhoPlaceCellAnalysis):

---

## Epoch-level Score Metrics (per-decoder, per-epoch)

These are computed by [`_compute_all_df_score_metrics`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L6662-6792) for both **laps** and **ripple** epochs, across all four directional decoders (`long_LR`, `long_RL`, `short_LR`, `short_RL`):

### 1. Radon Transform (`score`, `velocity`, `intercept`, `speed`)
Fits a line to the decoded posterior over time and measures how well a sequential sweep fits. Columns: `score`, `velocity`, `intercept`, `speed`.

### 2. Weighted Correlation (`wcorr`)
Computed via [`compute_weighted_correlations`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L6344-6346). A weighted Pearson correlation between decoded position and time bin centers. Per-decoder merged columns: `wcorr_long_LR`, `wcorr_long_RL`, `wcorr_short_LR`, `wcorr_short_RL`.

### 3. Simple PF Peak PearsonR (`pearsonr`)
Spike time vs. place field peak position Pearson correlation, computed via `compute_simple_spike_time_v_pf_peak_x_by_epoch`. Column: `pearsonr` per decoder.

### 4. Bayesian Decoder Probability (`P_decoder`)
The marginal probability assigned to each decoder from the Pseudo2D posterior: `P_LR`, `P_RL`, `P_Long`, `P_Short`.

---

## Heuristic Replay Scores (bin-wise, epoch-level)

Defined in [`HeuristicReplayScoring`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py) and assembled by [`build_all_score_computations_fn_dict`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2909-2940) + [`build_all_bin_wise_subseq_partitioning_computation_fn_dict`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2502-2512):

**From `build_all_bin_wise_computation_fn_dict`** (currently active):
| Column | Description |
|--------|-------------|
| `avg_jump_cm` | Average bin-to-bin position jump in cm |
| `travel` | Total position difference (distance traveled) across the epoch |
| `coverage` | Fraction of the track covered by the decoded positions |

**From `SequenceScoringComputations` (position-based)**:
| Column | Description |
|--------|-------------|
| `total_distance_traveled` | Total distance traversed |
| `track_coverage_score` | Track coverage score |

**From `SubsequencesPartitioningResultScoringComputations`** (bin-wise subsequence analysis):
| Column | Description |
|--------|-------------|
| `mseq_len` | Length of the longest monotonic subsequence |
| `mseq_len_ignoring_intrusions` | Same, ignoring intrusion bins |
| `mseq_len_ignoring_intrusions_and_repeats` | Same, also ignoring repeated positions |
| `mseq_len_ratio_ignoring_intrusions_and_repeats` | Ratio version of the above |
| `mseq_tcov` | Track coverage of the main subsequence |
| `mseq_dtrav` | Total distance traveled by the main subsequence |

**Temporal functions (disabled by default, `enable_temporal_functions=True`)**:
- `sequential_correlation`, `monotonicity_score`, `laplacian_smoothness`

---

## Older / Legacy Per-epoch Heuristics

The [`HeuristicScoresTuple`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L39-50) dataclass records the original set per epoch:
- `longest_sequence_length`, `longest_sequence_length_ratio`
- `direction_change_bin_ratio`, `congruent_dir_bins_ratio`, `total_congruent_direction_change`
- `total_variation`, `integral_second_derivative`, `stddev_of_diff`

---

## Aggregated / Agreement Statistics

[`DecoderDecodedEpochsResult.compute_matching_best_indicies`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L6711-6781) measures the **agreement ratio** between the most-likely decoder (from Bayesian marginals) and the best decoder (from each score metric) — printed for Radon, WCorr, and PearsonR across laps and ripples.

---

All of the epoch score columns are consolidated into a single merged dataframe via [`build_complete_all_scores_merged_df`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3576-3573), with the full column taxonomy listed in [`get_all_scores_column_names`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3545-3574).

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ decoding performance across time metrics](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20decoding%20performance%20across%20time%20metrics)
- [repo:^github.com/CommanderPho/pyPhoCoreHelpers$ decoding performance across time metrics](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoCoreHelpers$%20decoding%20performance%20across%20time%20metrics)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ decoding score metric time wcorr pearson epoch](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20decoding%20score%20metric%20time%20wcorr%20pearson%20epoch)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ _compute_all_df_score_metrics](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20_compute_all_df_score_metrics)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L6556-6760](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L6556-6760)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L6760-6840](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L6760-6840)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "across_time" decoding performance metric](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22across_time%22%20decoding%20performance%20metric)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ continuous decoding locality measure entropy time bin](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20continuous%20decoding%20locality%20measure%20entropy%20time%20bin)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ heuristic_replay_scoring score_metric replay](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20heuristic_replay_scoring%20score_metric%20replay)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L1-120](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1-120)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ get_all_scores_column_names HeuristicReplayScoring](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20get_all_scores_column_names%20HeuristicReplayScoring)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:heuristic_replay_scoring compute_pho_heuristic_replay_scores](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:heuristic_replay_scoring%20compute_pho_heuristic_replay_scores)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:heuristic_replay_scoring get_all_score_computation_col_names](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:heuristic_replay_scoring%20get_all_score_computation_col_names)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3540-3600](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3540-3600)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:heuristic_replay_scoring build_all_score_computations_fn_dict build_all_bin_wise_subseq_partitioning_computation_fn_dict](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:heuristic_replay_scoring%20build_all_score_computations_fn_dict%20build_all_bin_wise_subseq_partitioning_computation_fn_dict)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2502-2545](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2502-2545)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2909-2946](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2909-2946)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:heuristic_replay_scoring build_all_bin_wise_computation_fn_dict](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:heuristic_replay_scoring%20build_all_bin_wise_computation_fn_dict)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2885-2912](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2885-2912)


---

# find code involving the distance between most-likely decoded position and observed position

Here is a map of all the code involving the distance between most-likely decoded position and observed/measured position:

---

## 1. `MeasuredDecodedPositionComparison` + `CustomDecodeEpochsResult` — MSE / RMSE per epoch

**File:** [`DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4057-4181)

The primary structured comparison. For each decoded epoch:
- Measures are interpolated from the global position df to the decoded time bin centers via `TrainTestLapsSplitting.interpolate_positions`
- Computes per-epoch **MSE** (`sq_err`) and **RMSE** in cm (`err_cm`) using `sklearn.metrics.mean_squared_error`
- Stores results in a `decoded_measured_diff_df` with columns `['t', 'sq_err', 'err_cm']`

```python
test_decoded_measured_diff: float = mean_squared_error(a_valid_interpolated_measured_x, a_valid_decoded_positions)
test_decoded_measured_diff_cm: float = np.sqrt(test_decoded_measured_diff)
```

The `MeasuredDecodedPositionComparison` dataclass holds:
- `measured_positions_dfs_list` — interpolated measured positions, one df per epoch
- `decoded_positions_df_list` — most-likely decoded positions, one df per epoch
- `decoded_measured_diff_df` — the per-epoch MSE/RMSE summary

Entry points:
- [`build_single_measured_decoded_position_comparison`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4108-4181) — single decoder
- [`build_measured_decoded_position_comparison`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4183-4208) — dict of decoders

---

## 2. Leave-One-Out (LOO) Decoding — Squared Euclidean Distance per epoch

**File:** [`decoder_result.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py?L762-L823)

Used to assess each neuron's contribution to decoding quality. For each held-out neuron's decoder, per epoch:
- Measured linearized position (`lin_pos`) is interpolated to the decoded time bin centers via `np.interp`
- **Squared Euclidean distance** computed via `scipy.spatial.distance.cdist(..., 'sqeuclidean')`, but only over non-silent (non-zero spike) time bins

```python
curr_omit_aclu_distance = distance.cdist(
    np.atleast_2d(window_center_measured_pos_x[~curr_is_time_bin_non_firing]),
    np.atleast_2d(curr_most_likely_positions[~curr_is_time_bin_non_firing]),
    'sqeuclidean'
)
```

Results are aggregated into `one_left_out_omitted_aclu_distance_df` with `avg_dist` per omitted neuron, sorted descending (largest = most important neuron). Used to rank cell importance.

---

## 3. `display_predicted_position_difference` — Visual Arrow Display

**File:** [`decoder_difference.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/decoder_difference.py?L11-L150)

Not a numeric metric — a matplotlib 2D animated visualization. Draws a `FancyArrowPatch` from the **predicted** (`most_likely_positions`) to the **measured** position at each time window. Useful for qualitative inspection of per-frame prediction error. Marked `NOT YET USED`.

---

## 4. `DiagnosticDistanceMetricFigure` — Interactive Per-Timebin Diagnostic

**File:** [`decoder_result.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py?L1160-L1280)

A PyQtGraph interactive figure for the LOO analysis. Shows per-timebin comparisons (placefield of active cell, a random non-firing cell's placefield, and the decoded posterior `p_x_given_n`) synchronized via a slider. Works on top of the LOO result objects (`LeaveOneOutDecodingAnalysisResult`).

---

## 5. `plot_1D_most_likely_position_comparsions` — Overlay Plot

**File:** [`DecoderPredictionError.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py?L439-L542)

The main display function that overlays the decoded most-likely position curve on top of the measured position trajectory and the posterior heatmap. Not a metric per se, but the canonical visual for evaluating decoded vs. observed position.

---

## Summary

| Location | Distance Measure | Applied To | Epoch-level or Bin-level |
|---|---|---|---|
| `build_single_measured_decoded_position_comparison` | MSE / RMSE (cm) | Laps, per decoder | Epoch-level summary |
| LOO analysis in `decoder_result.py` | Squared Euclidean | Non-silent time bins | Accumulated per epoch, averaged per cell |
| `decoder_difference.py` | Visual arrow (no scalar) | 2D position per frame | Bin-level (visualization) |
| `DecoderPredictionError.py` | Visual overlay (no scalar) | 1D position over time | Bin-level (visualization) |

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ most likely decoded position observed measured distance error](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20most%20likely%20decoded%20position%20observed%20measured%20distance%20error)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ most_likely_position measured_position distance](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20most_likely_position%20measured_position%20distance)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L4057-4280](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4057-4280)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py@26f673d?L755-830](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py?L755-830)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "err_cm" OR "sq_err" decoded measured diff position](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22err_cm%22%20OR%20%22sq_err%22%20decoded%20measured%20diff%20position)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:DecoderPredictionError decoded position error distance](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:DecoderPredictionError%20decoded%20position%20error%20distance)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ mean_squared_error decoded position](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20mean_squared_error%20decoded%20position)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ sqeuclidean cdist decoded measured position distance](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20sqeuclidean%20cdist%20decoded%20measured%20position%20distance)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ DiagnosticDistanceMetricFigure decoder distance metric figure](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20DiagnosticDistanceMetricFigure%20decoder%20distance%20metric%20figure)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py@26f673d?L1155-1280](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py?L1155-1280)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/decoder_difference.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/decoder_difference.py)

