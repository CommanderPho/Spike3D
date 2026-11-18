---
name: Filtered copy by min seq length
overview: Add a new method on `MatchingPastFuturePositionsResult` that returns a copy of `self` filtered by `minimum_included_matching_sequence_length`, without mutating `self`. Filtering reuses the logic from `get_filtered_by_min_seq_length` and `_prepare_epoch_data`; the copy is built via `attrs.evolve` with updated and recomputed fields.
todos: []
isProject: false
---

# Add `filtered_copy_by_min_sequence_length` to MatchingPastFuturePositionsResult

## Goal

Implement a new method that returns a **new instance** of `MatchingPastFuturePositionsResult` containing only position sequences whose merged segment has `num_epoch_t_bins >= minimum_included_matching_sequence_length`. The method must not modify `self`.

## Filtering logic (reuse existing)

Same as [get_filtered_by_min_seq_length](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) (lines 1967–1991) and [prepare_epoch_data](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) (5015–5026):

1. If `merged_segment_epochs is None` or `minimum_included_matching_sequence_length is None` or `<= 0`: return a **simple deep copy** of `self` (no filtering), so the API is safe and consistent.
2. Otherwise:
  - `good_merged_segment_epochs = merged_segment_epochs[merged_segment_epochs['num_epoch_t_bins'] >= minimum_included_matching_sequence_length]`
  - Filter `relevant_positions_df` to rows where `matching_found_relevant_merged_pos_epoch` is in `good_merged_segment_epochs['label']` and `matching_found_relevant_pos_epoch > -1` → `good_only_relevant_positions_df`
  - From that, get unique `matching_found_relevant_pos_epoch` → `good_only_included_epoch_labels`
  - Filter `matching_pos_epochs_df` to rows with `label` in `good_only_included_epoch_labels` → `good_only_matching_pos_epochs_df`

Use **copies** of the dataframes (e.g. `deepcopy` or `.copy()` where needed) so the returned instance does not share mutable state with `self`.

## Fields to set on the filtered copy

**Priority (important, used across the codebase):**

- **relevant_positions_df** — `good_only_relevant_positions_df` (copy).
- **matching_pos_epochs_df** — `good_only_matching_pos_epochs_df` (copy). Required by `MaskDataSource.init_from_list_of_MatchingPastFuturePositionsResult` and properties like `matching_past_position_df_list`.
- **merged_segment_epochs** — `good_merged_segment_epochs` (copy). Used by `_prepare_epoch_data` and list-based datasource.
- **matching_past_positions_df** — slice of filtered `relevant_positions_df` where `is_future_present_past == 'past'` (same as `_recompute_all_pos_dfs`, lines 1598–1599).
- **matching_future_positions_df** — slice where `is_future_present_past == 'future'` (lines 1604–1605).

**Recomputed for consistency with filtered rows:**

- **is_relevant_past_times** — boolean array from `(good_only_relevant_positions_df['is_future_present_past'].values == 'past')`.
- **is_relevant_future_times** — `(good_only_relevant_positions_df['is_future_present_past'].values == 'future')`.
- **n_relevant_past_times** — count of past rows in filtered df.
- **n_relevant_future_times** — count of future rows in filtered df.

**Leave unchanged (unchanged semantics or “irrelevant”):**

- **decoded_epoch_result**, **epoch_high_prob_mask**, **epoch_t_bins_high_prob_pos_mask**, **relevant_past_times**, **relevant_future_times** — same epoch and masks.
- **pos_matches_epoch_mask**, **n_total_possible_past_times**, **n_total_possible_future_times** — keep as-is (doc marks pos_matches_epoch_mask as “Basically IRRELEVANT”; totals refer to the original epoch).
- **centroids**, **centroids_df**, **a_centroids_search_segments_df**, **pos_segment_to_centroid_seq_segment_idx_map** — leave as default/unchanged; they can be recomputed later via `recompute_all()` if needed (optional follow-up: call `recompute_all()` on the copy when `should_defer_extended_computations=False` or leave deferred for performance).
- All other attribute/configuration fields (e.g. **epoch_id_key_name**, **merged_found_pos_epoch_id_key_name**, **max_allowed_trajectory_gap_seconds**, **should_defer_extended_computations**) — copy from `self`.

## Implementation approach

- **Location:** Add the new method on `MatchingPastFuturePositionsResult` in [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py), immediately after `get_filtered_by_min_seq_length` (after line 1991).
- **Signature:** `def filtered_copy_by_min_sequence_length(self, minimum_included_matching_sequence_length: int) -> "MatchingPastFuturePositionsResult":`
- **Construction:** Use `attrs.evolve(self, ...)` (already imported at line 20) to create the new instance, passing the updated and recomputed fields. For DataFrame/array fields, pass copies so the result does not share mutable state with `self`.
- **Edge case:** If `merged_segment_epochs` is None or filtering is skipped, return a copy built with `attrs.evolve(self, relevant_positions_df=deepcopy(self.relevant_positions_df), ...)` or equivalent (only replace fields that must be copies); alternatively use the same filtering path with “keep all” behavior so one code path builds the copy. Prefer: single path that first computes the three filtered dataframes (or no-op when no filtering), then builds one evolved instance from those.

## Optional: delegate to get_filtered_by_min_seq_length

Reuse the existing method for the filtering math: call `good_merged, good_positions, good_epochs = self.get_filtered_by_min_seq_length(minimum_included_matching_sequence_length)`. That method currently does not guard against `merged_segment_epochs is None` (it would raise). So:

- In the new method, first check: if `self.merged_segment_epochs is None` or `minimum_included_matching_sequence_length is None` or `minimum_included_matching_sequence_length <= 0`, return an evolved copy of `self` with deepcopied key dataframes (or a simple deepcopy of the whole object if the codebase has a standard way to copy this result).
- Else call `get_filtered_by_min_seq_length(minimum_included_matching_sequence_length)` and build the evolved instance from the returned tuple.

This keeps filtering logic in one place and avoids duplication.

## Summary

- Add **filtered_copy_by_min_sequence_length(minimum_included_matching_sequence_length: int)** that returns a new `MatchingPastFuturePositionsResult` without modifying `self`.
- When filtering is inapplicable (no merged segments or invalid min length), return a copy of `self` with deep-copied key dataframes.
- When applicable, use the same filtering as `get_filtered_by_min_seq_length` and set **relevant_positions_df**, **matching_pos_epochs_df**, **merged_segment_epochs**, **matching_past_positions_df**, **matching_future_positions_df**, and the recomputed **is_relevant_*_times** and **n_relevant_*_times** on the new instance via `attrs.evolve(self, ...)`.
- Use copies for all updated DataFrame/array attributes so the returned instance is independent of `self`.

