---
name: 3D match trajectory lines
overview: Extend `Volumentric2DTimeSeriesPlotter` epoch overlay so that past/future "match" trajectories are rendered as 3D `vz.Line` visuals using the same time→z mapping as `pos3d`, reusing `MaskDataSource._prepare_epoch_data` for identical partitioning to the 2D viewer.
todos:
  - id: cache-mask-ds
    content: Add MaskDataSource cache + optional filter_epochs / minimum_included_matching_sequence_length / show_match_lines on set_epoch_visual_source; dummy filter_epochs when None
    status: completed
  - id: build-3d-lines
    content: "Implement _t_to_z + _build_match_lines_for_epoch: _prepare_epoch_data → vz.Line (x,y,z) per past/future trajectory under epoch_node"
    status: completed
  - id: lifecycle
    content: Clear match line visuals in _clear_active_slot_visuals; wire into _add_epoch_visuals_single_slot / _set_active_epoch_single_source; update docstring
    status: completed
isProject: false
---

# Add 3D match-line rendering to Volumentric2DTimeSeriesPlotter

## Context

- **Source of match polylines (2D):** `[MaskDataSource._prepare_epoch_data](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)` returns `curr_matching_past_future_positions_df_dict` with `'past'` / `'future'` keys mapping epoch ids → DataFrames with at least `x`, `y` (and usually `t`). This is already wired from `[MatchingPastFuturePositionsResult](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)` via `[init_from_list_of_MatchingPastFuturePositionsResult](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)` (lines 5214–5225).
- **Time→z in 3D:** `[setup_position_trajectory_curves](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` sets `self.t_min`, `self.t_max`, `self.z_scale`, and `z_vals = (t_vals - self.t_min) * self.z_scale` (lines 2001–2029). Match lines must use the **same** formula so they sit on the same time axis as the main `position_line`.
- **Where to hook:** Single-slot epoch UI already rebuilds overlays in `[_set_active_epoch_single_source](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` → `[_add_epoch_visuals_single_slot](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`. Match lines should be added **under the same** `epoch_node` from `[_get_or_create_single_slot_epoch_node](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` so they move with epoch grouping and are cleared with other slot visuals.

```mermaid
flowchart LR
  epochList[epoch_flat_mask_future_past_result]
  maskDS[MaskDataSource cached]
  prep[_prepare_epoch_data idx]
  dicts[curr_matching_past_future_positions_df_dict]
  lines3d[vz.Line x y z]
  epochList --> maskDS
  maskDS --> prep
  prep --> dicts
  dicts --> lines3d
```



## Design decisions

1. **Reuse `MaskDataSource` (no duplicate partition logic):** On `[set_epoch_visual_source](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`, build and cache `MaskDataSource` from the full `epoch_flat_mask_future_past_result` list (same factory as the 2D widget). Call `_prepare_epoch_data(an_epoch_idx, minimum_included_matching_sequence_length=...)` when rendering an epoch.
2. `**filter_epochs`:** `_prepare_epoch_data` indexes `filter_epochs` by `an_epoch_idx` (mainly for logging / returned `curr_epoch_row`). If callers do not pass real decode filter epochs, synthesize a **dummy** `pd.DataFrame` with `len(epoch_flat_mask_future_past_result)` rows (e.g. `original_epoch_idx` = `0..n-1`, optional placeholder `start`/`stop`) so indexing never misaligns. Expose optional `filter_epochs: Optional[pd.DataFrame] = None` on `set_epoch_visual_source`; when provided, assert or warn if length mismatches the result list.
3. **Optional minimum sequence length:** Add optional `minimum_included_matching_sequence_length: Optional[int] = None` on `set_epoch_visual_source` (mirrors `[PredictiveDecodingVispyWidget](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`) and pass through to `_prepare_epoch_data`.
4. **Toggle:** Add `show_match_lines: bool = True` so users can disable match overlays without losing contours/planes.
5. **Visuals:** For each trajectory DataFrame with ≥2 valid `x`,`y` points: if `'t'` is present and finite, compute `z = (t - self.t_min) * self.z_scale`; if `t` is missing, either skip or fall back to linear interpolation along the polyline in z (prefer requiring `t` for consistency). Use distinct RGBA defaults aligned with 2D semantics: **past** hue ~0 (reddish), **future** hue ~0.5 (cyan), width ~2–2.5, `order` slightly above main `position_line` (e.g. 7–9) so matches read clearly.
6. **Lifecycle:** Track created visuals in a list (e.g. `self._match_line_visuals` or reuse a named field). In `[_clear_active_slot_visuals](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`, detach every match line (`parent = None`) before removing contours/planes. Invalidate cached `MaskDataSource` when `set_epoch_visual_source` replaces the list.
7. **Scope:** Implement for **single-slot** path (`epoch_flat_mask_future_past_result` set) only. The legacy multi-group `[add_epoch_visuals](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` path can remain unchanged unless you explicitly need match lines there (would require passing per-epoch `MatchingPastFuturePositionsResult` or a shared datasource).

## Files to change

- `**[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`** (`Volumentric2DTimeSeriesPlotter` ~1716–3140):
  - New attrs fields: cached `MaskDataSource`, `minimum_included_matching_sequence_length`, `show_match_lines`, optional `filter_epochs` storage, list of match line visuals.
  - Helper: `_t_to_z(t)` using `self.t_min` and `self.z_scale` (single place for mapping).
  - Helper: `_build_match_lines_for_epoch(epoch_idx, parent_node)` calling `_prepare_epoch_data` and creating `vz.Line` 3D segments.
  - Update `set_epoch_visual_source` signature + docstring (usage example with optional `filter_epochs`).
  - Update `_clear_active_slot_visuals` and `_set_active_epoch_single_source` to build/clear match lines; call `_build_match_lines_for_epoch` from `_add_epoch_visuals_single_slot` or immediately after it when `show_match_lines` is True.

## Testing / validation

- Instantiate viewer with `curr_position_df` + bins + optional `p_x_given_n`, call `set_epoch_visual_source` with a real `epoch_flat_mask_future_past_result` list; scrub epoch slider and confirm past/future polylines appear in 3D and align in z with the main trajectory when times overlap.
- Toggle `show_match_lines=False` and confirm only contours/plane/extrusion remain.

