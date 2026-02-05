---
name: Generalize contours for pipeline reuse
overview: Extend the vispy_helpers contour API to support per-mask contour data and optional view/list pairing, then replace the inline contour-rendering block in PredictiveDecodingComputations._render_central_view with a call to the new API so the same logic is reused and the pipeline no longer duplicates contour extraction.
todos: []
isProject: false
---

# Generalize contour API to replace pipeline block

## Goal

Replace the inline contour-rendering block in [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) (lines 7863–7916) with the shared API in [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py), so behavior is unchanged and contour extraction runs in one place (reuse and performance).

## Current block behavior (summary)

- **Data**: `epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask` → 3D `(N_XBINS, N_Y_BINS, n_mask_t_bins)`. Each slice is transposed to `(n_y_bins, n_x_bins)` before `find_contours`. World bounds: `x_min`, `x_max`, `y_min`, `y_max`.
- **Colors**: `self._time_bin_colors(n_mask_t_bins, alpha=0.7)` → `(n_bins, 4)` RGBA; one color per mask index.
- **Shared views**: For every contour (all t_bins), draw the same Line on each of `(past_view, posterior_2d_view, future_view)` and append to `past_mask_contours`, `posterior_mask_contours`, `future_mask_contours` respectively (views/lists from `_update_dict`).
- **Per-time-bin views**: For each `t_idx`, draw only that mask’s contours on `time_bin_views[t_idx].scene` and append those Lines to `posterior_mask_contours` (order=11). Guard: `if t_idx < len(time_bin_views)`.
- **Output**: Update `_update_dict` with `posterior_mask_contours` and the three named contour lists.

## Gaps in current API

1. **Per-mask contour data**: The pipeline needs “contours for mask index i only” for `time_bin_views[i]`. Today `contours_from_masks` returns a single flat list, so we cannot attach only mask-i contours to the i-th time-bin view without either calling the API once per mask (wasteful) or returning structure per mask.
2. **View/list pairing**: The block draws the same (flat) contours on several views and appends to different lists. The current `render_contours(..., parents=[...])` creates one list of Line lists (one per parent) but does not append into caller-provided lists or support the “(view, list_to_append)” pairing.

## Implementation plan

### 1. Add per-mask return option in vispy_helpers

**File**: [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)

- Add optional parameter to `contours_from_masks`:
  - `return_per_mask: bool = False`
- **When `return_per_mask=False**` (default): keep current behavior; return type remains `List[Tuple[NDArray, Tuple[float, float, float, float]]]` (flat list).
- **When `return_per_mask=True**`: return type `List[List[Tuple[NDArray, Tuple[float, float, float, float]]]]` — one list per mask index; for mask index `i`, `result[i]` is the list of `(pos, rgba)` for that mask only. Mask indices with no contours get an empty list. This allows one contour extraction pass to serve both “all contours” (flatten) and “contours for mask i” (result[i]).
- Implementation: in the loop over `mask_list`, append to a per-mask list (e.g. `per_mask_out[idx]`) and also to the flat list when `return_per_mask` is False; when `return_per_mask` is True, build `per_mask_out` and return it (caller can flatten if needed). Single pass, no double computation.

### 2. Replace the inline block in PredictiveDecodingComputations

**File**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

- **Import**: Ensure `contours_from_masks` and `create_contour_line_visuals` (and optionally `VispyHelpers`) are imported from `pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers`.
- **Remove** the existing block from the “Renders the contours” comment through the end of the contour logic (including the `_update_dict.update` and the loop that writes `past_mask_contours`, `posterior_mask_contours`, `future_mask_contours` into `_update_dict`), but keep the final `return _update_dict`.
- **Insert** replacement logic that:
  1. Keeps the same guard: `if self.epoch_flat_mask_future_past_result is not None and (new_epoch_idx < len(...))` and the check for `epoch_result_for_contours` and `epoch_t_bins_high_prob_pos_mask`.
  2. Builds masks in the convention expected by vispy_helpers: list of 2D arrays **(n_rows, n_cols) = (y, x)**. From `per_t_bin_mask = epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask` (shape `(N_XBINS, N_Y_BINS, n_mask_t_bins)`), use `masks = [per_t_bin_mask[:, :, t_idx].T for t_idx in range(n_mask_t_bins)]` so each mask is `(N_Y_BINS, N_XBINS)` and matches vispy’s row→y, col→x mapping.
  3. Gets colors: `contour_time_bin_colors = self._time_bin_colors(n_mask_t_bins, alpha=0.7)` and passes them as `colors=[tuple(contour_time_bin_colors[i]) for i in range(n_mask_t_bins)]` (or equivalent) to `contours_from_masks`.
  4. Calls `contours_from_masks(masks, x_bounds=(x_min, x_max), y_bounds=(y_min, y_max), colors=..., level=0.5, return_per_mask=True)`. Gets `contour_data_per_mask` (list of list of (pos, rgba)).
  5. Builds flat contour data for shared views: `contour_data_flat = [item for sublist in contour_data_per_mask for item in sublist]`.
  6. Builds `views_and_lists_to_draw` as now: `[(past_view, past_list), (posterior_2d_view, posterior_list), (future_view, future_list)]` from `_update_dict`, filtering out None views. For each `(view, cont_list)`, gets `view.scene` (skip if None), calls `lines = create_contour_line_visuals(contour_data_flat, view.scene, line_width=2, order=10)` and extends `cont_list` with `lines`.
  7. For per-time-bin: `time_bin_views` from `_update_dict` (or self). For `t_idx` in `range(min(len(contour_data_per_mask), len(time_bin_views)))`, get `time_bin_views[t_idx].scene`, call `lines = create_contour_line_visuals(contour_data_per_mask[t_idx], scene, line_width=2, order=11)` and extend `posterior_mask_contours` with those lines.
  8. Updates `_update_dict` with `posterior_mask_contours` and the three named contour lists exactly as the current code does (same keys and structure).

No new helper that takes “view + list” is strictly required: the caller can do the loop over `(view, cont_list)` and call `create_contour_line_visuals` then `cont_list.extend(lines)`. Optionally, a small helper in vispy_helpers could accept a list of `(parent, list_to_append)` and do that loop for symmetry with `render_contours(..., parents=...)`; the plan keeps the replacement minimal and does not add that unless you want it.

### 3. Mask convention and imports

- **Convention**: vispy_helpers treats each mask as (n_rows, n_cols) with row → y_world, col → x_world. The pipeline’s 3D array is (N_XBINS, N_Y_BINS, n_t_bins); passing `.T` per slice yields (n_rows, n_cols) = (N_Y_BINS, N_XBINS), which matches the existing world mapping in the block (x_world from col, y_world from row).
- **Imports**: Add or confirm `from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import contours_from_masks, create_contour_line_visuals` (and remove any redundant `measure`/`vz` usage that was only used in the removed block).

### 4. Edge cases (preserve current behavior)

- If `time_bin_views` is shorter than `n_mask_t_bins`, only draw on existing time-bin views (same as current `if t_idx < len(time_bin_views)`).
- If a view in `views_and_lists_to_draw` has `view.scene is None`, skip creating visuals for that view (or document that callers must pass views with non-None scene).
- Empty masks: `contours_from_masks` already skips masks with no contour; per-mask list for that index will be empty.

## Files to touch

- [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py): add `return_per_mask` to `contours_from_masks`, implement per-mask aggregation and return type.
- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py): add import; replace lines 7863–7916 with the new logic; remove unused `measure`/contour-local imports if they become redundant in that method.

## Result

- One shared contour implementation (vispy_helpers) used by both standalone callers and the pipeline.
- Pipeline behavior unchanged: same views, same lists, same order (10 for shared, 11 for time-bin), same _update_dict shape.
- Single contour extraction pass for the pipeline (return_per_mask=True), then flatten for shared views and per-mask for time_bin_views.

