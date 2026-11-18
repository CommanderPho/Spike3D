---
name: Centroid arrow color index fix
overview: Fix the centroid arrow color IndexError in the central view by building a time-bin-indexed color array so arrow and marker lookups use original time bin indices (including missing centroids) instead of position indices.
todos: []
isProject: false
---

# Centroid arrow color index fix

## Problem

In [predictive_decoding_central_view.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_decoding_central_view.py):

- `centroid_colors` is built with shape `(n_centroids, 4)` and indexed by **position** in the valid-centroid list (0..n_centroids-1).
- Arrow code uses `original_index_start` and `original_index_end`, which are **time bin indices** (row indices in the full `centroids_df`), e.g. 0, 2, 5, 7.
- `_safe_color_map_fn(t_idx)` does `centroid_colors[t_idx, :]`, so when `t_idx` is a time bin index (e.g. 7) and there are only 4 valid centroids, this raises **IndexError**.

## Approach

Build a single **time-bin-indexed** color array that covers all original time bin indices (0 through the maximum index we need), so that:

- Every time bin index has a defined color (even when that time bin has no valid centroid).
- Arrow and marker color lookups use this array indexed by time bin index, eliminating the index mismatch.

## Implementation

**File:** [predictive_decoding_central_view.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_decoding_central_view.py)

### 1. Build `color_by_time_bin` (time-bin-indexed palette)

Immediately after computing `original_indices` and before building `centroid_colors` (around lines 87–92):

- Compute the number of time bin indices to support:
  - `n_time_bin_slots = len(time_bin_colors)` when there are no valid centroids.
  - When `original_indices` is non-empty: `n_time_bin_slots = max(len(time_bin_colors), int(np.max(original_indices)) + 1)` so we cover all indices that appear in the centroid data (and at least the caller-provided time bins).
- Allocate `color_by_time_bin = np.zeros((n_time_bin_slots, 4), dtype=np.float32)`.
- Fill `color_by_time_bin[0:len(time_bin_colors)]` from `time_bin_colors`.
- For any remaining slots (if `n_time_bin_slots > len(time_bin_colors)`), set `color_by_time_bin[len(time_bin_colors):]` to the existing fallback `(1.0, 1.0, 1.0, 0.8)`.

This gives unique colors for all original time bin indices 0..n_time_bin_slots-1, including indices that have no valid centroid.

### 2. Use `color_by_time_bin` for centroid marker colors

Replace the current `centroid_colors` construction (lines 90–92) with:

- `centroid_colors = color_by_time_bin[original_indices]` (NumPy indexing; one color per valid centroid, by time bin index). If an index is out of range (should not happen if `n_time_bin_slots` is computed as above), you can clip: `np.clip(original_indices, 0, n_time_bin_slots - 1)` or keep the same `n_time_bin_slots` logic so all `original_indices` are in range.

Keep the rest of the centroid markers block unchanged (face_color still uses `centroid_colors`).

### 3. Use `color_by_time_bin` for arrow color lookups

- **Single-arrows object branch (lines 127–129):** Change `_safe_color_map_fn` to index by time bin with bounds check:
  - `_safe_color_map_fn = lambda t_idx: tuple(color_by_time_bin[t_idx]) if (0 <= t_idx < n_time_bin_slots) else (1.0, 1.0, 1.0, 0.8)`
  - Use this for `_original_index_start_colors_list` and `_original_index_end_colors_list` (no change to how they are computed, only the function used).
- **Per-arrow loop (line 151):** Replace `an_arrow_color = tuple(time_bin_colors[t_idx]) if t_idx < len(time_bin_colors) else ...` with the same logic: `an_arrow_color = tuple(color_by_time_bin[t_idx]) if (0 <= t_idx < n_time_bin_slots) else (1.0, 1.0, 1.0, 0.8)`.

### 4. Legacy arrow branch (segment_Vp_deg, lines 168–178)

In the `else` branch that uses `segment_Vp_deg`, the per-arrow color uses `t_idx = original_indices[centroid_idx]` and then `time_bin_colors[t_idx]`. Replace with `color_by_time_bin[t_idx]` with the same bounds check (0 <= t_idx < n_time_bin_slots) and fallback.

### 5. Scope of `n_time_bin_slots`

`n_time_bin_slots` (and `color_by_time_bin`) must be in scope for:

- Building `centroid_colors`.
- The `_safe_color_map_fn` used for the single-arrows object.
- The per-arrow loop in the `use_new_centroid_arrows` else branch.
- The legacy `segment_Vp_deg` arrow loop.

So define `n_time_bin_slots` and `color_by_time_bin` once inside the `if np.any(valid_mask):` block (right after `original_indices`), and use them in all of the above places.

## Summary of edits


| Location                                 | Change                                                                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| After line 87 (`original_indices = ...`) | Add computation of `n_time_bin_slots`, allocation and fill of `color_by_time_bin`.                    |
| Lines 90–92 (centroid_colors)            | Replace with `centroid_colors = color_by_time_bin[original_indices]` (with safe indexing if desired). |
| Line 127 (_safe_color_map_fn)            | Use `color_by_time_bin` and `n_time_bin_slots` instead of `centroid_colors` and its length.           |
| Line 151 (an_arrow_color)                | Use `color_by_time_bin[t_idx]` with `0 <= t_idx < n_time_bin_slots` and fallback.                     |
| Line 178 (legacy arrow color)            | Use `color_by_time_bin[t_idx]` with same bounds check and fallback.                                   |


No changes are required in PredictiveDecodingComputations.py; the fix is confined to the central view module.

## Optional robustness

- If `original_indices` can ever contain values beyond `len(time_bin_colors)-1` (e.g. from a different epoch or misalignment), the `n_time_bin_slots = max(len(time_bin_colors), int(np.max(original_indices)) + 1)` logic already extends the palette and fills extra slots with the fallback color, so indexing remains valid.

