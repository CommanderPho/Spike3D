---
name: 2D position transition matrices
overview: Add 2D (x, y) binned-position transition-matrix computation in `transition_matrix.py` by flattening grid states with the same (n_x, n_y) / C-order convention as `build_spanning_grid_matrix` in NeuroPy, reusing the existing `transition_matrix()` implementation unchanged. Do not modify `_compute_position_transition_matrix` or any 1D-only logic.
todos:
  - id: helpers-flat-xy
    content: Add flat_index / unravel / optional reshape helpers with C-order (n_x, n_y) convention documented
    status: completed
  - id: compute-tm-2d
    content: Implement _compute_position_transition_matrix_2d mirroring 1D flow on flattened sequence; no edits to _compute_position_transition_matrix
    status: completed
  - id: docs-tests
    content: Document pf2D usage and add minimal test or runnable example
    status: completed
isProject: false
---

# 2D binned (x, y) transition matrix support

## Context

- Core counting/normalization already lives in [`neuropy/utils/mixins/binning_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\binning_helpers.py) as `transition_matrix(state_sequence, markov_order, max_state_index, ...)`. It only needs a **1D integer state ID** in `[0, max_state_index]`.
- 2D grid layout in this codebase is documented in the same file: `build_spanning_grid_matrix` sets `original_data_shape = (num_cols, num_rows)` = `(len(x_values), len(y_values))`, i.e. **first axis = x bins, second = y bins**, and C-order flatten yields **`flat = ix * n_y + iy`** (equivalently `np.ravel_multi_index((ix, iy), (n_x, n_y), order='C')`). This matches the existing TODO sketch for pf2D in the `transition_matrix` docstring (lines 949–977).
- Current 1D API: [`TransitionMatrixComputations._compute_position_transition_matrix`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\transition_matrix.py) — **leave this method and its body untouched.**

## Implementation (all in [`transition_matrix.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\transition_matrix.py))

### 1. Small helpers (module-level or `@staticmethod`, documented)

- **`position_flat_index_from_xy(ix, iy, n_y_bins)`** — scalar or vectorized; implement via `ix * n_y_bins + iy` (or `np.ravel_multi_index` with `order='C'`) and document equivalence to `occupancy.shape == (n_x, n_y)` row-major flatten.
- **`position_unravel_flat(flat, n_x_bins, n_y_bins)`** — `np.unravel_index(flat, (n_x_bins, n_y_bins), order='C')`.
- **`reshape_square_tm_to_grid(T, n_x_bins, n_y_bins) -> NDArray`** — optional convenience: reshape `(n_x*n_y, n_x*n_y)` to `(n_x, n_y, n_x, n_y)` so `out[ix_from, iy_from, ix_to, iy_to]` is readable (no change to stored math, just a view/reshape).

### 2. New classmethod: `_compute_position_transition_matrix_2d`

Mirror the **control flow** of `_compute_position_transition_matrix` (same `n_powers`, `use_direct_observations_for_order`, `should_validate_normalization`, same calls to `transition_matrix(..., nan_entries_replace_value=0.0, ...)` and optional `np.linalg.matrix_power` branch), but:

- **Inputs**: `xbin_labels`, `ybin_labels` (same role as 1D: define `n_x = len(xbin_labels)`, `n_y = len(ybin_labels)`), plus **`binned_x_index_sequence`** and **`binned_y_index_sequence`** — **0-based bin indices** in `[0, n_x-1]` and `[0, n_y-1]` (callers subtract 1 from `binned_x` / `binned_y` labels, same pattern as [`DirectionalPlacefieldGlobalComputationFunctions.compute_decoder_transition_matricies`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) line 835).
- **Alignment**: `assert len(binned_x_index_sequence) == len(binned_y_index_sequence)`; optionally support boolean mask or paired `dropna` is caller’s responsibility (document clearly).
- **State space**: `num_position_states = n_x * n_y`, `max_state_index = num_position_states - 1`.
- **Flatten**: `flat_sequence = position_flat_index_from_xy(binned_x_index_sequence, binned_y_index_sequence, n_y)` then reuse existing logic on `flat_sequence`.
- **Output**: `List[NDArray]` each shape `(n_x * n_y, n_x * n_y)` — parallel to 1D’s list of square matrices.

### 3. Docstrings and usage

- In the class docstring or method docstring, add a **short usage block** with `pf2D.xbin_labels`, `pf2D.ybin_labels`, and `filtered_pos_df['binned_x'].to_numpy() - 1` (and same for y).
- Explicitly state the **flattening convention** so results align with `pf2D.occupancy.shape`, decoder flat indices, and `build_spanning_grid_matrix`.

### 4. Out of scope for this task (unless you want them in the same PR)

- **Pipeline wiring** (e.g. a `compute_decoder_transition_matricies_2d` on `TrackTemplates`) — not required to satisfy “implement support” in this module alone.
- **`plot_transition_matricies`**: matrices become very large (`(n_x*n_y)^2`); no change unless you want a separate thin wrapper (e.g. plot a single order, or a downsampled block). Default plan: **no new plot API** unless requested.
- **`_compute_expected_velocity_out_per_node`**: it uses **`j - i` in 1D index space**, which is **wrong for flattened 2D**. Do **not** change that function; if you need 2D kinematic summaries later, add a **new** helper (e.g. expected Δx/Δy from unraveled indices) rather than overloading the 1D semantics.

### 5. Tests

- Add a small **unit test** (if the package has a test tree for this module; otherwise a doctest-style example in docstring): e.g. `2x2` grid, hand-known transitions, assert shape `(4,4)` and row normalization for non-zero rows.

## Note (pre-existing)

The file currently defines `_compute_expected_velocity_list_dict` twice (lines ~1114 and ~1240); the second definition wins. This plan does **not** alter that duplication.
