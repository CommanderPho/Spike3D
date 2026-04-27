---
name: 8-way exit likelihood
overview: Add a new classmethod on `TransitionMatrixComputations` (next to the existing 2D transition matrix helper) that accumulates consecutive binned (x, y) moves into eight king-neighbor direction buckets and returns a per-bin probability tensor `(n_x, n_y, 8)`.
todos:
  - id: define-api
    content: Add `_compute_position_direction_exit_likelihood_8way` with docstring, direction table, assertions, vectorized counts + row L1 normalize
    status: completed
  - id: doc-crossref
    content: Cross-reference in `TransitionMatrixComputations` class docstring (2D helpers line)
    status: completed
  - id: optional-test
    content: Add minimal synthetic-path test if an appropriate test package/layout exists
    status: cancelled
isProject: false
---

# 8-direction exit likelihood from 2D binned positions

## Context

- [`_compute_position_transition_matrix_2d`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\transition_matrix.py) already uses **0-based** `binned_x` / `binned_y` indices and the flattening rule `flat = ix * n_y + iy` (x first, y second), aligned with `pf2D.occupancy`.
- [`neuropy.utils.mixins.binning_helpers.transition_matrix`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\binning_helpers.py) counts consecutive state pairs then **L1 row-normalizes**; rows with no transitions become NaN unless replaced.

## Behavior (proposed defaults)

1. **Input**: Same style as `_compute_position_transition_matrix_2d`: `xbin_labels`, `ybin_labels` (lengths `n_x`, `n_y`), and parallel 0-based integer arrays `binned_x_index_sequence`, `binned_y_index_sequence` (same assertions: equal length, max indices in range).
2. **Pairs**: For each `t`, consider `(ix0, iy0) -> (ix1, iy1)` from consecutive samples (one-step, same as the Markov transition matrix).
3. **Eligible “exit in one of 8 directions”**: `dx = ix1 - ix0`, `dy = iy1 - iy0`. Count toward direction buckets only if **Chebyshev distance** `max(|dx|, |dy|) == 1` and **not** `(dx, dy) == (0, 0)` (so true 8-neighbor moves only). **Stay** and **jumps** (e.g. `|dx|>1` or `|dy|>1`, or diagonal-equivalent but actually `(2,0)` etc.) are **not** assigned to any of the 8 channels (document this explicitly).
4. **Output**: `NDArray` of shape **`(n_x, n_y, 8)`** where `out[ix, iy, d]` is the fraction of **observed** one-step 8-neighbor exits from bin `(ix, iy)` that went in direction `d` (so for bins with at least one such exit, `sum_d out[ix, iy, d] == 1`). Bins with **no** qualifying exits: entire row of 8 zeros, or NaNs then filled — mirror the existing pattern with `nan_entries_replace_value: Optional[float] = 0.0` and optional `should_validate_normalization: bool` asserting non-degenerate rows sum to 1.
5. **Direction index order**: Define a single explicit ordering in the docstring and a small module-level or class-level constant, e.g. a length-8 list of `(dx, dy)` in **compass order starting at North** (adjust to match your plotting convention: **document** that `+ix` is east / `+iy` is north if that matches `pf2D`; if the codebase uses image-style y, state that so plots are not flipped silently). Example mapping (if +y is “north”): N `(0,1)`, NE `(1,1)`, E `(1,0)`, SE `(1,-1)`, S `(0,-1)`, SW `(-1,-1)`, W `(-1,0)`, NW `(-1,1)` — verify against one sample trajectory or existing axis labels in GUI code if needed during implementation.

## Implementation location and style

- Add **`@classmethod`** `TransitionMatrixComputations._compute_position_direction_exit_likelihood_8way` (or a slightly shorter name you prefer) **immediately after** `_compute_position_transition_matrix_2d` (after line ~239) in [transition_matrix.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\transition_matrix.py).
- Decorate with `@function_attributes(..., tags=['UNTESTED', 'AI', '2D', 'direction'], ...)` consistent with the neighboring method.
- **Vectorized** implementation preferred: build `dx`, `dy` arrays from shifted sequences; mask valid moves; map `(dx, dy)` to `d` via a **lookup table** (e.g. `9x9` center slice or dict keyed by `(dx, dy)`) then `np.add.at` or bincount on flattened `(ix, iy, d)` indices for speed and clarity.
- Update the class docstring bullet (~line 103) to mention the new helper alongside `_compute_position_transition_matrix_2d`.

## Tests (optional but recommended)

- If the package has a nearby test module for decoder/transition logic, add a tiny test: synthetic path on a 3x3 grid with known moves so each direction count is predictable; assert shape and row sums. If no suitable test home exists, skip tests unless you ask to add one.

## Out of scope (unless you want them later)

- Wrapping **toroidal** boundaries.
- A ninth “stay” or “long jump” probability mass (would change interpretation from “among 8-neighbor exits”).
