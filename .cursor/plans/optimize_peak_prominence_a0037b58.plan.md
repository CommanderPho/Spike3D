---
name: Optimize peak prominence
overview: Optimize `compute_2d_peak_prominence` and `compute_2d_dt_posterior_peak_promenences` in `peak_prominence2d.py` by batching ndimage operations across all time bins, reusing pre-allocated buffers, and eliminating redundant computations — all while preserving identical output.
todos:
  - id: batch-maxfilter
    content: Vectorize maximum_filter and local_max computation across all time bins using a 3D (3,3,1) footprint, plus batched per-slice min
    status: completed
  - id: inline-buffers
    content: Inline the per-slice prominence computation into the loop with pre-allocated seed_buf and buffer reuse via np.copyto
    status: completed
  - id: elim-redundancy
    content: Eliminate redundant peak_heights extraction (was computed inside compute_2d_peak_prominence then re-extracted by caller)
    status: completed
  - id: verify-output
    content: Verify the optimized code produces identical output by testing with a small synthetic 3D array
    status: completed
isProject: false
---

# Optimize 2D Peak Prominence Computation

Target file: [peak_prominence2d.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\peak_prominence2d.py) lines 2058-2159.

## Current bottlenecks

The per-time-bin loop in `compute_2d_dt_posterior_peak_promenences` (line 2142) calls `compute_2d_peak_prominence` for every `t_idx`. Each call:

- Re-creates `ndimage.generate_binary_structure(2, 2)` (invariant, line 2075)
- Calls `ndimage.maximum_filter` on a single 2D slice (Python loop overhead vs. one batched C call)
- Allocates a new `seed = Z_2d.copy()` array (line 2083)
- Allocates a new `reconstructed` array (line 2086, from skimage)
- Returns `peak_coords` and `prominences` but **not** `peak_heights`, which the caller redundantly re-extracts (line 2150)

## Optimization strategy

### 1. Batch `maximum_filter` and `local_max` across all time bins

Instead of calling `maximum_filter` per slice, extend the 2D footprint to 3D `(3,3,1)` and call once on the full `(n_x, n_y, n_t)` array:

```python
neighborhood_2d = ndimage.generate_binary_structure(2, 2)
neighborhood_3d = neighborhood_2d[:, :, np.newaxis]  # (3,3,1)
max_filtered = ndimage.maximum_filter(a_p_x_given_n, footprint=neighborhood_3d)
local_max_3d = (a_p_x_given_n == max_filtered)
del max_filtered  # free immediately
slice_mins = np.min(a_p_x_given_n, axis=(0, 1))  # (n_t,)
local_max_3d &= (a_p_x_given_n > slice_mins[np.newaxis, np.newaxis, :])
```

This replaces `n_t_bins` Python-level ndimage calls with a single C-level pass. The `max_filtered` temporary (same dtype as input) is freed immediately after producing the boolean `local_max_3d` (1/8 the memory of the input).

### 2. Buffer reuse for `reconstruction` seed

Pre-allocate a single `(n_x, n_y)` float buffer outside the loop and reuse it via `np.copyto` instead of allocating a fresh `.copy()` each iteration:

```python
seed_buf = np.empty((n_x_bins, n_y_bins), dtype=a_p_x_given_n.dtype)
```

Inside the loop:

```python
np.copyto(seed_buf, Z_2d)
seed_buf[local_max_slice] = -np.inf
reconstructed = reconstruction(seed_buf, Z_2d, method="dilation")
```

### 3. Eliminate redundant `peak_heights` extraction

Currently `compute_2d_peak_prominence` computes `peak_heights = Z_2d[local_max]` (line 2080) then discards it, and the caller re-extracts the same values at line 2150. Since the function is being inlined into the loop, `peak_heights` will be available directly — no second fancy-index needed.

### 4. Inline the per-slice computation

Inline the body of `compute_2d_peak_prominence` into the time-bin loop in `compute_2d_dt_posterior_peak_promenences`. This:

- Eliminates per-slice method call overhead (classmethod dispatch x n_t_bins)
- Enables buffer reuse (seed, thresholds) across iterations
- Removes the invariant `generate_binary_structure` call from the hot path

Keep `compute_2d_peak_prominence` unchanged as a standalone public method for external callers.

### 5. Minor streamlining

- Use `np.argwhere(local_max_slice)` instead of `np.column_stack(np.nonzero(local_max_slice))` (cleaner, same result)
- Pre-extract `local_max_slice = local_max_3d[:, :, t_idx]` as a view (zero-copy)
- Reuse existing `threshold_buf` and `seed_buf` (propagation seed) which are already in place

## Memory impact summary

- **Removed per-iteration**: `.copy()` allocation for seed (n_x * n_y * 8 bytes each iter)
- **Added once**: `local_max_3d` boolean array (n_x * n_y * n_t bytes, 1/8 of input), freed after loop
- **Added once**: `seed_buf` (n_x * n_y * 8 bytes), reused every iteration
- **Temporary**: `max_filtered` (same size as input), freed immediately after `local_max_3d` is computed
- **Net**: steady-state memory during the loop is reduced by eliminating per-iteration allocations; peak memory comparable

## Preserved behavior

- `compute_2d_peak_prominence` public API is unchanged
- `compute_2d_dt_posterior_peak_promenences` returns the same `(epoch_promenence_tuples, epoch_masks)` structure with identical values
- All downstream consumers (`compute_posterior_peak_promenences`, etc.) unaffected

