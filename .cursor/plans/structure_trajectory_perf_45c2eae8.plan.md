---
name: structure_trajectory perf
overview: Speed up [structure_trajectory.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\structure_trajectory.py) by caching the transition matrix, vectorizing its construction with NumPy broadcasting (same math as today), tightening the flat-to-(x,y) path, and replacing bare index prints with timed progress (percent, rate, ETA) using the stdlib only.
todos:
  - id: cache-transition
    content: Compute and store transition matrix once; reuse in _get_most_likely_trajectory
    status: completed
  - id: vectorize-transition
    content: Replace nested loops with broadcasted dist2 + column normalize + reshape to match column order i*nx+j
    status: completed
  - id: vectorize-xy
    content: Use column_stack (or equivalent) for flat→cm in get_most_likely_trajectory
    status: completed
  - id: progress-run-all
    content: Add timed progress lines (total, %, elapsed, ETA, flush) and optional verbose flag
    status: completed
isProject: false
---

# Optimize `Most_Likely_Trajectories` runtime and progress reporting

## Context

[`structure_trajectory.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\structure_trajectory.py) drives Viterbi decoding per ripple/spikemat. The hot spots in this file are:

1. **Redundant transition matrix work** — `_get_most_likely_trajectory` calls `_calc_transition_matrix(self.sd_bins)` on every spikemat ([lines 82–84](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\structure_trajectory.py)), but `sd_bins` and the spatial grid are fixed for the lifetime of `Most_Likely_Trajectories`. The same `(K × K)` matrix is rebuilt thousands of times.

2. **Nested Python loops in `_calc_transition_matrix`** — `O(n_x · n_y · n_x · n_y)` work with a small NumPy kernel inside the inner pair of loops ([lines 102–111](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\structure_trajectory.py)). This can be expressed as a single broadcasted Gaussian over “from” `(i, j)` and “to” grid `(mm, nn)`, then reshaped into columns `i * n_bins_x + j` to match the current layout **exactly** (preserve numerical behavior and column ordering so Viterbi stays bit-identical aside from floating-point reordering).

3. **Progress** — `run_all` only prints the loop index ([lines 47–48](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\structure_trajectory.py)).

## Proposed changes (all in this file)

### 1. Cache the transition matrix once

- After `self.sd_bins` is set in `__init__`, compute `self._transition_matrix = self._calc_transition_matrix(self.sd_bins)` once (or lazy-cache on first use if you prefer not to pay cost when `run_all=False` and trajectories are never run).
- In `_get_most_likely_trajectory`, set `self.viterbi_input["transition_matrix"] = self._transition_matrix` instead of recomputing.

This is typically the **largest** win when there are many spikemats.

### 2. Vectorize `_calc_transition_matrix`

- Keep `m`, `n`, `mm`, `nn = np.meshgrid(...)` as today.
- Build broadcast arrays matching the current distance: `(nn - i) ** 2 + (mm - j) ** 2` with `i ∈ [0, n_x)`, `j ∈ [0, n_y)` (same as the nested loops).
- Compute `kernel = exp(-dist2 / (2 * sd_bins**2 * time_window_s))` in one shot; normalize each column so column sums are 1 (same as `flat_transition / np.sum(flat_transition)`).
- Reshape to `(n_grid, n_grid)` with column index `i * n_bins_x + j` consistent with the existing assignment `transition_mat[:, i * n_bins_x + j]`.

**Memory note:** the broadcast intermediate has on the order of `n_x * n_y * n_x * n_y` elements (same order as the final dense `K×K` matrix you already allocate). For very large grids this is already heavy; the change trades Python loop overhead for one large vectorized pass—appropriate for typical `K` on the order of thousands.

### 3. Small cleanup in `get_most_likely_trajectory`

- Replace the list + `np.array(...).T` with a single `np.column_stack` (or equivalent) on 1D `(x_bins, y_bins)` in centimeters derived from `%` and `//` with `n_bins_x`, avoiding an extra Python list allocation.

### 4. Richer progress in `run_all` (stdlib only)

[`pyproject.toml`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\pyproject.toml) has no `tqdm`; avoid new dependencies.

- `import time` and use `time.perf_counter()` for elapsed time.
- Before the loop: `n = len(structure_data.spikemats)` (and optionally count how many indices are non-`None` for a line like “processing N spikemats (M non-empty)”).
- During the loop: on a sensible schedule (e.g. first, last, and every `max(1, n // 50)` iterations, or a `progress_every` constructor argument defaulting to something sane), print one line with:
  - current / total, percent
  - elapsed seconds
  - ETA from average time per iteration (or remaining × recent rate)
  - `flush=True` so logs appear on HPC/O2-style batch output

Optional: add `verbose: bool = True` to `__init__` / `run_all` so tests or batch jobs can silence prints without changing call sites much.

## Out of scope (optional follow-ups)

- **[`viterbi.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\viterbi.py)** recomputes `np.log(self.params["transition_matrix"])` every timestep; precomputing `log_transition` once would help further but touches another module—only pursue if profiling shows Viterbi dominates after the cache above.

## Verification

- Run existing tests for `replay_structure` if any reference this path; otherwise smoke-test: instantiate `Most_Likely_Trajectories` on a small fixture and compare `most_likely_trajectories` keys/shapes to a run **before** changes (or compare transition matrix max abs diff to old loop implementation on a tiny grid in a one-off check).
