---
name: Fix VisPy line NaN crash
overview: Harden predictive time sampling and time-bin edge line construction so NaN/Inf never reach VisPy GL Line/Markers buffers—addressing access violations on epoch change that are easy to misattribute to the new matplotlib colormap path.
todos:
  - id: finite-u-colormap
    content: Make _predictive_time_rgb_for_key (and alpha path if needed) NaN/Inf-safe before clip + mpl/interp
    status: completed
  - id: finite-edge-lines
    content: Filter NaN/Inf in _time_bin_edge_vertical_lines; subset edge_vu consistently; require >=2 edges
    status: completed
  - id: edgesvis-sanitize
    content: Optionally sanitize edges_vis in plot_multiple_raster_plot_vispy before line/grid use
    status: completed
  - id: default-cmap-restore
    content: Set _active_predictive_time_colormap default back to cyan_magenta in library file
    status: completed
  - id: smoke-test
    content: Run small uv python checks for nan inputs and edge line helper
    status: completed
isProject: false
---

# Fix VisPy GL crash on epoch change (non-finite geometry/colors)

## Likely cause

VisPy `_GLLineVisual` crashes in `glDrawArrays` on Windows when **vertex positions or color attributes contain NaN/Inf**. That matches your symptom (access violation) better than “matplotlib colormap type” (values are already cast to Python `float`).

Two concrete gaps in the current code:

1. **Non-finite colormap coordinate `u`** — In `[predictive_time_colormap.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_time_colormap.py)`, `_predictive_time_rgb_for_key` does `uu = float(np.clip(u, 0.0, 1.0))`. For `u = nan`, `clip` leaves `nan`, so both the custom `np.interp` path and `matplotlib.colormaps[mpl](uu)` propagate **NaN RGB** into any consumer (timeline ticks, raster edge coloring, etc.).
2. **Non-finite time-bin edge times** — In `[vispy_raster.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_raster.py)`, `_time_bin_edge_vertical_lines` copies `edge_times` straight into `pos` with **no `isfinite` check**. If `time_bin_edges` from decoded data ever contains NaN (your epoch row also shows `start_idx`/`stop_idx` as NaN—worth treating decoded edges as potentially dirty), the **line positions** are NaN while colors may still be finite—enough to blow up the driver.

Trajectory lines in `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` already guard with `np.all(np.isfinite(trajectory_colors))` before creating `vz.Line`; time-bin edge lines do not.

## Implementation plan

### A. Central clamp for predictive-time sampling

In `_predictive_time_rgb_for_key` (same file as above):

- Coerce `u` with something equivalent to: if not `math.isfinite(uu)`, set `uu = 0.0` (or `0.5`—pick one documented behavior; `0.0` matches “early time” and is predictable).
- Then `uu = float(np.clip(uu, 0.0, 1.0))` and proceed.

Optionally apply the same for `alpha` in `predictive_time_rgba_u` (already clipped but not NaN-safe).

### B. Finite-only time-bin edge lines

In `_time_bin_edge_vertical_lines`:

- After `t_arr = np.asarray(edge_times, dtype=np.float32).ravel()`, build `finite_mask = np.isfinite(t_arr)` and either:
  - **Preferred:** keep only `t_arr[finite_mask]` and, if `edge_vu` is passed, subset `vu` with the same mask (requires `edge_vu.size == original n`; after filter, need `m >= 2` to draw), **or**
  - If fewer than 2 finite edges, `return None`.

This keeps `pos` and per-segment colors aligned when `edge_vu` is used with `predictive_time_rgba_u`.

### C. Optional: sanitize `edges_vis` at the callsite

In `plot_multiple_raster_plot_vispy`, when `edges_vis` is assigned from `time_bin_edges` or `linspace`, run the same finite filter **before** building lines (thin wrapper or shared helper in `vispy_raster.py` to avoid duplication). Ensures unit grid still uses finite `x0,x1` (already `float(epoch)` but harmless).

### D. Housekeeping

- Restore module default `_active_predictive_time_colormap` to `"cyan_magenta"` in `[predictive_time_colormap.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_time_colormap.py)` (it is currently set to `"cool"` with commented alternatives—avoid shipping a non-default in library code).

### E. Verification

- `uv run python` smoke test: `predictive_time_rgb(float('nan'))` and `predictive_time_rgba_u(float('nan'), 0.5)` return finite tuples.
- `_time_bin_edge_vertical_lines(np.array([0., np.nan, 1.]), ...)` returns either a line with only finite breakpoints or `None`, never NaN `pos`.

## What we are not changing (unless you want follow-up)

- Broad refactors of epoch clearing (`_clear_epoch_visuals` vs `time_bin_raster`)—only consider “clear raster scene when raster plot skipped” if crashes persist after A+B.

