---
name: Raster exact time edges
overview: Wire optional `time_bin_edges` from `render_central_view` into `plot_multiple_raster_plot_vispy`, preferring decoder edges over `linspace` when valid; truncate edges to match `max_time_bins_to_show` like the time-bin strip. Tighten the widget guard when populating `_raster_kwargs`.
todos:
  - id: vispy-raster-kw
    content: Add time_bin_edges to plot_multiple_raster_plot_vispy; prefer over num_epoch_time_bins in loop; docstring
    status: completed
  - id: central-view-slice
    content: "In render_central_view raster block: slice edges to K+1, pass to plot_multiple_raster_plot_vispy, fallback num bins"
    status: completed
  - id: widget-guard
    content: "predicitive_decoding_vispy: safe numpy guard for time_bin_edges before _raster_kwargs"
    status: completed
  - id: pycompile
    content: Run py_compile on modified files
    status: completed
isProject: false
---

# Prefer exact `time_bin_edges` for raster bin guides

## Current state

- `[predictive_decoding_central_view.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_decoding_central_view.py)` already declares `**time_bin_edges: Optional[NDArray] = None**` on `render_central_view` (line 37) but the raster block (lines 371–386) only passes `**num_epoch_time_bins**` to `[plot_multiple_raster_plot_vispy](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_raster.py)`, which builds guides with `**np.linspace(x0, x1, n_tb+1)**`.
- `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` `_render_central_view` already copies `**single_epoch_decoded.time_bin_edges**` into `_raster_kwargs['time_bin_edges']` when a matching result exists (lines 763–770), so the data path is partly there.
- `[SingleEpochDecodedResult](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py)` exposes `**time_bin_edges**` (line 448).

## Behavior to implement

1. **Truncate edges to visible bins** — Match the time-bin row: `K = min(p_x_given_n.shape[2], max_time_bins_to_show)` when `p_x_given_n` is available. Use `**edges[:K+1]`** so vertical guides align with slices `[:, :, 0:K]` shown above the raster.
2. **Validity** — Treat edges as usable only if `time_bin_edges is not None` and `**np.asarray(time_bin_edges).ravel().size >= K + 1`** (if `p_x_given_n` missing, require at least 2 points and skip truncation or derive `K` from edge length − 1 consistently).
3. **Precedence** — If valid sliced edges exist, pass them to the raster helper and **do not** rely on `num_epoch_time_bins` for line placement (either pass `num_epoch_time_bins=None` or implement “edges win” inside vispy_raster). Otherwise keep today’s `**num_epoch_time_bins_for_raster`** fallback.

## Files / edits

### 1. `[vispy_raster.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_raster.py)`

- Add kw-only parameter `**time_bin_edges: Optional[np.ndarray] = None`** next to `**num_epoch_time_bins**` on `plot_multiple_raster_plot_vispy`.
- In the per-epoch loop (current lines 291–297):
  - If `**time_bin_edges**` is not `None` and `**np.asarray(time_bin_edges).size >= 2**`, pass the (caller-supplied) ravelled edges to `**_time_bin_edge_vertical_lines**`.
  - **Else** keep existing `**num_epoch_time_bins` + linspace** branch.
- Extend the docstring: `**time_bin_edges`** = absolute times (same axis as spikes / epoch `start`/`stop`); overrides `**num_epoch_time_bins`** when provided.

Optional hardening: if both are passed, document that edges take precedence (no need to error).

### 2. `[predictive_decoding_central_view.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_decoding_central_view.py)`

- In the `**can_plot_raster**` block, compute `**K**` the same way as `**n_bins_to_show**` (reuse `min(n_time_bins, max_time_bins_to_show)` when `p_x_given_n` is valid).
- `**raster_time_bin_edges**`: if `**time_bin_edges**` is not `None`, `te = np.asarray(time_bin_edges, dtype=np.float64).ravel()`, and `**te.size >= K + 1**`, set `**raster_time_bin_edges = te[:K+1].astype(np.float32)**` (or pass float64 through). Else `**None**`.
- Call `**plot_multiple_raster_plot_vispy(..., time_bin_edges=raster_time_bin_edges, num_epoch_time_bins=(None if raster_time_bin_edges is not None else num_epoch_time_bins_for_raster))**` — or only pass `**num_epoch_time_bins**` when edges are `None`, per vispy API choice above.

### 3. `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`

- Replace `**if time_bin_edges:**` with a **size-safe check** (e.g. `**time_bin_edges is not None and np.size(time_bin_edges) >= 2`**) so NumPy arrays never go through ambiguous truthiness.

## Verification

- `**uv run python -m py_compile`** on the three touched files.
- Quick mental check: when `**epoch_flat_mask_future_past_result**` is missing, `**time_bin_edges**` stays `None` and behavior matches current linspace fallback.

