---
name: Viewport posterior aggregation
overview: "Replace the bin-snapped raster plan: keep the spike raster on the full timeline viewport, but compute the displayed posterior by summing and renormalizing all decoded time bins whose intervals lie strictly inside `[start_t, end_t]` (with a fallback to at least one bin), then apply the existing low-value NaN mask and show the bin count in the title."
todos:
  - id: helpers-bin-edges
    content: Add _decoder_bin_left_right_edges + _indices_strictly_in_viewport; clamp searchsorted in update()
    status: completed
  - id: on-window-indices
    content: "on_window_changed: compute last_included_posterior_bin_indices (min 1 fallback); sync last_window_index/time (first bin); call update paths so layers refresh"
    status: completed
  - id: layer-aggregate
    content: "TimeSynchronizedGenericPlotterLayer._update_plots: nansum over parent.last_included_posterior_bin_indices, renormalize, then drop_below_threshold"
    status: completed
  - id: title-nbins
    content: "Decoder _update_plots title: append | n_bins: K"
    status: completed
  - id: cancel-raster-snap-todos
    content: Abandon PendingNotebookCode bin-snapped decoding_window proxy / helper from old plan (no code unless still wanted)
    status: completed
isProject: false
---

# Viewport-aggregated decoder posterior

## What changes vs the old plan

The prior plan ([align_raster_to_decoded_bin_7ea42c4e.plan.md](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/.cursor/plans/align_raster_to_decoded_bin_7ea42c4e.plan.md)) clipped/snapped the **decoding-row raster** to one decoded bin. **Drop that.** No `PendingNotebookCode` proxy changes for bin-snapped `update_zoomed_plot`. Optionally remove or archive those pending todos when executing.

## Current data path (brief)

- [TimeSynchronizedPositionDecoderPlotter.on_window_changed](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py) calls `update(start_t)` only; one index from `searchsorted`.
- The heatmap is drawn in [TimeSynchronizedGenericPlotterLayer._update_plots](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedGenericPlotterLayer.py) via `image = np.squeeze(main_data[:, :, curr_time_window_index])` then `drop_below_threshold` → NaN.

## Design

```mermaid
flowchart LR
  V[Viewport start_t end_t]
  B[Strictly_inside bin indices]
  A[nansum over time]
  N[Renormalize sum space to 1]
  T[drop_below_threshold]
  V --> B
  B --> A --> N --> T
```



1. **Bin edges** — Module-level helpers (or private methods) on/near the decoder plotter: prefer `active_one_step_decoder.time_bin_container.left_edges` / `right_edges` truncated to `len(time_window_centers)`; else infer from `time_window_centers` (median step, midpoint boundaries) as in the superseded draft. Use correct identifier `active_one_step_decoder` (no typos).
2. **Strict inclusion** — Include bin `i` iff `left[i] >= start_t` and `right[i] <= end_t` (closed viewport; matches “inside the active window”).
3. **At least one bin** — If the strict set is empty, **fallback**: same as today — `i0 = clamp(searchsorted(centers, start_t, side='left'), 0, n-1)` so the posterior is never undefined.
4. **Aggregation** — If `len(indices) > 1`: `image = np.nansum(main_data[:, :, indices], axis=-1)`, then `image /= np.nansum(image)` (guard: if sum is 0 or non-finite, avoid divide-by-zero; e.g. leave zeros or skip normalize). If `len(indices) == 1`, keep current single-slice path (behavior identical to now).
5. **Threshold** — Apply the same `drop_below_threshold` NaN masking as in `TimeSynchronizedGenericPlotterLayer._update_plots` **after** renormalization.
6. **State on plotter** — e.g. `last_included_posterior_bin_indices: Optional[NDArray]` (or list) set in `on_window_changed(start_t, end_t)` before stack `_update_plots`. In `update(t)` (scrubber / legacy), set it to a **length-1** array matching the clamped `searchsorted` index so export and single-`t` updates stay consistent.
7. `**TimeSynchronizedGenericPlotterLayer._update_plots`** — If `getattr(self.parent, 'last_included_posterior_bin_indices', None)` is not `None`, use those indices for the slice/sum path above; otherwise keep existing single-index logic. Other parents omit the attribute → no behavior change.
8. **Title** — In [TimeSynchronizedPositionDecoderPlotter._update_plots](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py) (~521–530), append the count, e.g. `| n_bins: K` (K = `len(last_included_posterior_bin_indices)`).
9. **Auxiliary UI state** — Keep `last_window_index` / `last_window_time` meaningful for PBE border, trajectory, and existing `t_idx` display: **recommend** using the **first** included index and its center time (or fallback index) so multi-bin mode does not break `decoded_time_bins_info_df.iloc[...]` or `curr_recent_trajectory` without a larger mixin change.
10. **Hardening** — Clamp `last_window_index` after `searchsorted` in `TimeSynchronizedPositionDecoderPlotter.update` to `[0, n-1]` (still valuable).

## Files to touch

- [TimeSynchronizedPositionDecoderPlotter.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py): helpers, `on_window_changed` computes indices + updates `last_window`_*, `update` sets single-bin indices, init `last_included_posterior_bin_indices = None`, title.
- [TimeSynchronizedGenericPlotterLayer.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedGenericPlotterLayer.py): branch in `_update_plots` for multi-bin sum → normalize → threshold.

## Verification

- Widen the synced timeline window: title `n_bins` increases when more full decoded bins fit strictly inside `[start_t, end_t]`; heatmap changes smoothly.
- Narrow until no bin fits strictly: fallback shows one bin; `n_bins` is 1.
- `update(t)` / export path: `n_bins` stays 1 and matches previous single-frame behavior.

