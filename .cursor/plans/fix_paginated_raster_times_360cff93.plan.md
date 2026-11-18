---
name: Fix paginated raster times
overview: Apply the same `.spikes.fixing_time_column()` canonicalization used by SpikeRaster2D/Window to the paginated decoded-epochs attached rasters path, at RankOrderRastersDebugger construction—after `co_filter_epochs_and_spikes` so its `t_rel_seconds` override still works.
todos:
  - id: fixup-rank-order-init
    content: Call global_spikes_df.spikes.fixing_time_column() at start of RankOrderRastersDebugger.init_rank_order_debugger before storing/building plots
    status: completed
isProject: false
---

# Fix time columns for paginated decoded-epochs rasters

## Problem

KDiba spikes often have disagreeing `t` vs `t_rel_seconds`, with the accessor pointing at `t_rel_seconds`. SpikeRaster2D/Window was fixed by calling `.spikes.fixing_time_column()` at construction (backs up bad `t` → `_t_BAK`, copies `t_rel_seconds` into `t`, sets `time_variable_name='t'`).

[`PhoPaginatedMultiDecoderDecodedEpochsWindow.plot_full_paginated_decoded_epochs_window`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py) builds `attached_ripple_rasters_widget` via `RankOrderRastersDebugger`, which never runs that fixup on the shared `global_spikes_df`. Spot-building in `NewSimpleRaster.build_spikes_all_spots_from_df` already fixes a local copy, but epoch assignment, `time_sliced` in `get_active_epoch_spikes_df`, and stored raster state still use the unfixed frame.

```mermaid
flowchart LR
  getSpikes[get_proper_global_spikes_df]
  coFilter["co_filter uses t_rel_seconds"]
  rankInit[RankOrderRastersDebugger.init]
  buildPlots[_build_internal_raster_plots]
  spots[NewSimpleRaster spots - already fixed]
  getSpikes --> coFilter --> rankInit --> buildPlots --> spots
```

## Constraint

Do **not** call `fixing_time_column()` before [`co_filter_epochs_and_spikes`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (~1585). That helper hardcodes `override_time_variable_name='t_rel_seconds'`, and fixup **drops** alias columns including `t_rel_seconds`.

## Change (single choke point)

In [`RankOrderRastersDebugger.init_rank_order_debugger`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\ContainerBased\RankOrderRastersDebugger.py) (~272), fix spikes **before** constructing `_obj` / calling `_build_internal_raster_plots`:

```python
global_spikes_df = global_spikes_df.spikes.fixing_time_column()
```

This mirrors [`Spike3DRasterWindowWidget`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\SpikeRasterWindows\Spike3DRasterWindowWidget.py) / [`SpikesDataframeWindow`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpikesDataframeWindow.py). It covers:

- `plot_full_paginated_decoded_epochs_window` → `_build_attached_raster_viewer`
- Other `RankOrderRastersDebugger.init_rank_order_debugger` callers (e.g. RankOrder display)

After fixup, `_build_internal_raster_plots`’s `adding_epochs_identity_column(...)` (no override) and `get_active_epoch_spikes_df()`’s `time_sliced` both use correct canonical `t`.

No notebook edits. No change to `co_filter_epochs_and_spikes` or `get_proper_global_spikes_df`.
