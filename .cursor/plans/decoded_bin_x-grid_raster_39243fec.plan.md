---
name: Decoded bin x-grid raster
overview: "Align the decoding-window spike raster’s vertical grid with posterior time bin edges by optionally configuring `Spike2DRaster` from Bapun notebook code: after each `update_zoomed_plot`, set the bottom axis major ticks to bin edges intersecting the visible `[min_t, max_t]` and enable `showGrid(x=True)`."
todos:
  - id: spike2d-grid-helper
    content: "Spike2DRaster: _apply_decoded_posterior_x_grid gated on applicationName BapunDecodingWindowRaster + decoder param; call from update_zoomed_plot"
    status: completed
  - id: pending-notebook-params
    content: "PendingNotebookCode: set _slave_params decoded_posterior_x_grid_one_step_decoder from first sync plotter"
    status: completed
isProject: false
---

# X-grid at decoded posterior bin edges (spike raster)

## Scope

Target **only** the **decoding-window dock** (`window_sync_raster` / “decoding window spikes” in `[PendingNotebookCode.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py)`): the small bottom strip whose x-range is updated via `update_zoomed_plot`.

**Naming:** the main timeline and this strip are both implemented as `[Spike2DRaster](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/SpikeRasterWidgets/Spike2DRaster.py)`. The main **controller** is typically `application_name='TimeSynchronizedPlotterControlSpikeRaster2D'` (what you call SpikeRaster2D). The strip uses `application_name='BapunDecodingWindowRaster'`. **Do not** enable decoded time-bin x-grid on the controller or on `[TimeSynchronizedPositionDecoderPlotter](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)` spatial heatmaps.

**Guard (required):** inside `_apply_decoded_posterior_x_grid`, return immediately unless `self.applicationName == 'BapunDecodingWindowRaster'` **and** `decoded_posterior_x_grid_one_step_decoder` is set. That keeps `update_zoomed_plot` safe for every other `Spike2DRaster` instance.

## Mechanism (PyQtGraph)

`[PlotItem.showGrid(x=True)](https://pyqtgraph.readthedocs.io)` draws vertical lines at **major** x ticks. So: compute bin **left/right edges** with the same rule as the decoder plotter (`[_decoder_bin_left_right_edges](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)`), collect edges from bins that **intersect** `[min_t, max_t]` (i.e. not `(right < min_t or left > max_t)`), `np.unique` + sort, build tick list `(edge, short_label)`, call `main_plot_widget.getAxis('bottom').setTicks([ticks])`, then `main_plot_widget.showGrid(x=True, y=False, alpha=...)`.

- Labels: keep readable (e.g. `g` format with modest precision) or empty strings if the strip is too crowded; prefer small labels only when edge count is below a threshold (e.g. 25) else `''` for tick text only (grid lines still show).

## Code changes

1. **[Spike2DRaster.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/SpikeRasterWidgets/Spike2DRaster.py)**
  - Import `_decoder_bin_left_right_edges` from `TimeSynchronizedPositionDecoderPlotter` (reuse; no duplicate edge math).
  - Add a small private helper, e.g. `_apply_decoded_posterior_x_grid(self, min_t, max_t)`, that:
    - **First** checks `self.applicationName == 'BapunDecodingWindowRaster'`; otherwise return (so the main SpikeRaster2D controller never gets bin grids from this path).
    - Reads optional `self.params.decoded_posterior_x_grid_one_step_decoder` (and optional `self.params.decoded_posterior_x_grid_show_labels` / `alpha` with sensible defaults).
    - If missing or `main_plot_widget` is None, return.
    - Builds edges intersecting the window; applies `setTicks` + `showGrid` as above.
  - Call this helper at the end of `[update_zoomed_plot](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/SpikeRasterWidgets/Spike2DRaster.py)` (after `setXRange` and `spikes_window.update_window_start_end`), so every scroll/resize refresh keeps the grid aligned.
2. **[PendingNotebookCode.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py)** (slave construction ~4671–4681)
  - After `_out_sync_plotters` is populated, set reference decoder to match posteriors, e.g. `_out_sync_plotters[included_filter_names[0]].active_one_step_decoder`.
  - Before `Spike2DRaster(..., application_name='BapunDecodingWindowRaster', ...)`, on `_slave_params`: `decoded_posterior_x_grid_one_step_decoder = ref_dec`. **Do not** set this key on the controlling widget’s `VisualizationParameters`.

## Verification

Open the Bapun combined window, scroll/zoom the timeline: decoding-row x-grid verticals align with posterior bin boundaries; count scales with window width.