---
name: Fix PDF Interval Labels
overview: Fix the pyqtgraph timeline PDF export so `IntervalRectsItem` in-rect labels that are visible in the widget are reliably present in the raster images embedded into the PDF.
todos:
  - id: interval-export-option
    content: Add an export-only label refresh option to `IntervalRectsItem.refresh_visible_labels`.
    status: completed
  - id: widget-export-refresh
    content: Update `PyqtgraphTimeSynchronizedWidget.export_as_img_arr` to force, process, and restore interval labels around `ImageExporter.export`.
    status: completed
  - id: validate-export
    content: Check lints and verify a one-page PDF export shows interval labels.
    status: completed
isProject: false
---

# Fix PDF Interval Labels

## Root Cause Hypothesis

The PDF export path already calls `PyqtgraphTimeSynchronizedWidget.export_as_img_arr(..., force_render_interval_labels=True)`, but the label refresh still goes through `IntervalRectsItem.refresh_visible_labels`, which requires each full text bounding box to fit inside the exported rect pixels. PDF track allocation can make interval tracks much thinner than the live widget, so every label can be culled. The export also refreshes labels immediately before `ImageExporter.export(...)` without explicitly forcing Qt text layout/transform updates after the temporary resize.

Relevant files:

- [h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\PyQtPlots\TimeSynchronizedPlotters\PyqtgraphTimeSynchronizedWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\PyQtPlots\TimeSynchronizedPlotters\PyqtgraphTimeSynchronizedWidget.py)
- [h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\GraphicsObjects\IntervalRectsItem.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\GraphicsObjects\IntervalRectsItem.py)
- [h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Mixins\ExportHelpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Mixins\ExportHelpers.py)

## Implementation Plan

1. Add an export-only option to `IntervalRectsItem.refresh_visible_labels(...)`, defaulting off, that keeps the existing on-screen culling behavior but can bypass the final “text must fit inside rect” test during export.

2. In `IntervalRectsItem._assign_label_item(...)`, after setting text/position/visibility, force the `CustomRectBoundedTextItem` transform/update when available so labels are ready for a synchronous scene render.

3. Update `PyqtgraphTimeSynchronizedWidget.export_as_img_arr(...)` so its local interval-label refresh helper:
   - Collects `IntervalRectsItem` instances from the active `PlotItem`, preferably recursively or through `pi.items` plus child traversal.
   - Temporarily saves label culling settings and restores them in `finally`.
   - Calls the new export refresh option when `force_render_interval_labels=True`.
   - Calls `QApplication.processEvents()` after range/geometry changes and after label refresh, before `ImageExporter.export(toBytes=True)`.
   - Optionally prints active label counts under existing `debug_print`.

4. Leave `FigureToImageHelpers.export_wrapped_tracks_to_paged_df(...)` behavior intact unless validation shows track height allocation is still starving labels. It already passes `force_render_interval_labels=True` at the call site.

## Validation

Run a focused export with `debug_max_num_pages=1` and visually inspect the generated PDF for interval labels. Then use `ReadLints` on the edited files and, if practical in the local session, run a small export smoke test using the existing notebook/workflow that produced the failing PDF.