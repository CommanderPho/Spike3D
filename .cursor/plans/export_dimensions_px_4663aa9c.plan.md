---
name: Export Dimensions Px
overview: Add `export_dimensions_px=(width, height)` to `TemplateDebugger.save_figure`, and make `export_pyqtgraph_plot` honor both width and height without the exporter’s linked aspect-ratio overwrite.
todos:
  - id: export-helper-both-dims
    content: In export_pyqtgraph_plot, when both width and height are provided set them with blockSignal so custom AR is preserved (int for PNG, float for SVG)
    status: completed
  - id: save-figure-export-dims
    content: Add export_dimensions_px to TemplateDebugger.save_figure and forward width/height to export_pyqtgraph_plot
    status: completed
isProject: false
---

# Pass `export_dimensions_px` through TemplateDebugger export

## Problem

[`TemplateDebugger.save_figure`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/TemplateDebugger.py) calls [`export_pyqtgraph_plot`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py) with no size control. PyQtGraph’s `ImageExporter` / `SVGExporter` expose `width` and `height`, but setting one auto-rescales the other to the **source** aspect ratio via `widthChanged` / `heightChanged`. Passing both through the current kwargs loop therefore cannot produce a custom aspect ratio like `(114.6818, 156.474)`.

## Approach

### 1. Teach `export_pyqtgraph_plot` to accept an absolute `(width, height)`

In [`ExportHelpers.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py) `export_pyqtgraph_plot`:

- Accept `width` / `height` via existing `**kwargs` (already forwarded to `exporter.parameters()`).
- When **both** `width` and `height` are present: set them with `blockSignal` so the linked AR callbacks do not overwrite each other:
  ```python
  exporter.parameters().param('width').setValue(w, blockSignal=exporter.widthChanged)
  exporter.parameters().param('height').setValue(h, blockSignal=exporter.heightChanged)
  ```
- When only one is present: keep current linked-AR behavior (set that one only).
- PNG: coerce to `int` (ImageExporter requires ints). SVG: allow floats (SVGExporter uses float params) — matches values like `114.6818`.

### 2. Add `export_dimensions_px` on `TemplateDebugger.save_figure`

In [`TemplateDebugger.save_figure`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/TemplateDebugger.py):

```python
def save_figure(self, shared_output_file_prefix=f'output/2025-07-21', export_format: str='.svg', export_merged: bool=True, export_dimensions_px: Optional[Tuple[float, float]]=None) -> Dict[str, Path]:
```

- When `export_dimensions_px` is set, unpack `(w, h)` and pass `width=w, height=h` into each `export_pyqtgraph_plot(...)` call.
- When `None`, preserve current behavior (PNG default width 4096; SVG uses scene size).
- Document usage:
  ```python
  template_debugger.save_figure(
      shared_output_file_prefix='output/2025-07-23',
      export_dimensions_px=(114.6818, 156.474),
  )
  ```

Merged SVG concatenation is unchanged: each panel is already sized, then `SVGHelpers.concatenate_svgs_horizontally` runs as today.

## Out of scope

- Resizing the on-screen dock widgets (export-only sizing)
- Track-boundary-lines unfinished display-fn wiring (separate work)
- Changing default export format
