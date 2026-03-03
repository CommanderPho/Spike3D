---
name: Integrate EditableColormap2DEditor timeline
overview: Replace the timeline's use of `PosteriorColormap2DEditorWidget` (the 2D preset-only editor) with `EditableColormap2DEditorWidget` (the one that includes gradient editors), and add a forwarding API on the editable widget so the renderer keeps using the same getCmap1/getCmap2/sigAdvancedColormapChanged interface.
todos: []
isProject: false
---

# Integrate EditableColormap2DEditorWidget into timeline

## Current state

- In [PhoOptimizedMultiEpochBatchRenderer.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PhoOptimizedMultiEpochBatchRenderer.py) (lines 906–917), when `use_advanced_3D_cmap` and `create_colormap_editor` are true, the code creates a **PosteriorColormap2DEditorWidget** (actually [Colormap2DEditorWidget](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PosteriorColormapEditorWidget.py) in the same module) with `preview_lut_builder=create_3d_lut_cmaps_interp` and `n_t_bins_preview=16`, stores it in `_out_dict['posterior_colormap_editor']`, and connects `editor.sigAdvancedColormapChanged` to a callback that re-applies posteriors using `editor.getCmap1()` and `editor.getCmap2()`.
- **EditableColormap2DEditorWidget** ([PosteriorColormapEditorWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PosteriorColormapEditorWidget.py) lines 403–469) is a `QMainWindow` that embeds a `Colormap2DEditorWidget` in `self.colorEditor` (created via `get_default_cmaps()`, which already uses `create_3d_lut_cmaps_interp` and `n_t_bins_preview=16`) and adds two `pg.GradientWidget`s for editing the 1D colormaps. It does **not** currently expose `getCmap1()`, `getCmap2()`, or `sigAdvancedColormapChanged` on itself; those exist only on `self.colorEditor`.

## Approach

1. **Add a forwarding API on EditableColormap2DEditorWidget** so the renderer can keep using the same interface (`editor.getCmap1()`, `editor.getCmap2()`, `editor.sigAdvancedColormapChanged.connect(...)`).
2. **Switch the timeline** to instantiate `EditableColormap2DEditorWidget()` instead of `PosteriorColormap2DEditorWidget(preview_lut_builder=..., n_t_bins_preview=16)` and update the import. No changes to the reapply callback or container logic.

Preview LUT behavior is unchanged: `get_default_cmaps()` already builds the inner editor with `create_3d_lut_cmaps_interp` and `n_t_bins_preview=16`, matching the current timeline call.

## Implementation

### 1. [PosteriorColormapEditorWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PosteriorColormapEditorWidget.py)

- **Forward signal**: On `EditableColormap2DEditorWidget`, define `sigAdvancedColormapChanged = QtCore.Signal(object, object)` and in `__init_`_ connect `self.colorEditor.sigAdvancedColormapChanged` to a slot that calls `self.sigAdvancedColormapChanged.emit(cmap1, cmap2)` (so the reapply callback can connect to the editable widget).
- **Forward getters**: Add `getCmap1(self)` and `getCmap2(self)` that return `self.colorEditor.getCmap1()` and `self.colorEditor.getCmap2()` respectively.
- **Export**: Add `EditableColormap2DEditorWidget` to `__all__` so the timeline can import it as part of the public API.

### 2. [PhoOptimizedMultiEpochBatchRenderer.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PhoOptimizedMultiEpochBatchRenderer.py)

- **Import**: Change the import from `PosteriorColormapEditorWidget` to include `EditableColormap2DEditorWidget` (e.g. add it to the existing import list that currently has `PosteriorColormapEditorWidget, PosteriorColormap2DEditorWidget`).
- **Creation**: Replace the block at lines 907–917 that creates and wires the advanced colormap editor:
  - Use `editor = EditableColormap2DEditorWidget()` (no constructor args; initial cmaps come from `get_default_cmaps` inside the widget).
  - Keep `_out_dict['posterior_colormap_editor'] = editor`, the same `_reapply_advanced_colormap` implementation (using `editor.getCmap1()`, `editor.getCmap2()`), and `editor.sigAdvancedColormapChanged.connect(_reapply_advanced_colormap)`.

Existing logic that adds the editor to `colormap_editor_container` (layout or widget) stays as-is; adding a `QMainWindow` to a layout embeds it. If a main-window frame appears in the timeline UI, a follow-up can switch to adding `editor.centralWidget()` instead of `editor`.

## Files to touch


| File                                     | Change                                                                                                                                                         |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PosteriorColormapEditorWidget.py`       | Add `sigAdvancedColormapChanged`, `getCmap1`, `getCmap2` on `EditableColormap2DEditorWidget`; add to `__all_`_.                                                |
| `PhoOptimizedMultiEpochBatchRenderer.py` | Import `EditableColormap2DEditorWidget`; use `EditableColormap2DEditorWidget()` instead of `PosteriorColormap2DEditorWidget(...)` in the advanced-cmap branch. |


No new files; no changes to `get_default_cmaps` or to the non-advanced branch (PosteriorColormapEditorWidget for 1D LUT/levels).