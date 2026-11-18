---
name: 2D Posterior Colormap Editor
overview: Add a 2D variant of PosteriorColormapEditorWidget that displays and edits the advanced_3D_cmap format (value x time -> RGBA) with a 2D gradient preview and two 1D colormap selectors, and wire it into the renderer so changing cmaps re-applies posteriors without re-decoding.
todos: []
isProject: false
---

# 2D Posterior Colormap Editor (advanced_3D_cmap)

## Context

- **advanced_3D_cmap format**: A 2D LUT `(v_bins, n_t_bins, 4)` mapping (value index, time bin index) -> RGBA. Built by interpolating two 1D colormaps (e.g. red-alpha, green-alpha) across time in [PhoOptimizedMultiEpochBatchRenderer](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PhoOptimizedMultiEpochBatchRenderer.py) via `create_3d_lut_cmaps_interp` (lines 135-185). The renderer currently builds `custom_cmap1` / `custom_cmap2` inline (lines 575-589) and does not accept them from outside.
- **1D editor**: [PosteriorColormapEditorWidget](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PosteriorColormapEditorWidget.py) provides ComboBox + ColorBarItem + optional GradientEditorItem; it only applies when `use_advanced_3D_cmap=False` and is never shown when advanced mode is on (see renderer ~1029-1056).

## Design

- **New widget**: `PosteriorColormap2DEditorWidget` — shows a **2D gradient preview** (value axis vs time axis, color = RGBA) and lets the user choose **two 1D colormaps** (cmap1 = “early time”, cmap2 = “late time”). No built-in 2D gradient editor in pyqtgraph; the 2D view is a **read-only preview** of the LUT rendered as an image.
- **Apply flow**: Widget emits `sigAdvancedColormapChanged(cmap1, cmap2)`. The renderer (or caller) re-calls `plot_decoded_posteriors_for_frames(..., extant_posterior_image_items=items, custom_cmap1=..., custom_cmap2=...)` so existing ImageItems get updated RGBA without re-decoding.
- **Avoid circular import**: The widget must not import the renderer. The renderer will pass a **LUT builder callable** into the widget for generating the preview (e.g. `create_3d_lut_cmaps_interp`), so the widget stays independent.

## Implementation

### 1. Renderer: accept external cmap1/cmap2 in `plot_decoded_posteriors_for_frames`

**File**: [PhoOptimizedMultiEpochBatchRenderer.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PhoOptimizedMultiEpochBatchRenderer.py)

- Pop from `kwargs`: `custom_cmap1`, `custom_cmap2` (optional, any type: `str` or `pg.ColorMap`).
- In the `if use_advanced_3D_cmap:` block (around 575): if `custom_cmap1` and `custom_cmap2` are provided, use them; otherwise keep building the current inline red/green alpha `pg.ColorMap` instances (unchanged behavior when not provided).

This allows the 2D editor (or any caller) to drive the two colormaps from outside.

### 2. New widget: `PosteriorColormap2DEditorWidget`

**File**: [PosteriorColormapEditorWidget.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PosteriorColormapEditorWidget.py) (add class and exports in same file).

- **Constructor**: e.g. `__init__(self, preview_lut_builder=None, n_t_bins_preview=16, initial_cmap1=None, initial_cmap2=None, parent=None)`.
  - `preview_lut_builder`: callable `(n_t_bins, cmap1, cmap2) -> np.ndarray` of shape `(v_bins, n_t_bins, 4)` uint8. Required for preview; when used from the renderer, the renderer will pass `create_3d_lut_cmaps_interp` (or a thin wrapper).
  - Defaults for the two cmaps: match current renderer behavior — build the same “Alpha Red” and “Alpha Green” `pg.ColorMap` (pos `[0,1]`, alpha 100->255) so the 2D preview matches what users see by default.
- **UI layout** (minimal):
  - **Row 1**: Two combo boxes — “Cmap 1 (early t)” and “Cmap 2 (late t)”. Populate with a small set of presets: at least “Alpha Red”, “Alpha Green” (and optionally “Reds”, “Greens” from matplotlib). Store either a name or a `pg.ColorMap` per preset; when a preset is selected, resolve to `pg.ColorMap` and update preview + emit signal.
  - **Row 2**: 2D preview — a `pg.GraphicsLayoutWidget` with a single `PlotItem` and an `ImageItem`. Image = preview LUT: shape `(v_bins, n_t_bins, 4)`. Use `preview_lut_builder(n_t_bins_preview, cmap1, cmap2)` to get the array. Display it so that **vertical axis = value index**, **horizontal axis = time index** (no extra colormap on ImageItem; the array is already RGBA). Axis labels optional but helpful (e.g. “value”, “time bin”).
  - Optional: spinbox for `n_t_bins_preview` (e.g. 8–64) so users can change preview resolution.
- **Signals**: `sigAdvancedColormapChanged(cmap1, cmap2)` — emit whenever the user changes either combo (and on initial sync).
- **Accessors**: `getCmap1()`, `getCmap2()` returning the current `pg.ColorMap` instances; `setCmap1(cmap)`, `setCmap2(cmap)` to set programmatically and update preview/combo without re-emitting if needed.
- **Preview update**: Whenever cmap1 or cmap2 changes, call `preview_lut_builder(n_t_bins_preview, self._cmap1, self._cmap2)`, set the result on the ImageItem (with correct axis order so value = rows, time = cols). If `preview_lut_builder` is None, either hide the preview or show a placeholder.
- **Presets**: Implement “Alpha Red” and “Alpha Green” in the same way as the renderer (e.g. helper in this file): `pos = np.array([0.0, 1.0])`, `colors_red = [[255,0,0,100], [255,0,0,255]]`, same for green. Add to `__all`__ and docstring.

### 3. Renderer: create 2D editor when `use_advanced_3D_cmap` is True and connect re-apply

**File**: [PhoOptimizedMultiEpochBatchRenderer.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PhoOptimizedMultiEpochBatchRenderer.py)

- Import: `PosteriorColormap2DEditorWidget` from the same module as `PosteriorColormapEditorWidget`.
- In `plot_all`, where the posterior colormap editor is created (around 1043-1065):
  - When `use_advanced_3D_cmap` is True (or when we want to show the 2D editor for advanced mode), create `PosteriorColormap2DEditorWidget` with `preview_lut_builder=create_3d_lut_cmaps_interp` and optional initial cmap1/cmap2 (e.g. the same inline red/green defaults so behavior matches).
  - Add the 2D editor to `colormap_editor_container` instead of (or in addition to) the 1D `PosteriorColormapEditorWidget` when in advanced mode. Decision: when `use_advanced_3D_cmap` is True, create and show only the 2D editor; when False, keep creating the existing 1D editor. That way one editor is shown and it matches the mode.
- Connect `PosteriorColormap2DEditorWidget.sigAdvancedColormapChanged` to a slot that re-calls `plot_decoded_posteriors_for_frames` with:
  - Same args as before: `a_decoded_subdivided_epochs_result`, `subdivided_epochs_df`, `maze_bounds_t`, `track_plot_item`, and `**extant_posterior_image_items`** = the list of posterior ImageItems already created,
  - Plus `custom_cmap1=editor.getCmap1()`, `custom_cmap2=editor.getCmap2()`.
- Store the references needed for this callback (decoded result, dataframe, maze_bounds_t, track_plot_item, posterior_image_items) in the scope where the connection is made (e.g. in `plot_all` so the slot is a closure over these variables).

### 4. ImageItem for 2D RGBA preview

- The LUT from `create_3d_lut_cmaps_interp` is `(v_bins, n_t_bins, 4)` uint8. For pyqtgraph `ImageItem`: set this as the image with `autoLevels=False` and no separate LUT so the RGBA is shown as-is. Confirm in the vendored ImageItem whether it supports (Nx, Ny, 4) directly; if it expects (Nx, Ny) + LUT, then we may need to use a different approach (e.g. convert to QImage or use a small wrapper that draws the RGBA array). If the codebase or pyqtgraph docs show RGBA image display, use that; otherwise the plan is to set the raw RGBA array and rely on ImageItem’s handling of 3/4-channel images.

### 5. Exports and **all**

- In [PosteriorColormapEditorWidget.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/chunked_2d/PosteriorColormapEditorWidget.py): add `PosteriorColormap2DEditorWidget` to `__all`__. Keep `_get_cmap` and any new preset helpers internal or exported as needed.

## Summary


| Area                                          | Change                                                                                                                                                                                              |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Renderer `plot_decoded_posteriors_for_frames` | Pop and use `custom_cmap1` / `custom_cmap2` from kwargs when provided and `use_advanced_3D_cmap` is True.                                                                                           |
| New widget                                    | `PosteriorColormap2DEditorWidget` in same file: two combo presets, 2D LUT preview via callable, `sigAdvancedColormapChanged`, get/set cmap1/cmap2.                                                  |
| Renderer `plot_all`                           | When advanced mode, create 2D editor with `preview_lut_builder=create_3d_lut_cmaps_interp`, connect signal to re-call `plot_decoded_posteriors_for_frames` with existing image items and new cmaps. |


No new package or LUT util file is required if the renderer passes the LUT builder into the widget. The 2D gradient is represented as a single read-only image (value x time -> color); editing is via the two 1D colormap presets.