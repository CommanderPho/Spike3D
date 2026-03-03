---
name: Colormap editor dropdown and height
overview: "Fix two issues in the colormap editor: (1) set dropdown selections to match the initial cmap1/cmap2 (e.g. \"Alpha Red\" / \"Alpha Green\") by giving ColorMaps a name and passing initial cmaps into the editor; (2) ensure the 2D preview has enough height to render by setting a minimum height and optionally hiding non-essential plot chrome when space is tight."
todos: []
isProject: false
---

# Colormap Editor: Dropdown Defaults and 2D Preview Height

## 1. Dropdown heatmap selections defaulting to correct preset

**Cause:** `Colormap2DEditorWidget` already has `_sync_combos_from_cmaps()` and `_preset_name_for_cmap()`, but both rely on `getattr(cmap, 'name', None)`. The ColorMaps from `_make_alpha_red_cmap()` / `_make_alpha_green_cmap()` are created without a `name`, and `get_default_cmaps()` does not pass `initial_cmap1` / `initial_cmap2` into the editor, so the combos always resolve to the first preset ("Alpha Red").

**Changes in** [PosteriorColormapEditorWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\chunked_2d\PosteriorColormapEditorWidget.py):

- **Set `name` on preset ColorMaps:** In `_make_alpha_red_cmap()` and `_make_alpha_green_cmap()`, pass `name='Alpha Red'` and `name='Alpha Green'` into `pg.ColorMap(...)`. The external pyqtgraph `ColorMap.__init`__ accepts `name=''` and sets `self.name` (see [colormap.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\colormap.py) ~376, 393).
- **Use named cmaps and pass them into the editor in `get_default_cmaps()`:** When building `custom_cmap1` / `custom_cmap2` with `pg.ColorMap(pos, colors_red)` (and green), add the `name` argument (e.g. `name='Alpha Red'` / `name='Alpha Green'`). Then construct the editor with those as initial cmaps:  
`editor = Colormap2DEditorWidget(preview_lut_builder=..., n_t_bins_preview=16, initial_cmap1=custom_cmap1, initial_cmap2=custom_cmap2)`.

After this, `_sync_combos_from_cmaps()` will see the correct `.name` and set combo1/combo2 to "Alpha Red" and "Alpha Green" (or whatever presets match the passed-in cmaps).

---

## 2. Initial height of 2D track so the colormap renders

**Cause:** In `Colormap2DEditorWidget`, the 2D preview is a `pg.GraphicsLayoutWidget` (`_gl_widget`) added with `layout.addWidget(self._gl_widget, 1)` and no minimum height. When the window (or embedding container) is short, the plot gets almost no vertical space, so only axes are visible and the image does not render.

**Approach (two parts):**

- **Guarantee minimum height for the 2D preview:** After creating `_gl_widget` and adding the plot (around lines 254–260), set a minimum height on `_gl_widget`, e.g. `self._gl_widget.setMinimumHeight(120)` (or 100–150 px). This ensures the track always has enough room for the image.
- **Optional: hide non-essential plot parts when height is tight:** In `Colormap2DEditorWidget`, override `resizeEvent` (or use a similar mechanism) and, when the widget height is below a threshold (e.g. 100 px), hide title and axis labels to give more space to the image: e.g. `self._plot_item.setTitle(None)`, `self._plot_item.hideAxis('left')`, `self._plot_item.hideAxis('bottom')`, and restore them when height is above the threshold. PyQtGraph’s `PlotItem` supports `setTitle(None)` and `hideAxis(axis)` / `showAxis(axis, True)` (see [PlotItem.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\graphicsItems\PlotItem\PlotItem.py)).

Recommend implementing the minimum height first; add the responsive hide/show of title and axes if you want the preview to remain usable at very small heights.

---

## Summary of file and call changes


| Location                                                      | Change                                                                                                                                                                   |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `_make_alpha_red_cmap`                                        | Return `pg.ColorMap(pos, colors, name='Alpha Red')`.                                                                                                                     |
| `_make_alpha_green_cmap`                                      | Return `pg.ColorMap(pos, colors, name='Alpha Green')`.                                                                                                                   |
| `get_default_cmaps`                                           | Build `custom_cmap1`/`custom_cmap2` with `name='Alpha Red'`/`'Alpha Green'`; call `Colormap2DEditorWidget(..., initial_cmap1=custom_cmap1, initial_cmap2=custom_cmap2)`. |
| `Colormap2DEditorWidget.__init`__ (after adding `_gl_widget`) | Call `self._gl_widget.setMinimumHeight(120)` (or chosen value).                                                                                                          |
| `Colormap2DEditorWidget` (optional)                           | Override `resizeEvent` to hide/show plot title and left/bottom axes based on widget height.                                                                              |


No changes are required in `EditableColormap2DEditorWidget` for the dropdown fix; it uses the editor returned by `get_default_cmaps`, which will now receive the correct initial cmaps and display the correct combo selection once the above edits are in place.