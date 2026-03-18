---
name: 3D Posterior Contour Rendering
overview: Add methods to `Volumentric2DTimeSeriesPlotter` that render posterior probability contours (one per time bin) as 3D line visuals, each lifted to the correct z-height corresponding to that time bin's time value.
todos:
  - id: add-fields
    content: Add posterior_contours_by_key and posterior_contours_counter fields to Volumentric2DTimeSeriesPlotter
    status: completed
  - id: add-private-helpers
    content: Add _next_posterior_contour_key and _build_posterior_contours_3d private methods
    status: completed
  - id: add-public-api
    content: Add add_posterior_contours, get/list/set_visibility/remove/clear public API methods
    status: completed
isProject: false
---

# 3D Posterior Contour Rendering in `Volumentric2DTimeSeriesPlotter`

## How the 2D pipeline works (in `predictive_decoding_central_view.py`)

```
per_t_bin_mask: NDArray[n_xbins, n_ybins, n_tbins]
  → masks = [per_t_bin_mask[:, :, t_idx].T  for t_idx in range(n_tbins)]
  → contours_from_masks(masks, x_bounds, y_bounds, colors, level=0.5, return_per_mask=True)
      → List[List[ContourItem]]  where ContourItem = (pos: NDArray[N,2], rgba: Tuple)
  → create_contour_line_visuals(flat_items, parent=view.scene, ...)
      → vz.Line(pos=pos_2d, ...) and vz.Polygon fills
```

## 3D adaptation

The only change needed is to lift each contour's `pos` from `(N, 2)` → `(N, 3)` by appending a constant z-column:

```
z_val = (t_bin_center(t_idx) - self.t_min) * self.z_scale
pos_3d = np.hstack([pos_2d, np.full((len(pos_2d), 1), z_val, dtype=np.float32)])
```

`vz.Line` already accepts 3D positions. `vz.Polygon` is 2D-only so fill is skipped in 3D mode.

## File to change

`[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` — add to the `Volumentric2DTimeSeriesPlotter` class (already imports `ContourItem`, `contours_from_masks`, `create_contour_line_visuals` at line 83).

## Changes

**1. New fields** (add after `coordinate_axes_labels` / `debug_crosshair_lines` block, ~line 1757):

```python
posterior_contours_by_key: Dict[str, List[Any]] = field(default=Factory(dict))
posterior_contours_counter: int = field(default=0)
```

**2. New private helpers** (add after `_next_decoded_posterior_key`):

- `_next_posterior_contour_key() -> str` — auto-generates unique string keys (same pattern as `_next_decoded_posterior_key`)
- `_build_posterior_contours_3d(per_t_bin_mask, t_bin_edges_for_contours=None, line_width=2.0, contour_alpha=0.7, level=0.5) -> List[Any]` — core builder:
  - Reads `x_min/x_max` from `self.xbin`, `y_min/y_max` from `self.ybin`
  - Iterates over `n_tbins = per_t_bin_mask.shape[2]`
  - For each `t_idx`: computes `t_bin_center` (via provided edges or linear fallback on `self.t_bin_edges`), then `z_val = (t_bin_center - self.t_min) * self.z_scale`
  - Extracts `mask = per_t_bin_mask[:, :, t_idx].T`
  - Calls `contours_from_masks([mask], x_bounds, y_bounds, colors=[color_for_t_idx], level=level)`
  - For each `ContourItem` `(pos_2d, rgba)`: builds `pos_3d = np.hstack([pos_2d, np.full((len(pos_2d),1), z_val)])` and creates `vz.Line(pos=pos_3d, color=rgba, width=line_width, parent=self.view.scene)`
  - Returns flat `List` of all created line visuals

**3. New public API methods** (grouped block after `clear_decoded_posteriors`):

- `add_posterior_contours(per_t_bin_mask, t_bin_edges=None, unique_identifier=None, visible=True, replace_if_exists=True) -> str`
- `get_posterior_contours(unique_identifier) -> Optional[List[Any]]`
- `list_posterior_contour_keys() -> List[str]`
- `set_posterior_contours_visibility(unique_identifier, is_visible) -> bool`
- `remove_posterior_contours(unique_identifier) -> bool`
- `clear_posterior_contours()`

Colors per t-bin are generated using the same hue-cycling strategy as `_time_bin_colors` in `predictive_decoding_central_view.py` (inline in the builder, no new import needed — just `colorsys` which is already imported in the file).