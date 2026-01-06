---
name: silx-epoch3d-peak-contours
overview: Add Silx 3D contour overlays for posterior peak-prominence results that track the epoch slider, analogous to the Napari peak-contour layer.
todos:
  - id: analyze-napari-contour-api
    content: Review `add_peak_contours_layer` and `PosteriorPeaksPeakProminence2dResult` to understand how contours are indexed and how world coordinates are mapped to viewer coordinates.
    status: completed
  - id: extend-epoch3d-state
    content: Augment `Epoch3DSceneTimeBinViewer` with fields to hold the peak-prominence result and the list of Silx contour items.
    status: completed
    dependencies:
      - analyze-napari-contour-api
  - id: implement-contour-extraction
    content: Implement helper(s) on `Epoch3DSceneTimeBinViewer` to extract per-epoch, per-time-bin contour vertex arrays from the peak-prominence result in a coordinate frame compatible with the existing height maps.
    status: completed
    dependencies:
      - extend-epoch3d-state
  - id: implement-silx-contour-drawing
    content: Implement helpers to create and add Silx 3D line items for each contour, positioned correctly relative to the existing time-bin height maps, and to clear them when needed.
    status: completed
    dependencies:
      - implement-contour-extraction
  - id: wire-into-epoch-slider
    content: Update `Epoch3DSceneTimeBinViewer.on_epoch_changed` and the new public API so contour overlays are created/updated when the epoch slider moves, without disturbing existing behavior.
    status: completed
    dependencies:
      - implement-silx-contour-drawing
---

## Add Silx peak-contour overlays to `Epoch3DSceneTimeBinViewer`

### 1. Understand existing APIs and data structures

- **Review Napari implementation**: Study `add_peak_contours_layer` in [`pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`](pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py) to see how it:
- Stores the `PosteriorPeaksPeakProminence2dResult` object.
- Indexes contours by `(decoded_epoch_index, decoded_epoch_time_bin_index)`.
- Converts `matplotlib.path.Path.vertices` in world coordinates (x/y bin centers) into viewer pixel coordinates.
- **Review Silx 3D viewer behavior** in [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py):
- How `Epoch3DSceneTimeBinViewer._create_time_bin_items` lays out each time-bin heatmap as a 3D height map via `scene_widget.add2DScatter(...).setHeightMap(True)`.
- How `x_translation` and `bin_spacing` encode the time-bin index along the global X axis.
- Where `xbin_centers`/`ybin_centers` are used to determine spatial coordinates.

### 2. Design the Silx contour-overlay API and state

- **New public method** on `Epoch3DSceneTimeBinViewer`:
- `def add_peak_contours_overlays(self, peak_prominence_result: "PosteriorPeaksPeakProminence2dResult", edge_color: str = '#ffaaff', line_width: float = 1.0) -> None:`
- Mirrors the Napari method conceptually but renders into the Silx `SceneWindow` instead of a Napari shapes layer.
- **Internal state additions** to `Epoch3DSceneTimeBinViewer`:
- `self.peak_prominence_result` to store the passed result object.
- `self.peak_contour_items: List[Any] `to track per-epoch contour graphics added to `self.scene_widget` so they can be cleared on epoch change.
- **Coordinate mapping strategy**:
- Assume peak contours are returned in the same world coordinate system as the decoder (`xbin_centers` / `ybin_centers`).
- Reuse the same `x_min/x_max/y_min/y_max/bin_spacing` logic as `_create_time_bin_items` to convert each time-bin’s contour vertices into Silx 3D coordinates `(X, Y, Z)` that align with the corresponding height-map tile.

### 3. Implement contour extraction helpers (Silx-agnostic)

- **Helper function** within `Epoch3DSceneTimeBinViewer` analogous to the Napari inner helper:
- `def _extract_contours_for_epoch_timebin(self, epoch_idx: int, t_bin_idx: int) -> List[NDArray]:`
- Responsibilities:
    - Look up the peak entry for `(epoch_idx, t_bin_idx)` from `peak_prominence_result.results` (or equivalent structure; mirror the access pattern used in the Napari mixin).
    - Iterate its `level_slices` or similar field and gather each `slice_info['contour'] `(a `matplotlib.path.Path`).
    - Convert `Path.vertices` (world X/Y) to **viewer X/Y** consistent with the Silx scene:
    - Use the same world units as your height maps (xbin/ybin), so in many cases no rescaling is needed aside from time-bin translation.
    - Optionally, if height maps assume indices rather than world units, linearly map world X/Y to [0, n_bins) using the same interpolation scheme as the Napari helper (reusing or inlining its `world_to_pixel_coord` logic).
    - Return a list of `(n_points, 2)` arrays representing contours for that epoch/time-bin in “local” X/Y coordinates (before time-bin translation).

### 4. Implement Silx-specific contour drawing

- **Create a Silx drawing helper** on `Epoch3DSceneTimeBinViewer`:
- `def _add_contours_for_current_epoch(self, edge_color: str, line_width: float) -> None:`
- Steps:
    - Compute epoch-local posterior shape and bin metadata similarly to `_create_time_bin_items` to know `n_time_bins`, `x_min/x_max/y_min/y_max`, and `bin_spacing`.
    - For each `t_bin_idx` in `range(n_time_bins)`:
    - Call `_extract_contours_for_epoch_timebin(self.curr_epoch_idx, t_bin_idx)` to get contour vertex arrays.
    - For each contour vertex array:
        - Compute the **global X** by adding the same `x_translation = t_bin_idx * bin_spacing` used for the corresponding height-map tile.
        - Use `z` coordinate equal to either:
        - The underlying height map’s average/posterior value at those positions (if you want true 3D contours), or
        - A small constant offset above the base plane (e.g., `z_offset = 0.01`) for “floating” contour borders.
        - Feed `(x, y, z)` plus dummy `values` into a `plot3d_items.Scatter3D()` or equivalent Silx 3D item with `setVisualization('lines')` and `setLineWidth(line_width)`, `setColor(rgba(edge_color))`.
        - Add the item to `self.scene_widget` and append it to `self.peak_contour_items` for later cleanup.
- **Cleanup helper**:
- `def _clear_peak_contour_items(self) -> None:` that removes all items in `self.peak_contour_items` from `scene_widget` (if they are still present) and clears the list.

### 5. Wire overlays into epoch-slider updates

- **Extend `__init__`** of `Epoch3DSceneTimeBinViewer`:
- Initialize `self.peak_prominence_result = None` and `self.peak_contour_items = []`.
- **Extend `add_peak_contours_overlays`**:
- Store the `peak_prominence_result`.
- Immediately generate contour overlays for the current `self.curr_epoch_idx` by calling `_clear_peak_contour_items()` then `_add_contours_for_current_epoch(...)`.
- **Modify `on_epoch_changed`** in `Epoch3DSceneTimeBinViewer`:
- After the existing calls that rebuild time-bin items (`_clear_time_bin_items`, `_create_time_bin_items`, `_update_text_label_positions`), add logic:
    - If `self.peak_prominence_result is not None`, call `_clear_peak_contour_items()` and `_add_contours_for_current_epoch(...)` so the contours match the newly selected epoch.
- Keep any table/point-like-mode logic intact.

### 6. Documentation and minimal validation

- **Docstrings and tags**:
- Add a `@function_attributes`-style decorator and clear docstring to `add_peak_contours_overlays` describing that it mirrors the Napari `add_peak_contours_layer` but targets Silx 3D scenes and is driven by the epoch slider.
- **Basic consistency checks**: