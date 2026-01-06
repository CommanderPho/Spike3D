---
name: epoch3d-scatter-bbox-and-style
overview: Adjust Epoch3DSceneTimeBinViewer to control bounding box visibility and scatter rendering mode/scale for per-time-bin items in the SceneWindow-based 3D view.
todos:
  - id: locate-root-node
    content: Locate and understand the SceneWindow root "Data" node and how to toggle its bounding box visibility without affecting children.
    status: completed
  - id: configure-per-bin-items
    content: Update `_create_time_bin_items` to configure each per-time-bin scatter item with bounding box enabled, mode=points, square markers of size 5, and scale (1,1,1000).
    status: completed
    dependencies:
      - locate-root-node
  - id: epoch-refresh-behavior
    content: Ensure epoch changes and any re-creations of time-bin items preserve the new per-bin configuration and root bounding box state.
    status: completed
    dependencies:
      - configure-per-bin-items
---

## Adjust Epoch3DSceneTimeBinViewer scatter bounding boxes and style

### Scope

- **File**: [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)
- **Class**: `Epoch3DSceneTimeBinViewer`
- Only touch logic related to creation and configuration of the SceneWindow root data node and per-time-bin scatter/height-map items.

### Steps

- **Identify root "Data" node and disable its bounding box**
- Inspect how `SceneWindow` organizes its scene graph; use `self.scene_window.getSceneWidget()` / `self.scene_widget.getItems()` to locate the top-level data/root item that corresponds to the overall "Data" node in the sidebar.
- After scene initialization and/or after `_create_time_bin_items`, call the appropriate API on this root item (e.g., a `setBoundingBoxVisible(False)` / equivalent flag) so that its bounding box checkbox is off in the sidebar.
- **Ensure per-time-bin scatter items are directly accessible and bounding boxes enabled**
- In `_create_time_bin_items`, where each time-bin item is created via `self.scene_widget.add2DScatter(...)`, keep storing these objects in `self.time_bin_items`.
- Immediately after creating each item, explicitly enable its bounding box (e.g., `item.setBoundingBoxVisible(True)` or equivalent), ensuring their individual "Bounding box" sidebar checkboxes are on even if the root is off.
- **Change per-time-bin scatter visualization to point-cloud style**
- Replace the current height-map styling for each time-bin item:
    - Keep `setHeightMap(True)` only if required for Z-mapping; otherwise adjust to use a point-based Z if the API supports it while in point mode.
    - Change visualization mode from `'solid'` to `'points'` (or `setVisualization('points')`), matching the requested **Mode: Points**.
- Set marker shape and size on each item (e.g., `item.setSymbol('s')` / `setMarker('square')` and `item.setSymbolSize(5)` / `setMarkerSize(5)` depending on the silx API) to get **Marker: Square** and **Marker Size: 5**.
- **Apply anisotropic transform scale (z stretch) per scatter item**
- For each time-bin item, adjust its transform scale to **(1, 1, 1000)** by replacing the current `item.setScale(x_scale, y_scale, 1.0)` call with `item.setScale(1.0, 1.0, 1000.0)` (or a small wrapper respecting any future configuration).
- Keep the existing per-bin X translation (`item.setTranslation(x_translation, 0.0, 0.0)`) so horizontal layout is unchanged.
- **Wire up behavior on epoch change and scene refresh**
- Ensure `_clear_time_bin_items` and `_create_time_bin_items` continue to manage only the per-time-bin items, without re-enabling a root bounding box.