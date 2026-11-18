---
name: Create Epoch3DSceneTimeBinViewer widget
overview: Create a new widget `Epoch3DSceneTimeBinViewer` that displays all time bins from the active epoch as adjacent 3D height map surfaces in a SceneWindow, controlled by a single epoch slider.
todos:
  - id: add_imports
    content: Add SceneWindow import to the file
    status: completed
  - id: create_class
    content: Create Epoch3DSceneTimeBinViewer class with __init__ method setting up SceneWindow and epoch slider
    status: completed
    dependencies:
      - add_imports
  - id: implement_item_creation
    content: Implement _create_time_bin_items method to create 2D scatter height maps for each time bin
    status: completed
    dependencies:
      - create_class
  - id: implement_cleanup
    content: Implement _clear_time_bin_items method to remove items from scene
    status: completed
    dependencies:
      - create_class
  - id: implement_epoch_handler
    content: Implement on_epoch_changed method to update scene when epoch slider changes
    status: completed
    dependencies:
      - implement_item_creation
      - implement_cleanup
  - id: add_metadata
    content: Add metadata decorator and docstring to the new class
    status: completed
    dependencies:
      - create_class
---

# Create Epoch3DSceneTimeBinViewer Widget

## Overview

Create a new widget `Epoch3DSceneTimeBinViewer` in the same file as `EpochTimeBinViewer`. This widget will:

- Use `SceneWindow` instead of `ScalarFieldView` for the 3D visualization
- Display all time bins from the active epoch as 3D height map surfaces arranged horizontally
- Only have one slider (epoch selection)
- Remove the 2D Plot2D view

## Implementation Details

### File to Modify

- `[pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py]`

### Key Changes

1. **New Class: `Epoch3DSceneTimeBinViewer`**

- Similar structure to `EpochTimeBinViewer` but simplified
- Uses `SceneWindow` instead of `ScalarFieldView`
- No 2D Plot2D view
- Only epoch slider (no time bin slider)

2. **3D Scene Setup**

- Import `SceneWindow` from `silx.gui.plot3d.SceneWindow`
- Create `SceneWindow` instance and get `SceneWidget` via `getSceneWidget()`
- Store list of time bin items for cleanup when epoch changes

3. **Time Bin Visualization**

- For each time bin in current epoch:
    - Extract 2D slice: `p_x_given_n[:, :, t_bin_idx] `(shape: `(n_x_bins, n_y_bins)`)
    - Create coordinate arrays from `xbin_centers` and `ybin_centers` (or indices if None)
    - Use `sceneWidget.add2DScatter(x, y, values)` where values come from the 2D slice
    - Enable height map: `item.setHeightMap(True)`
    - Set visualization mode: `item.setVisualization('solid')` or `'points'`
    - Position horizontally: `item.setTranslation(x_offset, 0, 0)` where x_offset increases for each bin
    - Set appropriate scale and colormap

4. **Layout**

- Single layout with SceneWindow taking full space
- Epoch slider at the bottom
- No splitter (no 2D view)

5. **Methods to Implement**

- `__init__`: Setup UI with SceneWindow and epoch slider
- `on_epoch_changed`: Clear existing items, create new time bin items for selected epoch
- `_create_time_bin_items`: Create and position all time bin height maps for current epoch
- `_clear_time_bin_items`: Remove all time bin items from scene

### Data Handling

- Each time bin is a 2D array `(n_x_bins, n_y_bins)`
- Convert to scatter data: create meshgrid from bin centers, flatten arrays
- Values array: flatten the 2D slice
- Position items with spacing based on data extent

### Positioning Logic

- Calculate spacing: `(x_max - x_min) * spacing_factor` between adjacent bins
- For bin `i`: `x_translation = i * spacing`
- Keep Y and Z at 0 (or small offset for visibility)

## Implementation Steps

1. Add imports: `SceneWindow` from `silx.gui.plot3d.SceneWindow`
2. Create `Epoch3DSceneTimeBinViewer` class after `EpochTimeBinViewer`
3. Implement `__init__` with SceneWindow and epoch slider only
4. Implement `_create_time_bin_items` to create height map surfaces
5. Implement `_clear_time_bin_items` for cleanup