---
name: Convert Scatter2D to Mesh
overview: Convert the Scatter2D items used for height map visualization in Epoch3DSceneTimeBinViewer to Mesh items, maintaining the same visual appearance and functionality.
todos:
  - id: add-mesh-helper
    content: Add _create_mesh_from_grid() helper method to generate vertices and faces from 2D grid data
    status: pending
  - id: update-create-items
    content: Update _create_time_bin_items() to use Mesh items instead of add2DScatter
    status: pending
    dependencies:
      - add-mesh-helper
  - id: update-configure-item
    content: Update _configure_time_bin_item() to work with Mesh items (remove setHeightMap, setPointSize)
    status: pending
---

# Convert Scatter2D Items to Mesh Items

## Overview

Replace the `add2DScatter()` calls in `Epoch3DSceneTimeBinViewer` with `Mesh` items from silx. The current implementation uses Scatter2D with height maps enabled to create 3D surfaces. Mesh items will provide a more explicit surface representation using vertices and triangular faces.

## Changes Required

### 1. Add Helper Function for Mesh Generation

Create a new helper method `_create_mesh_from_grid()` in `Epoch3DSceneTimeBinViewer` that:

- Takes X, Y meshgrids and values array (2D slice)
- Generates 3D vertices: (x, y, z) where z = values
- Creates triangular faces connecting adjacent grid points
- Returns vertices array and faces (indices) array

### 2. Update `_create_time_bin_items()` Method

In [`1903:1968:pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py):

- Replace `self.scene_widget.add2DScatter(x_flat, y_flat, values_flat)` with Mesh creation
- Use the helper function to generate vertices and faces from the 2D grid
- Create `plot3d_items.Mesh()` and call `setData()` with vertices, faces, and values
- Add the mesh item to the scene using `self.scene_widget.addItem()`

### 3. Update `_configure_time_bin_item()` Method

In [`1840:1901:pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py):

- Remove `item.setHeightMap(True)` call (not applicable to Mesh)
- Keep colormap configuration (Mesh supports colormaps)
- Keep translation, scale, and visualization settings
- Remove `setPointSize()` call (not applicable to Mesh)
- Ensure `setVisualization('solid')` works with Mesh

### 4. Handle Edge Cases

- Ensure proper handling of NaN values in the data
- Maintain the same translation and positioning logic
- Preserve the colormap range configuration per time bin

## Implementation Details

The mesh generation will:

- Create vertices from the (X, Y) meshgrid with z-coordinates from the values
- Generate triangular faces using a standard grid triangulation pattern:
- For each grid cell (i, j), create two triangles:
- Triangle 1: (i,j), (i+1,j), (i,j+1)
- Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
- Handle grid boundaries correctly

## Files to Modify

- `pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`
- Add `_create_mesh_from_grid()` helper method
- Modify `_create_time_bin_items()` method
- Modify `_configure_time_bin_item()` method