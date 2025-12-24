---
name: Add wireframe borders for time bin planes
overview: Add wireframe bounding boxes/outlines for each time bin plane in the 3D ScalarFieldView visualization to make it easy to see how many frames there are and which data belongs to which frame.
todos:
  - id: research_silx_api
    content: Research silx's plot3d API for adding wireframe boxes or 3D line primitives to SceneWidget
    status: completed
  - id: create_wireframe_function
    content: Create helper function to generate wireframe box coordinates for a single time bin plane
    status: completed
    dependencies:
      - research_silx_api
  - id: integrate_wireframes
    content: Integrate wireframe box generation into update_views() method to add boxes for all time bins
    status: completed
    dependencies:
      - create_wireframe_function
  - id: handle_coordinates
    content: Ensure wireframe boxes are positioned correctly using scale factors and bin centers
    status: completed
    dependencies:
      - create_wireframe_function
  - id: test_visualization
    content: Test the visualization to ensure wireframes are visible and correctly positioned
    status: completed
    dependencies:
      - integrate_wireframes
      - handle_coordinates
---

# A

dd Wireframe Borders for Time Bin Planes in 3D Visualization

## Overview

Enhance the `EpochTimeBinViewer` widget to draw wireframe borders/outlines for each time bin plane along the time axis in the 3D visualization. This will help visualize:

- The number of time bin frames
- Which data belongs to which frame
- The spatial extent of each time bin plane

## Implementation Plan

### 1. Research silx API for 3D primitives

- Explore `silx.gui.plot3d` module for wireframe/box primitives
- Check `SceneWidget` methods for adding custom 3D items
- Look for existing examples of adding lines/boxes to ScalarFieldView scenes

### 2. Create wireframe box generation function

- Function to create a wireframe box outline for a single time bin plane
- Calculate box dimensions based on:
- X/Y bin centers or data dimensions
- Time bin position (Z coordinate)
- Scale factors from `setScale()`
- Generate 12 edges of a rectangular box (4 edges per face, 3 faces visible)

### 3. Integrate wireframe boxes into `update_views()`

- In `update_views()` method, after setting the 3D data:
- Calculate number of time bins (`curr_n_time_bins`)
- For each time bin index (0 to n_time_bins-1):
    - Calculate Z position based on time bin index and scale
    - Create wireframe box at that Z position
    - Add box to scene widget
- Store references to wireframe items for potential cleanup/updates

### 4. Handle coordinate system

- Account for data format: `(time, y, x)` after transpose
- Use `xbin_centers` and `ybin_centers` if available for accurate positioning
- Apply scale factors from `setScale()` to position boxes correctly
- Ensure boxes align with actual data planes

### 5. Styling considerations

- Use semi-transparent color (e.g., `#FFFFFF40` or `#80808080`) so boxes don't obscure data
- Thin line width for wireframes
- Possibly different color for current time bin plane (highlighted)

## Files to Modify

- [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)
- Add method to create wireframe boxes
- Modify `update_views()` to add wireframe boxes for all time bins
- Store wireframe items for management

## Technical Details

### Wireframe Box Structure

Each time bin plane needs a rectangular outline at its Z position:

- 4 corners: (x_min, y_min, z), (x_max, y_min, z), (x_max, y_max, z), (x_min, y_max, z)
- 4 edges forming the rectangle perimeter

### Coordinate System

- Data is transposed to `(time, y, x)` format
- Time is the Z dimension
- X/Y dimensions come from `xbin_centers` and `ybin_centers` or data shape
- Scale factors from `setScale()` must be applied

### Implementation Approach

1. After `setData()` in `update_views()`, get scene widget
2. For each time bin, create wireframe box using silx's 3D primitive API
3. Position boxes at correct Z coordinates
4. Add all boxes to scene

## Potential Challenges

- **silx API discovery**: May need to explore silx documentation or source to find correct API for adding wireframe boxes
- **Performance**: Adding many wireframe boxes might impact rendering performance
- **Coordinate alignment**: Ensuring boxes align perfectly with data planes

## Alternative Approaches (if direct wireframe API unavailable)