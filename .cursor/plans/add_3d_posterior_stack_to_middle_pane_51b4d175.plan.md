---
name: Add 3D posterior stack to middle pane
overview: Add a 3D volume visualization of per-time-bin 2D posteriors in the bottom half of the middle pane, splitting it into top (2D heatmap) and bottom (3D stack) sections.
todos:
  - id: "1"
    content: Split posterior view into 2D (top) and 3D (bottom) views in grid layout
    status: completed
  - id: "2"
    content: Update state dictionary to include both posterior views and 3D volume reference
    status: completed
  - id: "3"
    content: Modify 2D heatmap rendering to use top view with adjusted camera range
    status: completed
  - id: "4"
    content: Add 3D volume rendering in bottom view using scene.visuals.Volume
    status: completed
  - id: "5"
    content: Set up TurntableCamera for 3D interaction in bottom view
    status: completed
  - id: "6"
    content: Handle cleanup and updates when switching epochs
    status: completed
isProject: false
---

## Plan: Add 3D Stack of Per-Time-Bin Posteriors

### Overview

Split the middle pane (posterior view) into two sections:

- **Top half**: Existing 2D posterior heatmap (collapsed over time)
- **Bottom half**: New 3D volume visualization showing per-time-bin 2D posteriors stacked vertically

### Implementation Steps

1. **Modify Grid Layout** (`PredictiveDecodingComputations.py` ~line 6554-6559)
  - Replace single `posterior_view` with a nested grid structure
  - Create `posterior_2d_view` for top half (2D heatmap)
  - Create `posterior_3d_view` for bottom half (3D stack)
  - Both views should span the middle column
2. **Update State Dictionary** (~line 6575-6595)
  - Replace `'posterior_view'` with `'posterior_2d_view'` and `'posterior_3d_view'`
  - Add `'posterior_3d_volume'` to store the 3D volume visual reference
  - Update all references to use the appropriate view
3. **Modify `update_epoch_display` Function** (~line 6826-6842)
  - **2D Heatmap Rendering** (top half):
    - Keep existing 2D posterior rendering logic
    - Render to `posterior_2d_view` instead of `posterior_view`
    - Adjust camera range to only cover top half of space
  - **3D Volume Rendering** (bottom half):
    - Get `p_x_given_n` data (shape: `(n_x_bins, n_y_bins, n_time_bins)`)
    - Reshape/transpose to vispy format: `(n_time_bins, n_y_bins, n_x_bins)` for `(z, y, x)`
    - Create `scene.visuals.Volume` with:
      - `method='translucent'` or `'mip'` for rendering
      - `cmap='viridis'` to match 2D heatmap
      - Appropriate threshold/opacity settings
    - Set up `TurntableCamera` for 3D interaction
    - Position volume in bottom half coordinate space
4. **Camera Setup** (~line 6929-6950)
  - **2D View**: Use `PanZoomCamera` (existing) with range covering top half
  - **3D View**: Use `TurntableCamera` with:
    - `fov=45` or `0` (orthographic)
    - Initial `elevation=30`, `azimuth=-45` for good viewing angle
    - Range covering bottom half spatial extent + time dimension
5. **Coordinate System Considerations**
  - 2D view: Use existing `(x_min, x_max, y_min, y_max)` extent
  - 3D view: 
    - X, Y: Same spatial extent as 2D
    - Z: Time dimension, scaled appropriately (e.g., `0` to `n_time_bins * time_scale`)
  - Apply transforms to position volume correctly in 3D space
6. **Cleanup in `update_epoch_display**` (~line 6600-6645)
  - Clear existing `posterior_3d_volume` before creating new one
  - Update camera ranges for both views when epoch changes

### Key Technical Details

- **Data Format**: `p_x_given_n` is `(n_x_bins, n_y_bins, n_time_bins)`, vispy Volume expects `(z, y, x)`, so transpose to `(n_time_bins, n_y_bins, n_x_bins)`
- **Rendering Method**: Use `'translucent'` for semi-transparent stack or `'mip'` for maximum intensity projection
- **Spatial Alignment**: Ensure 3D volume X/Y coordinates align with 2D heatmap spatial extent
- **Time Scaling**: Scale Z-axis (time) appropriately so stack is visible but not too stretched

### Files to Modify

- `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)` - Main visualization function `render_predictive_decoding_with_vispy`

