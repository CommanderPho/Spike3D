---
name: Add 3D Axis Lines to Slices
overview: Add 3D axis line visuals with tick marks and labels around each time bin slice in the 3D stack view, keeping the TurntableCamera for rotation.
todos:
  - id: add-state-for-axes
    content: Add 'posterior_3d_axes' list to state dictionary for storing axis visuals
    status: completed
  - id: add-cleanup-code
    content: Add cleanup code to clear axis visuals when switching epochs
    status: completed
  - id: create-axis-helper
    content: Create helper function to draw X/Y axis lines with ticks at a given z-position
    status: completed
  - id: integrate-in-loop
    content: Call axis helper for each slice in the rendering loop, storing visuals
    status: completed
  - id: add-time-labels
    content: Add time bin index labels to each slice
    status: completed
isProject: false
---

# Add 3D Axis Lines to Time Bin Slices

## Overview

Add X and Y axis lines with tick marks and position labels around each 2D posterior slice in the 3D stack, positioned at each slice's z-coordinate. The axes will rotate with the 3D view.

## Current State

The 3D stack is rendered in `render_predictive_decoding_with_vispy` at lines 6856-6903:

- Each time bin is a `scene.visuals.Image` positioned at a different z-height
- Uses `MatrixTransform` for scaling and positioning
- View uses `TurntableCamera` for 3D rotation

## Implementation

### 1. Create Helper Function for 3D Axis Drawing

Add a helper function to draw axis lines with ticks for a single slice:

```python
def _draw_slice_axes(parent_scene, x_min, x_max, y_min, y_max, z_pos, xbin, ybin, num_ticks=5):
    """Draw X and Y axis lines with tick marks for a slice at z_pos."""
    visuals = []
    
    # X-axis line (bottom edge)
    x_axis_line = scene.visuals.Line(
        pos=np.array([[x_min, y_min, z_pos], [x_max, y_min, z_pos]]),
        color='white', width=2, parent=parent_scene
    )
    visuals.append(x_axis_line)
    
    # Y-axis line (left edge)
    y_axis_line = scene.visuals.Line(
        pos=np.array([[x_min, y_min, z_pos], [x_min, y_max, z_pos]]),
        color='white', width=2, parent=parent_scene
    )
    visuals.append(y_axis_line)
    
    # Add tick marks and labels...
    return visuals
```

### 2. Modify the 3D Slice Rendering Loop

In the existing loop (lines 6873-6898), after creating each slice image:

- Call the helper function to draw axes at `z_pos`
- Store axis visuals in state for cleanup
- Add tick marks at evenly spaced positions along X and Y
- Add small text labels showing position values (from `xbin`/`ybin`)

### 3. Add Tick Marks

For each axis, draw short perpendicular lines as tick marks:

```python
# X-axis ticks
tick_positions = np.linspace(x_min, x_max, num_ticks)
tick_length = (y_max - y_min) * 0.02  # 2% of Y extent
for tick_x in tick_positions:
    tick_line = scene.visuals.Line(
        pos=np.array([[tick_x, y_min, z_pos], [tick_x, y_min - tick_length, z_pos]]),
        color='white', width=1, parent=parent_scene
    )
```

### 4. Add Position Labels

Use `scene.visuals.Text` for tick labels:

```python
# Label at tick position
label = scene.visuals.Text(
    f'{tick_x:.1f}', pos=(tick_x, y_min - tick_length * 2, z_pos),
    color='white', font_size=8, anchor_x='center', anchor_y='top',
    parent=parent_scene
)
```

### 5. Update Cleanup Code

Add axis visuals to the state dictionary for proper cleanup when switching epochs:

```python
state['posterior_3d_axes'] = []  # Store axis line and text visuals

# In cleanup:
for axis_visual in state['posterior_3d_axes']:
    axis_visual.parent = None
state['posterior_3d_axes'].clear()
```

### 6. Optional: Add Time Bin Label

Add a text label showing the time bin index above or beside each slice:

```python
time_label = scene.visuals.Text(
    f't={t_idx}', pos=(x_min, y_max + label_offset, z_pos),
    color='white', font_size=10, parent=parent_scene
)
```

## Files to Modify

- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) - `render_predictive_decoding_with_vispy` function

## Key Considerations

- Tick density: Use `num_ticks` parameter (default ~5) to avoid cluttering small slices
- Label positioning: Labels need slight offset from axis to avoid overlap with tick marks
- Font size: Keep small (8-10pt) so labels don't dominate the view
- Only show subset of tick labels if there are many time bins to avoid visual clutter

