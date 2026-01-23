---
name: Switch to Arrow instances for trajectory direction
overview: Replace the current Line-based arrow rendering with properly oriented scene.visuals.Arrow instances, starting with just beginning and end arrows.
todos: []
isProject: false
---

# Switch to scene.visuals.Arrow Instances for Trajectory Direction

## Current State

The code currently uses `scene.visuals.Line` with `connect='segments'` to draw arrow-like line segments. This approach doesn't provide proper arrowheads and orientation.

## Changes Required

### 1. Update State Management

- **File**: `PredictiveDecodingComputations.py` (around line 6650)
- Change `trajectory_arrows` from a single Line visual to a list of Arrow instances
- Store: `trajectory_arrows = []` (list of Arrow visuals)

### 2. Replace Arrow Rendering Logic

- **File**: `PredictiveDecodingComputations.py` (around lines 7058-7114)
- Remove the current Line-based arrow rendering code
- Implement new logic that:
  - Gets the first and last valid trajectory points with valid angles
  - Calculates arrow endpoints for start and end positions
  - Creates two `scene.visuals.Arrow` instances (one for start, one for end)
  - Uses `arrow_type='triangle_30'` or similar for proper arrowheads
  - Sets appropriate `arrow_size` (proportional to data scale)
  - Matches colors to trajectory opacity

### 3. Arrow Instance Management

- **File**: `PredictiveDecodingComputations.py` (around lines 7115-7148)
- Update cleanup logic to:
  - Remove old Arrow instances from parent and clear the list
  - Handle cases where arrows don't exist or need to be hidden
  - Properly clean up when trajectory data is missing

### 4. Arrow Creation Details

For each arrow (start and end):

- Calculate arrow endpoint: `end_point = start_point + arrow_length * [cos(angle), sin(angle)]`
- Create Arrow with:
  - `pos=[start_point, end_point]` (line body)
  - `arrows=[[start_x, start_y, end_x, end_y]]` (arrow head at endpoint)
  - `arrow_type='triangle_30'` (or 'triangle_60' for larger heads)
  - `arrow_size` proportional to data scale (e.g., 2-3% of scale)
  - `color` and `arrow_color` matching trajectory opacity
  - `width=1.5` for line thickness
  - `parent=state['posterior_2d_view'].scene`

### 5. Implementation Notes

- Arrow length should be ~1.5% of data scale (keep current size)
- Arrow colors: yellow (1.0, 0.9, 0.0) with opacity matching trajectory
- Only render if both start and end points have valid angles
- Handle angle conversion (degrees to radians) if needed
- Use fallback angle computation from trajectory direction if no angle column exists

## Code Structure

```python
# In update_epoch_display function, replace arrow rendering section:

# Get start and end points with angles
start_idx = 0
end_idx = len(x_valid) - 1

# Calculate arrow endpoints
arrow_length = data_scale * 0.015
start_angle_rad = arrow_angles_rad[start_idx]
end_angle_rad = arrow_angles_rad[end_idx]

start_arrow_end = [x_valid[start_idx] + arrow_length * np.cos(start_angle_rad),
                   y_valid[start_idx] + arrow_length * np.sin(start_angle_rad)]
end_arrow_end = [x_valid[end_idx] + arrow_length * np.cos(end_angle_rad),
                 y_valid[end_idx] + arrow_length * np.sin(end_angle_rad)]

# Create Arrow instances
# Clean up old arrows first
for arrow in state['trajectory_arrows']:
    arrow.parent = None
state['trajectory_arrows'].clear()

# Create start arrow
start_arrow = scene.visuals.Arrow(
    pos=[[x_valid[start_idx], y_valid[start_idx]], start_arrow_end],
    arrows=[[x_valid[start_idx], y_valid[start_idx], start_arrow_end[0], start_arrow_end[1]]],
    arrow_type='triangle_30',
    arrow_size=data_scale * 0.02,  # Arrow head size
    color=(1.0, 0.9, 0.0, arrow_alphas[start_idx]),
    arrow_color=(1.0, 0.9, 0.0, arrow_alphas[start_idx]),
    width=1.5,
    method='agg',
    parent=state['posterior_2d_view'].scene
)
start_arrow.order = 6
state['trajectory_arrows'].append(start_arrow)

# Create end arrow (similar)
```

## Testing Considerations

- Verify arrows point in correct direction at trajectory start/end
- Check arrow visibility and sizing
- Ensure proper cleanup when switching epochs
- Test with and without angle columns in position dataframe

