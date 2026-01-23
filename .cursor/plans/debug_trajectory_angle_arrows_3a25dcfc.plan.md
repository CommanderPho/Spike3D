---
name: Debug trajectory angle arrows
overview: Implement `enable_debug_plot_trajectory_aveerage_angle_arrows` mode to plot small direction arrows at the temporal center of each trajectory, pointing in the representative direction from `segment_Vp_deg`.
todos:
  - id: state-dict
    content: Add enable_debug_plot_trajectory_aveerage_angle_arrows and trajectory_debug_arrows to state dictionary
    status: completed
  - id: clear-arrows
    content: Add code to clear trajectory debug arrows on epoch change
    status: completed
  - id: past-arrows
    content: Add arrow drawing after past trajectory lines are created
    status: completed
  - id: future-arrows
    content: Add arrow drawing after future trajectory lines are created
    status: completed
  - id: docstring
    content: Update docstring to document the new parameter
    status: completed
isProject: false
---

# Implement Debug Trajectory Average Angle Arrows

## Summary

Add a debug visualization mode that draws an arrow at the temporal center of each past/future trajectory, pointing in the trajectory's representative direction (circular mean of `segment_Vp_deg`).

## Implementation Steps

### 1. Add Parameter to State Dictionary (~line 6738)

Add the new parameter to the state dictionary:

```python
'color_matches_by_matching_angle': color_matches_by_matching_angle,
'enable_debug_plot_trajectory_aveerage_angle_arrows': enable_debug_plot_trajectory_aveerage_angle_arrows,
'trajectory_debug_arrows': [],  # List of Arrow visuals for debug trajectory direction
```

### 2. Clear Debug Arrows on Epoch Change (~line 6850)

In the `update_epoch_display` function, add clearing of debug arrows alongside existing visual clearing:

```python
# Clear existing trajectory debug arrows
for arrow in state['trajectory_debug_arrows']:
    arrow.parent = None
state['trajectory_debug_arrows'] = []
```

### 3. Add Arrow Drawing After Past Trajectory Line (~line 7169)

After the past trajectory line is created and appended, add arrow drawing code:

```python
state['past_lines'].append(line)

# Draw debug arrow at trajectory temporal center if enabled
if state['enable_debug_plot_trajectory_aveerage_angle_arrows'] and 'segment_Vp_deg' in positions_df.columns:
    segment_angles = positions_df['segment_Vp_deg'].values
    valid_angles = segment_angles[~np.isnan(segment_angles)]
    if len(valid_angles) > 0:
        # Compute circular mean angle
        mean_angle_deg = np.degrees(np.arctan2(np.mean(np.sin(np.radians(valid_angles))), np.mean(np.cos(np.radians(valid_angles)))))
        mean_angle_rad = np.radians(mean_angle_deg)
        
        # Find temporal center position
        center_idx = len(x_valid) // 2
        x_center = x_valid[center_idx]
        y_center = y_valid[center_idx]
        
        # Arrow size based on data scale
        data_scale = max(x_max - x_min, y_max - y_min)
        arrow_head_size = data_scale * 0.04
        arrow_length = arrow_head_size * 0.5
        
        x_end = x_center + (arrow_length * np.cos(mean_angle_rad))
        y_end = y_center + (arrow_length * np.sin(mean_angle_rad))
        
        arrow = scene.visuals.Arrow(
            pos=np.array([[x_center, y_center], [x_end, y_end]]),
            arrows=np.array([[x_center, y_center, x_end, y_end]]),
            arrow_type='triangle_30', arrow_size=arrow_head_size,
            color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9),
            arrow_color=(base_rgb[0], base_rgb[1], base_rgb[2], 0.9),
            width=2.0, method='agg', parent=state['past_view'].scene)
        arrow.order = 5
        state['trajectory_debug_arrows'].append(arrow)
```

### 4. Add Arrow Drawing After Future Trajectory Line (~line 7730)

Apply the same pattern after future trajectory lines are created, using `state['future_view'].scene` as parent.

### 5. Update Docstring (~line 6555)

Add documentation for the new parameter:

```python
enable_debug_plot_trajectory_aveerage_angle_arrows: If True, draws small arrows at the temporal center of each 
    past/future trajectory indicating the trajectory's representative direction (circular mean of segment_Vp_deg). Default: True.
```

## Files to Modify

- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)
  - Line ~6738: Add to state dictionary
  - Line ~6850: Clear debug arrows on epoch change
  - Line ~7169: Add arrow after past trajectory line
  - Line ~7730: Add arrow after future trajectory line
  - Line ~6555: Update docstring

## Visual Result

When `enable_debug_plot_trajectory_aveerage_angle_arrows=True`:

- Each trajectory will have a small arrow at its temporal center
- The arrow points in the circular mean direction of `segment_Vp_deg`
- Arrow color matches the trajectory color
- Helps debug angle matching by showing the representative direction of each trajectory