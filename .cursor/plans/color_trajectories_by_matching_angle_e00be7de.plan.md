---
name: Color Trajectories by Matching Angle
overview: Implement the `color_matches_by_matching_angle` parameter to color past/future trajectories based on which time bin's angle they matched with, using the corresponding time bin color.
todos:
  - id: state-dict
    content: Add color_matches_by_matching_angle to the state dictionary
    status: completed
  - id: move-colors
    content: Move time_bin_colors generation earlier (after p_x_given_n) so it is available for trajectory coloring
    status: completed
  - id: past-traj-color
    content: Modify past trajectory coloring to use matched time bin color when color_matches_by_matching_angle is True
    status: completed
  - id: future-traj-color
    content: Modify future trajectory coloring to use matched time bin color when color_matches_by_matching_angle is True
    status: completed
  - id: docstring
    content: Update the function docstring to document the new parameter
    status: completed
isProject: false
---

# Color Trajectories by Matching Angle

## Summary

Modify the `render_predictive_decoding_with_vispy` function to use the new `color_matches_by_matching_angle` parameter. When enabled, trajectories that have a valid `centroid_pos_traj_matching_angle_idx` value (>= 0) will be colored using the corresponding time bin's color instead of the default red (past) or cyan (future).

## Key Data Relationships

- `positions_df['centroid_pos_traj_matching_angle_idx']`: Column containing the index of the matching centroid segment (-1 if no match, >= 0 if matched)
- `time_bin_colors`: HSV colormap array with distinct colors for each time bin (currently generated at line ~7185)
- Past trajectories: default red (hue=0.0), rendered at lines 7019-7133
- Future trajectories: default cyan (hue=0.5), rendered at lines 7568-7682

## Implementation Details

### 1. Add `color_matches_by_matching_angle` to State Dict (~line 6735)

Add the new parameter to the state dictionary:

```python
'require_angle_match': require_angle_match,
'color_matches_by_matching_angle': color_matches_by_matching_angle,
```

### 2. Move Time Bin Color Generation Earlier (~line 6900)

The `time_bin_colors` array is currently generated inside the centroid rendering block (lines 7184-7190), which runs after past trajectories are rendered. Move it earlier, right after `p_x_given_n` is defined (around line 6870), so it's available for trajectory coloring:

```python
# Get posterior data
p_x_given_n = state['a_flat_matching_results_list_ds'].p_x_given_n_list[new_epoch_idx]
posterior_2d = np.sum(p_x_given_n, axis=2)

# Generate time bin colors for use in trajectory and centroid coloring
n_time_bins = p_x_given_n.shape[2]
time_bin_colors = np.zeros((n_time_bins, 4), dtype=np.float32)
for t_idx in range(n_time_bins):
    hue = (t_idx / max(n_time_bins, 1)) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    time_bin_colors[t_idx] = (rgb[0], rgb[1], rgb[2], 0.9)
```

### 3. Modify Past Trajectory Coloring (~lines 7037-7041)

Replace the fixed red color assignment with conditional logic:

```python
# Determine base color for this trajectory
if state['color_matches_by_matching_angle'] and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
    matching_idx_values = positions_df['centroid_pos_traj_matching_angle_idx'].values
    valid_match_indices = matching_idx_values[matching_idx_values >= 0]
    if len(valid_match_indices) > 0:
        # Use the first valid matching time bin index
        matched_t_idx = int(valid_match_indices[0])
        if matched_t_idx < len(time_bin_colors):
            base_rgb = tuple(time_bin_colors[matched_t_idx][:3])
        else:
            base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)  # Fallback to red
    else:
        base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)  # Default red for non-matching
else:
    # Fixed red color for all past trajectories (matches colorbar)
    base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
```

### 4. Modify Future Trajectory Coloring (~lines 7586-7590)

Apply the same conditional logic for future trajectories:

```python
# Determine base color for this trajectory
if state['color_matches_by_matching_angle'] and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
    matching_idx_values = positions_df['centroid_pos_traj_matching_angle_idx'].values
    valid_match_indices = matching_idx_values[matching_idx_values >= 0]
    if len(valid_match_indices) > 0:
        # Use the first valid matching time bin index
        matched_t_idx = int(valid_match_indices[0])
        if matched_t_idx < len(time_bin_colors):
            base_rgb = tuple(time_bin_colors[matched_t_idx][:3])
        else:
            base_rgb = colorsys.hsv_to_rgb(0.5, 0.8, 0.9)  # Fallback to cyan
    else:
        base_rgb = colorsys.hsv_to_rgb(0.5, 0.8, 0.9)  # Default cyan for non-matching
else:
    # Fixed cyan color for all future trajectories (matches colorbar)
    base_rgb = colorsys.hsv_to_rgb(0.5, 0.8, 0.9)
```

### 5. Update Docstring (~line 6552)

Add documentation for the new parameter:

```python
color_matches_by_matching_angle: If True, trajectories that have a valid angle match (centroid_pos_traj_matching_angle_idx >= 0) 
    will be colored using the corresponding time bin's color instead of default red/cyan. Default: True.
```

### 6. Update Centroid Section to Reuse Colors (~lines 7184-7190)

The centroid section already generates `time_bin_colors`. Since we moved it earlier, we can remove the duplicate generation in the centroid section, or just keep it as-is since Python will just overwrite the variable (no harm, but slightly redundant).

## Files to Modify

- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

## Visual Result

When `color_matches_by_matching_angle=True`:

- Trajectories with a valid angle match will be colored with the same color as the corresponding time bin's centroid/contour
- This creates visual correlation between centroids and the trajectories that match their angles
- Non-matching trajectories retain the default red (past) or cyan (future) color