---
name: Re-enable Time Bin Colors
overview: Modify the `render_predictive_decoding_with_vispy` function to re-enable per-time-bin coloring for centroids, arrows, contours, and time bin view frames, with diagnostic prints to verify count consistency.
todos:
  - id: diagnostics
    content: Add diagnostic print statements to verify count consistency between n_time_bins, n_centroids, n_mask_time_bins, and n_views
    status: completed
  - id: colormap
    content: Create time_bin_colors array using HSV colormap with n_time_bins distinct colors
    status: completed
  - id: centroid-colors
    content: Replace white centroid colors with per-time-bin colors based on row index in centroids_df
    status: completed
  - id: arrow-colors
    content: Replace white arrow colors with per-time-bin colors matching their corresponding centroid
    status: completed
  - id: contours
    content: Replace 2D collapsed mask contours with per-time-bin 3D mask slice contours, each with distinct color
    status: completed
  - id: view-borders
    content: Set time bin view border_color to match the corresponding time bin color
    status: completed
isProject: false
---

# Re-enable Individual Time Bin Colors in Vispy Visualization

## Summary

Modify the `render_predictive_decoding_with_vispy` function in [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) to:

1. Add diagnostic prints verifying consistent counts between time bins, centroids, masks, and views
2. Re-enable per-time-bin coloring for centroids and arrows (currently overridden to white at lines 7189-7192)
3. Render per-time-bin contours using the 3D `epoch_t_bins_high_prob_pos_mask` instead of 2D collapsed mask
4. Add colored borders to time bin views in the bottom row

## Key Data Relationships

- `p_x_given_n.shape[2]` = number of time bins
- `epoch_t_bins_high_prob_pos_mask.shape[2] `= number of time bins (from `epoch_result.epoch_t_bins_high_prob_pos_mask`)
- `centroids_df` rows = one per time bin (row index = time bin index)
- `time_bin_views` = one view per time bin (limited by `max_time_bins_to_show`)

## Implementation Details

### 1. Add Diagnostic Count Verification (line ~7170)

After getting centroids data, add verification prints:

```python
# Verify count consistency
n_time_bins = p_x_given_n.shape[2]
n_centroids = len(centroids_df)
n_mask_time_bins = epoch_result.epoch_t_bins_high_prob_pos_mask.shape[2] if epoch_result.epoch_t_bins_high_prob_pos_mask is not None else 0
n_views = min(n_time_bins, state['max_time_bins_to_show'])
print(f"COUNT VERIFICATION - n_time_bins: {n_time_bins}, n_centroids: {n_centroids}, n_mask_time_bins: {n_mask_time_bins}, n_views: {n_views}")
if n_centroids != n_time_bins:
    print(f"  WARNING: Centroid count ({n_centroids}) != time bin count ({n_time_bins})")
if n_mask_time_bins != n_time_bins:
    print(f"  WARNING: Mask time bin count ({n_mask_time_bins}) != posterior time bin count ({n_time_bins})")
```

### 2. Generate Time Bin Colormap (before centroid rendering, ~line 7172)

Create a helper function or inline code to generate distinct colors for each time bin:

```python
def generate_time_bin_colors(n_bins: int) -> np.ndarray:
    """Generate distinct colors for each time bin using HSV color space."""
    colors = np.zeros((n_bins, 4), dtype=np.float32)
    for i in range(n_bins):
        hue = (i / max(n_bins, 1)) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = (rgb[0], rgb[1], rgb[2], 0.9)  # High opacity
    return colors

time_bin_colors = generate_time_bin_colors(n_time_bins)
```

### 3. Re-enable Per-Time-Bin Centroid/Arrow Colors (~lines 7189-7247)

Replace the white color override:

```python
# BEFORE (current - lines 7189-7196):
centroid_base_color = (1.0, 1.0, 1.0, 0.8)  # White
centroid_colors[:] = centroid_base_color

# AFTER: Use time bin index for coloring
n_centroids = len(x_centroids)
centroid_colors = np.zeros((n_centroids, 4), dtype=np.float32)
for i in range(n_centroids):
    t_idx = i  # Row index in centroids_df corresponds to time bin index
    if t_idx < len(time_bin_colors):
        centroid_colors[i] = time_bin_colors[t_idx]
    else:
        centroid_colors[i] = (1.0, 1.0, 1.0, 0.8)  # Fallback to white
```

For arrows (~line 7242), use the same color mapping:

```python
# Get color for this arrow based on time bin index
arrow_t_idx = np.where(valid_angle_mask)[0][i]  # Get original index
if arrow_t_idx < len(time_bin_colors):
    arrow_color = tuple(time_bin_colors[arrow_t_idx])
else:
    arrow_color = (1.0, 1.0, 1.0, 0.8)
```

### 4. Render Per-Time-Bin Contours (~lines 7489-7529)

Replace the 2D collapsed mask contours with per-time-bin 3D mask contours:

```python
# Get 3D per-time-bin mask from epoch_result
if new_epoch_idx < len(state['epoch_flat_mask_future_past_result']):
    epoch_result = state['epoch_flat_mask_future_past_result'][new_epoch_idx]
    if epoch_result is not None and hasattr(epoch_result, 'epoch_t_bins_high_prob_pos_mask') and epoch_result.epoch_t_bins_high_prob_pos_mask is not None:
        per_t_bin_mask = epoch_result.epoch_t_bins_high_prob_pos_mask  # Shape: (N_XBINS, N_Y_BINS, N_TIME_BINS)
        n_mask_t_bins = per_t_bin_mask.shape[2]
        
        for t_idx in range(n_mask_t_bins):
            mask_slice = per_t_bin_mask[:, :, t_idx]
            if np.any(mask_slice):
                mask_transposed = mask_slice.T.astype(np.float32)
                contours = measure.find_contours(mask_transposed, level=0.5)
                
                # Get color for this time bin
                contour_color = tuple(time_bin_colors[t_idx]) if t_idx < len(time_bin_colors) else (1.0, 1.0, 1.0, 0.55)
                
                for contour in contours:
                    # Convert pixel to world coordinates (same as current code)
                    x_world = x_min + (contour[:, 1] / n_x_bins) * (x_max - x_min)
                    y_world = y_min + (contour[:, 0] / n_y_bins) * (y_max - y_min)
                    contour_coords = np.column_stack([x_world, y_world]).astype(np.float32)
                    
                    # Add colored contours to views
                    posterior_contour = scene.visuals.Line(pos=contour_coords, color=contour_color, width=2, parent=state['posterior_2d_view'].scene)
                    posterior_contour.order = 10
                    state['posterior_mask_contours'].append(posterior_contour)
```

### 5. Add Colored Borders to Time Bin Views (~lines 7455-7459)

When creating time bin views, set colored borders:

```python
for t_idx in range(n_bins_to_show):
    t_bin_color = tuple(time_bin_colors[t_idx][:3]) if t_idx < len(time_bin_colors) else (0.5, 0.5, 0.5)
    view = state['time_bin_grid'].add_view(row=0, col=t_idx, border_color=t_bin_color)
    # ... rest of view setup
```

## Files to Modify

- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) - `render_predictive_decoding_with_vispy` function (lines 6524-7720+)

## Key Changes Summary

| Element | Current | After |

|---------|---------|-------|

| Centroids | White | Per-time-bin HSV colors |

| Arrows | White | Per-time-bin HSV colors (matching centroid) |

| Contours | White, from 2D collapsed mask | Per-time-bin HSV colors, from 3D mask slices |

| Time bin view borders | Gray | Per-time-bin HSV colors |

| Diagnostic prints | None | Count verification for time bins, centroids, masks, views |