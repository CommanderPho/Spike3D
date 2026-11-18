---
name: Fix trajectory color bugs
overview: "Fix two bugs: (1) timeline ticks should always use red/cyan colors, not time-bin colors, and (2) correctly map from segment index to time bin index when coloring trajectories."
todos:
  - id: fix-past-timeline
    content: Store original red color for past trajectory timeline ticks before conditional coloring
    status: completed
  - id: fix-future-timeline
    content: Store original cyan color for future trajectory timeline ticks before conditional coloring
    status: completed
  - id: fix-past-segment-mapping
    content: Add segment-to-time-bin mapping for past trajectory color lookup
    status: completed
  - id: fix-future-segment-mapping
    content: Add segment-to-time-bin mapping for future trajectory color lookup
    status: completed
isProject: false
---

# Fix Trajectory Color Matching Bugs

## Problem Summary

Two bugs were introduced when implementing `color_matches_by_matching_angle`:

1. **Timeline ticks use wrong colors**: Timeline ticks at the bottom should always be red (past) / cyan (future), but they now use the time-bin matched color.

2. **Segment index vs time bin index mismatch**: The `centroid_pos_traj_matching_angle_idx` column stores a **segment row index** (into `a_centroids_search_segments_df`), but the code uses it directly as a **time bin index** to look up colors.

## Data Structure Relationships

```
centroids_df (one row per TIME BIN):
  Row 0: time_bin=0, segment_idx=0
  Row 1: time_bin=1, segment_idx=0  
  Row 2: time_bin=2, segment_idx=1
  Row 3: time_bin=3, segment_idx=1

a_centroids_search_segments_df (one row per SEGMENT):
  Row 0: segment_idx=0 (aggregated from time bins 0,1)
  Row 1: segment_idx=1 (aggregated from time bins 2,3)

centroid_pos_traj_matching_angle_idx:
  Stores ROW INDEX into a_centroids_search_segments_df (0 or 1)
  NOT the time bin index
```

## Fix Implementation

### Fix 1: Timeline Ticks (~lines 7070, 7630)

Store the **original** red/cyan color for timeline ticks, separate from the trajectory color.

**Past trajectories (~line 7070):**

```python
# Store ORIGINAL red color for timeline (before conditional coloring)
timeline_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)  # Always red for past
past_trajectory_colors_and_times.append((timeline_rgb, mean_time))
```

**Future trajectories (~line 7630):**

```python
# Store ORIGINAL cyan color for timeline (before conditional coloring)  
timeline_rgb = colorsys.hsv_to_rgb(0.5, 0.8, 0.9)  # Always cyan for future
future_trajectory_colors_and_times.append((timeline_rgb, mean_time))
```

### Fix 2: Correct Segment-to-Time-Bin Mapping (~lines 7049-7056, 7608-7616)

Replace the incorrect direct lookup with proper segment-to-time-bin mapping using the epoch's `centroids_df`.

**Approach:**

1. Get `centroid_pos_traj_matching_angle_idx` (segment row index)
2. Look up the actual `segment_idx` value from `a_centroids_search_segments_df`
3. Find time bins in `centroids_df` that have this `segment_idx`
4. Use the first matching time bin's index for color lookup

**Implementation for past trajectories (~lines 7049-7056):**

```python
if state['color_matches_by_matching_angle'] and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
    matching_idx_values = positions_df['centroid_pos_traj_matching_angle_idx'].values
    valid_match_indices = matching_idx_values[matching_idx_values >= 0]
    if len(valid_match_indices) > 0:
        segment_row_idx = int(valid_match_indices[0])
        # Map segment row index to actual time bin index
        if epoch_result is not None and hasattr(epoch_result, 'centroids_df') and epoch_result.centroids_df is not None:
            if hasattr(epoch_result, 'a_centroids_search_segments_df') and epoch_result.a_centroids_search_segments_df is not None:
                search_df = epoch_result.a_centroids_search_segments_df
                if segment_row_idx < len(search_df):
                    actual_segment_idx = search_df.iloc[segment_row_idx]['segment_idx']
                    # Find first time bin with this segment_idx
                    matching_t_bins = epoch_result.centroids_df[epoch_result.centroids_df['segment_idx'] == actual_segment_idx].index
                    if len(matching_t_bins) > 0:
                        matched_t_idx = matching_t_bins[0]  # First time bin in segment
                        if matched_t_idx < len(time_bin_colors):
                            base_rgb = tuple(time_bin_colors[matched_t_idx][:3])
                        else:
                            base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
                    else:
                        base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
                else:
                    base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
            else:
                base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
        else:
            base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
    else:
        base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
else:
    base_rgb = colorsys.hsv_to_rgb(0.0, 0.8, 0.9)
```

**Same fix for future trajectories (~lines 7608-7616)** with cyan fallback (hue=0.5).

## Files to Modify

- [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)
  - Lines ~7049-7070: Past trajectory color and timeline storage
  - Lines ~7608-7630: Future trajectory color and timeline storage

## Testing

After the fix:

- Timeline ticks should always be red (past) / cyan (future)
- Trajectory colors should match the contour colors they're associated with