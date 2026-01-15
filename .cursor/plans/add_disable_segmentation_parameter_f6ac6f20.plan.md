---
name: Add disable_segmentation parameter
overview: "Add a `disable_segmentation: bool = True` kwarg to `_compare_centroid_and_pos_traj_angle` that bypasses ruptures/changepoint detection and treats all data as a single segment when True."
todos:
  - id: "1"
    content: Add disable_segmentation parameter to _compare_centroid_and_pos_traj_angle method signature and pass it to adding_segmented_trajectories_columns
    status: completed
  - id: "2"
    content: Add disable_segmentation parameter to adding_segmented_trajectories_columns method signature and pass it to perform_segment_trajectories
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Add disable_segmentation parameter to perform_segment_trajectories and implement logic to skip ruptures when True, assigning all rows to segment_idx=0
    status: completed
    dependencies:
      - "2"
---

# Add disable_segmentation parameter to skip changepoint detection

## Overview

Add a `disable_segmentation: bool = True` parameter that, when True, skips ruptures/changepoint detection entirely and treats all trajectory data as a single segment (segment_idx=0 for all rows).

## Implementation Details

### 1. Update `_compare_centroid_and_pos_traj_angle` method

**File**: `pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py` (line 394)

- Add `disable_segmentation: bool = True` parameter to the method signature
- Pass this parameter to `adding_segmented_trajectories_columns()` call on line 403

### 2. Update `adding_segmented_trajectories_columns` method  

**File**: `H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py` (line 755)

- Add `disable_segmentation: bool = False` parameter to the method signature
- Pass this parameter through to `perform_segment_trajectories()` call on line 763

### 3. Update `perform_segment_trajectories` method

**File**: `H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py` (line 615)

- Add `disable_segmentation: bool = False` parameter to the method signature
- When `disable_segmentation=True`, skip the ruptures/changepoint detection logic (lines 687-722) and:
- Set `needs_segmentation = False` immediately after signal validation
- Assign all rows to `segment_idx = 0` (similar to the fallback behavior on lines 657, 663, 673, 677, 721)
- Still compute the segment statistics (Vp_rad, segment_Vp_deg, segment_Vp_scatteredness, etc.) for the single segment

## Key Changes

1. **Parameter propagation**: The `disable_segmentation` flag flows from `_compare_centroid_and_pos_traj_angle` → `adding_segmented_trajectories_columns` → `perform_segment_trajectories`

2. **Early exit in segmentation**: When `disable_segmentation=True`, skip the ruptures import and PELT algorithm entirely, assigning all data points to segment 0

3. **Preserve downstream logic**: Even when segmentation is disabled, still compute segment-level statistics (mean angles, scatteredness) for the single segment so downstream code continues to work

## Default Behavior

- Default is `True` in `_compare_centroid_and_pos_traj_angle` (as requested)
- Default is `False` in the lower-level methods to maintain backward compatibility for other callers