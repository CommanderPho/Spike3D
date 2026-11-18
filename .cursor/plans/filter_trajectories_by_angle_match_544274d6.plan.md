---
name: Filter Trajectories by Angle Match
overview: Add a parameter to `render_predictive_decoding_with_vispy` that filters displayed past/future trajectories based on their `centroid_pos_traj_matching_angle_idx` column value, allowing users to exclude trajectories that don't have angular alignment with the decoded posterior centroids.
todos:
  - id: add-param
    content: "Add `require_angle_match: bool = False` parameter to function signature"
    status: completed
  - id: update-docstring
    content: Add parameter documentation to docstring
    status: completed
  - id: add-state
    content: Store `require_angle_match` in state dict
    status: completed
  - id: filter-past
    content: Add filtering logic to past trajectories rendering loop
    status: completed
  - id: filter-future
    content: Add filtering logic to future trajectories rendering loop
    status: completed
isProject: false
---

# Filter Trajectories by Angle Match

## Context

The `render_predictive_decoding_with_vispy` function renders past and future position trajectories. Each trajectory's `positions_df` may contain a `centroid_pos_traj_matching_angle_idx` column:

- Value `>= 0`: trajectory direction matches a centroid segment direction (within 60 degrees)
- Value `== -1`: no angular match found

## Changes

### 1. Add New Parameter to Function Signature

In [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) at line ~6510-6513:

```python
def render_predictive_decoding_with_vispy(..., 
    require_angle_match: bool = False,  # NEW PARAMETER
    **kwargs):
```

- `require_angle_match=False` (default): show all trajectories (current behavior)
- `require_angle_match=True`: only show trajectories where `centroid_pos_traj_matching_angle_idx >= 0`

### 2. Update Docstring

Add documentation for the new parameter at line ~6537:

```python
require_angle_match: If True, only display trajectories whose direction aligns with the decoded posterior centroid direction (centroid_pos_traj_matching_angle_idx >= 0). Default: False.
```

### 3. Store in State Dict

Add the parameter to the `state` dict at line ~6719:

```python
'require_angle_match': require_angle_match,
```

### 4. Add Filtering Logic to Past Trajectories Loop

At line ~7008 inside the past trajectories loop, add a check before rendering:

```python
for idx, (epoch_id, positions_df) in enumerate(past_trajectory_items):
    # Filter by angle match if required
    if state['require_angle_match'] and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
        if not (positions_df['centroid_pos_traj_matching_angle_idx'] >= 0).any():
            continue  # Skip this trajectory - no angle match
    # ... existing rendering code
```

### 5. Add Filtering Logic to Future Trajectories Loop

At line ~7511 inside the future trajectories loop, add the same check:

```python
for idx, (epoch_id, positions_df) in enumerate(future_trajectory_items):
    # Filter by angle match if required
    if state['require_angle_match'] and 'centroid_pos_traj_matching_angle_idx' in positions_df.columns:
        if not (positions_df['centroid_pos_traj_matching_angle_idx'] >= 0).any():
            continue  # Skip this trajectory - no angle match
    # ... existing rendering code
```

## Notes

- The filter checks if ANY position in the trajectory has a valid match. An alternative stricter approach would filter individual positions, but since each trajectory is a contiguous sequence, checking the whole trajectory makes sense.
- The column may not exist if `recompute_relevant_position_active_mask_centroid_traj_angle()` was not called, so we check for column existence before filtering.