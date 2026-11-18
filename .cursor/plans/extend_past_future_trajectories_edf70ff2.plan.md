---
name: Extend Past/Future Trajectories
overview: Add an optional parameter to extend past/future position trajectories beyond their computed bounds, similar to how `current_traj_seconds_pre_post_extension` works for the current epoch trajectory.
todos:
  - id: add-param
    content: Add past_future_trajectory_extension_seconds parameter to function signature and docstring
    status: completed
  - id: store-state
    content: Store the new parameter in the state dictionary
    status: completed
  - id: extend-past
    content: Modify past trajectory rendering to prepend extended positions with 0.2 alpha
    status: completed
  - id: extend-future
    content: Modify future trajectory rendering to append extended positions with 0.2 alpha
    status: completed
isProject: false
---

# Extend Past/Future Position Trajectories

## Overview

Add a new parameter `past_future_trajectory_extension_seconds` that optionally extends the past/future trajectories (colored red/cyan lines) beyond their computed time bounds, using position data from `curr_position_df`. Extended portions render with reduced alpha (0.2).

## Changes Required

### 1. Add Parameter to Function Signature

**File**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) (line 6510)

```python
def render_predictive_decoding_with_vispy(..., current_traj_seconds_pre_post_extension: float = 0.750, past_future_trajectory_extension_seconds: float = 0.0, **kwargs):
```

### 2. Store Parameter in State Dict

**Location**: Around line 6687

Add `'past_future_trajectory_extension_seconds': past_future_trajectory_extension_seconds` to the state dictionary.

### 3. Modify Past Trajectory Rendering

**Location**: Lines 6927-6978

For each past trajectory:

- Get the min/max time of the trajectory from `positions_df['t']`
- If extension > 0, query `curr_position_df` for positions in range `[t_min - extension, t_min]`
- Prepend extended positions to the trajectory
- Assign alpha=0.2 to extended points, preserve existing opacity for original points

### 4. Modify Future Trajectory Rendering

**Location**: Lines 7263-7314

For each future trajectory:

- Get the min/max time of the trajectory from `positions_df['t']`
- If extension > 0, query `curr_position_df` for positions in range `[t_max, t_max + extension]`
- Append extended positions to the trajectory
- Assign alpha=0.2 to extended points, preserve existing opacity for original points

## Implementation Pattern

For each trajectory (past or future), the rendering logic follows this pattern:

```python
# Get trajectory time bounds
if 't' in positions_df.columns:
    traj_t_min = positions_df['t'].min()
    traj_t_max = positions_df['t'].max()
    
    extension = state['past_future_trajectory_extension_seconds']
    if extension > 0 and state['curr_position_df'] is not None:
        # For past: extend backward (before traj_t_min)
        # For future: extend forward (after traj_t_max)
        ext_mask = (state['curr_position_df']['t'] >= ext_start) & (state['curr_position_df']['t'] <= ext_end)
        ext_positions = state['curr_position_df'][ext_mask]
        
        # Combine: prepend/append extended positions
        # Set alpha=0.2 for extended, keep computed opacity for original
```

## Notes

- The extension uses `curr_position_df` (same source as current epoch trajectory)
- Extended portions are rendered with 0.2 alpha to visually distinguish them
- Past trajectories extend backward in time (before their start)
- Future trajectories extend forward in time (after their end)
- Default value is 0.0 (disabled) to maintain backward compatibility

