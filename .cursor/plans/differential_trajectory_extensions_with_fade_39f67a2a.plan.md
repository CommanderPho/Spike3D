---
name: Differential trajectory extensions with fade
overview: Modify `render_predictive_decoding_with_vispy` to support separate extension amounts for trajectory starts vs ends, with gradual fade-out for end extensions while keeping start extensions solid.
todos: []
isProject: false
---

# Differential Trajectory Extensions with Fade-Out

## Overview

Enable separate `past_future_trajectory_extension_seconds` values for trajectory starts vs ends, allowing visual indication of trajectory direction through gradual fade-out at ends while keeping start extensions solid.

## Changes Required

### 1. Function Signature Update

**File**: `PredictiveDecodingComputations.py:6510`

Change the parameter from a single float to support either:

- A single float (backward compatible) - applies to both start and end
- A tuple/list of two floats `(start_extension, end_extension)` - separate values
- A dict `{'start': float, 'end': float}` - explicit naming

**Preferred approach**: Accept a tuple `(start_extension, end_extension)` for clarity, with backward compatibility for single float values.

### 2. Parameter Parsing Logic

**Location**: After line 6695 in the state dictionary setup

Add logic to parse the extension parameter:

- If it's a float: use same value for both start and end
- If it's a tuple/list with 2 elements: `start_extension, end_extension = extension_seconds`
- If it's a dict: extract `start` and `end` keys

Store both values in the state dictionary as:

- `past_future_trajectory_extension_start_seconds`
- `past_future_trajectory_extension_end_seconds`

### 3. Past Trajectory Rendering

**Location**: Lines 7019-7043

Modify the extension logic for past trajectories:

- **Start extension** (backward from `traj_t_min`): Use `start_extension_seconds`, render with solid opacity (0.2) - no fade-in, matching current extended trajectory opacity
- **End extension** (forward from `traj_t_max`): Use `end_extension_seconds`, render with gradual fade-out from trajectory end opacity to 0.2

Implementation:

- Compute both extensions separately
- For start extension: use constant opacity (0.2) - solid rendering, matching current behavior for extended portions
- For end extension: create opacity gradient from the trajectory end point's actual opacity (which varies from 0.2 to 1.0 based on distance from epoch) down to 0.2 at the extension end
- Prepend start extension, append end extension

### 4. Future Trajectory Rendering  

**Location**: Lines 7475-7499

Apply the same logic to future trajectories:

- **Start extension** (backward from `traj_t_min`): Use `start_extension_seconds`, render with solid opacity (0.2) - no fade-in, matching current extended trajectory opacity
- **End extension** (forward from `traj_t_max`): Use `end_extension_seconds`, render with gradual fade-out from trajectory end opacity to 0.2

### 5. Opacity Gradient Implementation

**Important Context**:

- Trajectory points have variable opacity (0.2 to 1.0) based on distance from epoch
- Extended portions currently use fixed 0.2 opacity
- The trajectory end point may have opacity anywhere from 0.2 to 1.0

For gradual fade-out at ends:

- Get the opacity of the trajectory end point: `trajectory_end_opacity = opacity[-1]` (last point in trajectory)
- Calculate distance from trajectory boundary (end point) for each extension point
- Normalize distance by extension length
- Map to opacity: `opacity = trajectory_end_opacity - (distance / extension_length) * (trajectory_end_opacity - 0.2)` 
- This creates a smooth transition from the trajectory end's actual opacity down to 0.2 at the extension end
- Range: `trajectory_end_opacity` → 0.2

For start extensions:

- Use constant opacity: `opacity = 0.2` (solid, no gradient, matching current extended trajectory behavior)

### 6. Documentation Update

**Location**: Lines 6528

Update the docstring to document the new parameter format and behavior:

- Explain tuple/list format for separate start/end values
- Describe solid rendering behavior at starts (no fade-in)
- Describe gradual fade-out behavior at ends

## Implementation Details

### Key Code Locations

- Function signature: `6510`
- State dictionary: `6695`
- Past trajectory extension: `7019-7043`
- Future trajectory extension: `7475-7499`
- Docstring: `6528`

### Backward Compatibility

Maintain backward compatibility by checking parameter type:

```python
if isinstance(past_future_trajectory_extension_seconds, (int, float)):
    start_extension = end_extension = past_future_trajectory_extension_seconds
elif isinstance(past_future_trajectory_extension_seconds, (tuple, list)) and len(past_future_trajectory_extension_seconds) == 2:
    start_extension, end_extension = past_future_trajectory_extension_seconds
elif isinstance(past_future_trajectory_extension_seconds, dict):
    start_extension = past_future_trajectory_extension_seconds.get('start', 0.0)
    end_extension = past_future_trajectory_extension_seconds.get('end', 0.0)
```

### Visual Effect

- **Start extensions**: Solid opacity (0.2) - no fade-in, constant reduced opacity matching current extended trajectory behavior
- **End extensions**: Gradually fading out from trajectory end opacity (0.2-1.0) → 0.2 to indicate trajectory direction and termination