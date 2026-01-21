---
name: Match Trajectory Colors to Colorbar
overview: Update trajectory and timeline tick coloring to use the same red (past) and cyan (future) color scheme shown in the colorbar, with opacity varying based on time distance from the epoch boundary.
todos:
  - id: update-past-color
    content: Change past trajectory hue from variable (0-0.7) to fixed red (0.0)
    status: completed
  - id: update-future-color
    content: Change future trajectory hue from variable (0.5-0.8) to fixed cyan (0.5)
    status: completed
isProject: false
---

# Match Trajectory Colors to Colorbar

## Problem

Currently, the colorbar shows:

- **Past**: Red (hue=0.0) with opacity based on time distance
- **Future**: Cyan (hue=0.5) with opacity based on time distance

But trajectories use:

- **Past**: Variable hue 0-0.7 based on trajectory *index* (gives each trajectory a unique color)
- **Future**: Variable hue 0.5-0.8 based on trajectory *index*

This creates a visual mismatch where the colorbar legend doesn't represent the actual trajectory colors.

## Solution

Change trajectory coloring from unique-color-per-trajectory to fixed-color-per-category matching the colorbar scheme:

- All past trajectories: Red (hue=0.0)
- All future trajectories: Cyan (hue=0.5)
- Opacity continues to vary based on time distance (already implemented correctly)

## Changes Required

### 1. Update Past Trajectory Color (line ~6864-6868)

In [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py):

**Current:**

```python
hue = (idx / max(len(past_trajectory_items), 1)) * 0.7  # Use 0-0.7 range
saturation = 0.8
value = 0.9
base_rgb = colorsys.hsv_to_rgb(hue, saturation, value)
```

**Change to:**

```python
hue = 0.0  # Fixed red for all past trajectories (matches colorbar)
saturation = 0.8
value = 0.9
base_rgb = colorsys.hsv_to_rgb(hue, saturation, value)
```

### 2. Update Future Trajectory Color (line ~7029-7033)

**Current:**

```python
hue = 0.5 + (idx / max(len(future_trajectory_items), 1)) * 0.3  # Use 0.5-0.8 range
saturation = 0.8
value = 0.9
base_rgb = colorsys.hsv_to_rgb(hue, saturation, value)
```

**Change to:**

```python
hue = 0.5  # Fixed cyan for all future trajectories (matches colorbar)
saturation = 0.8
value = 0.9
base_rgb = colorsys.hsv_to_rgb(hue, saturation, value)
```

## Result

After this change:

- All past trajectories will be red with varying opacity (bright red close to epoch, faint red far from epoch)
- All future trajectories will be cyan with varying opacity (bright cyan close to epoch, faint cyan far from epoch)
- Timeline ticks will automatically inherit these colors since they use `base_rgb` from trajectory rendering
- The colorbar will accurately represent the trajectory coloring scheme

