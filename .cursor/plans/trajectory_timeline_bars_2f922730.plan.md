---
name: trajectory timeline bars
overview: Add timeline bars below the past and future views that display vertical tick marks for each trajectory, showing their temporal positions relative to the full recording duration and current epoch.
todos:
  - id: modify-grid-layout
    content: "Modify grid layout: change past/future views to row 0 only, add timeline views at row 1, shift colorbar to row 3"
    status: completed
  - id: add-state-variables
    content: Add timeline state variables and store curr_position_df for computing recording time range
    status: completed
  - id: add-cleanup-code
    content: Add cleanup code in update_epoch_display for timeline ticks and markers
    status: completed
  - id: render-past-timeline
    content: Implement past timeline bar rendering with trajectory ticks (capturing colors during trajectory rendering)
    status: completed
  - id: render-future-timeline
    content: Implement future timeline bar rendering with trajectory ticks (capturing colors during trajectory rendering)
    status: completed
  - id: setup-timeline-cameras
    content: Setup timeline view cameras with appropriate ranges and add time axis labels
    status: completed
isProject: false
---

# Trajectory Timeline Bars Feature

## Overview

Add two horizontal timeline bars below the past and future trajectory views. Each bar represents the full recording duration with vertical tick marks showing where each displayed trajectory falls in time, colored to match the trajectory in the plot above.

## Current Grid Layout

```
Row 0-1: past_view (col 0)  | posterior_2d_view (col 1, row 0)   | future_view (col 2)
                            | time_bin_grid (col 1, row 1)       |
Row 2:   colorbar_view (col 0-2)
```

## New Grid Layout

```
Row 0:   past_view (col 0)  | posterior_2d_view (col 1)          | future_view (col 2)
Row 1:   past_timeline (col 0) | time_bin_grid (col 1)           | future_timeline (col 2)
Row 2:   colorbar_view (col 0-2)
```

## Implementation Steps

### 1. Modify Grid Layout in `render_predictive_decoding_with_vispy`

In [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) around line 6565-6580:

- Change `past_view` from `row_span=2` to single row (row 0)
- Change `future_view` from `row_span=2` to single row (row 0)
- Add `past_timeline_view` at row 1, col 0
- Add `future_timeline_view` at row 1, col 2
- Move `colorbar_view` to row 3

### 2. Store Recording Time Range in State

- Store `curr_position_df` reference in state (needed for computing full recording duration)
- Compute `recording_t_min` and `recording_t_max` from `curr_position_df['t']`
- Add these to state for use in `update_epoch_display`

### 3. Add Timeline State Variables

Add to state dictionary:

- `past_timeline_view`: ViewBox reference
- `future_timeline_view`: ViewBox reference  
- `past_timeline_ticks`: List of Line visuals for past trajectory ticks
- `future_timeline_ticks`: List of Line visuals for future trajectory ticks
- `past_timeline_epoch_marker`: Visual for current epoch marker
- `future_timeline_epoch_marker`: Visual for current epoch marker

### 4. Add Cleanup in `update_epoch_display`

Clear previous timeline visuals:

- Clear `past_timeline_ticks` list
- Clear `future_timeline_ticks` list
- Clear epoch markers

### 5. Render Past Timeline Bar

After rendering past trajectories (around line 6860):

- Draw background rectangle representing full recording duration
- Draw vertical line at `epoch_start_t` marking current epoch boundary
- For each past trajectory: calculate representative time (mean of trajectory times), draw vertical tick at that position using same `base_rgb` color as the trajectory line

### 6. Render Future Timeline Bar

After rendering future trajectories (around line 7020):

- Draw background rectangle representing full recording duration
- Draw vertical line at `epoch_end_t` marking current epoch boundary
- For each future trajectory: calculate representative time (mean of trajectory times), draw vertical tick at that position using same `base_rgb` color as the trajectory line

### 7. Setup Timeline Cameras

After the main view camera setup (around line 7040):

- Set `past_timeline_view.camera` to PanZoomCamera with x-range covering recording duration, y-range for bar height
- Set `future_timeline_view.camera` similarly

## Key Code Locations

- Grid layout setup: lines 6565-6580
- State dictionary: lines 6593-6618
- Cleanup code: lines 6628-6680
- Past trajectory rendering: lines 6815-6861 (need to capture colors and times)
- Future trajectory rendering: lines 6975-7021 (need to capture colors and times)
- Camera setup: lines 7038-7060

## Visual Design

- Timeline bar height: ~30-40 pixels
- Bar background: dark gray rectangle
- Current epoch boundary: bright vertical line (white or yellow)
- Trajectory ticks: vertical lines with same HSV color as corresponding trajectory
- Add axis labels showing time values at key positions