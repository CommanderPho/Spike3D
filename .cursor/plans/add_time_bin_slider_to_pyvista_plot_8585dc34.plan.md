---
name: Add time bin slider to PyVista plot
overview: Add an interactive slider widget to the `plot_prominence_peaks_3d_pyvista` method that allows dynamically paging through available time_bin_idx values, updating the plot in real-time.
todos:
  - id: extract-time-bins
    content: Extract all available time_bin_idx values for the given epoch_idx from posterior_peaks_result.results
    status: pending
  - id: refactor-plotting
    content: Refactor mesh creation logic into _update_plot_for_time_bin() function that can be called to update the plot
    status: pending
    dependencies:
      - extract-time-bins
  - id: mesh-tracking
    content: Implement mesh actor tracking system to store references for removal when updating
    status: pending
    dependencies:
      - refactor-plotting
  - id: add-slider
    content: Add slider widget using plotter.add_slider_widget() with proper range and positioning
    status: pending
    dependencies:
      - refactor-plotting
  - id: slider-callback
    content: Implement slider callback function that calls _update_plot_for_time_bin() with the selected time_bin_idx
    status: pending
    dependencies:
      - add-slider
      - mesh-tracking
  - id: edge-cases
    content: "Handle edge cases: single time bin, invalid initial time_bin_idx, missing results"
    status: pending
    dependencies:
      - slider-callback
---

# Add Time Bin Slider to PyVista Plot

## Overview

Modify `plot_prominence_peaks_3d_pyvista` in [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py) to add an interactive slider that allows paging through available `time_bin_idx` values dynamically.

## Implementation Details

### 1. Determine Available Time Bins

- Extract all available `time_bin_idx` values for the given `epoch_idx` from `posterior_peaks_result.results.keys()`
- Filter keys where `key[0] == epoch_idx` and extract `key[1]` values
- Sort the list to ensure proper slider ordering
- Use the first available time_bin_idx as the initial value if the provided one doesn't exist

### 2. Refactor Plotting Logic

- Extract the mesh creation logic (grid, contours, spheres) into a reusable function `_update_plot_for_time_bin(plotter, time_bin_idx, ...)`
- This function will:
- Get the result data for the specified `(epoch_idx, time_bin_idx)`
- Clear existing meshes (store mesh actors in a list/container for removal)
- Create and add new meshes for the selected time bin
- Return references to the created mesh actors

### 3. Add Slider Widget

- Use `plotter.add_slider_widget()` to create an interactive slider
- Slider range: `[0, len(available_time_bins) - 1]` (indices)
- Slider value: index of current `time_bin_idx` in the sorted list
- Position: Bottom of the plotter window
- Title: "Time Bin" with current value displayed

### 4. Implement Callback Function

- Create callback function `_on_slider_change(value)` that:
- Converts slider value (index) to actual `time_bin_idx`
- Calls `_update_plot_for_time_bin()` to refresh the plot
- Updates any text labels showing the current time bin

### 5. Store Mesh References

- Maintain a list/dictionary of mesh actors that need to be removed when updating
- Include: grid mesh, col contour polylines, probe level polylines, peak center spheres
- Clear this list before adding new meshes

### 6. Handle Edge Cases

- If only one time bin is available, either hide the slider or disable it
- Handle cases where a time_bin_idx might not have results (skip it in the slider)
- Ensure the initial time_bin_idx is valid or use the first available

## Code Structure

```python
# At the start of the method:
available_time_bins = sorted([k[1] for k in posterior_peaks_result.results.keys() if k[0] == epoch_idx])
if len(available_time_bins) == 0:
    raise ValueError(f"No results found for epoch_idx={epoch_idx}")

# Initialize with first available if provided one doesn't exist
if time_bin_idx not in available_time_bins:
    time_bin_idx = available_time_bins[0]

# Store mesh actors for cleanup
mesh_actors = []

# Refactored plotting function
def _update_plot_for_time_bin(t_idx):
    # Clear existing meshes
    for actor in mesh_actors:
        plotter.remove_actor(actor)
    mesh_actors.clear()
    
    # ... recreate all meshes for t_idx ...
    # Append new actors to mesh_actors
    
# Initial plot
_update_plot_for_time_bin(time_bin_idx)

# Add slider if multiple time bins available
if len(available_time_bins) > 1:
    def slider_callback(value):
        idx = int(round(value))
        t_idx = available_time_bins[idx]
        _update_plot_for_time_bin(t_idx)
    
    plotter.add_slider_widget(
        slider_callback,
        value=available_time_bins.index(time_bin_idx),
        rng=[0, len(available_time_bins) - 1],
        title='Time Bin',
        pointa=(0.02, 0.02),
        pointb=(0.98, 0.02)
    )
```