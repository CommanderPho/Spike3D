---
name: Complete peak contours layer implementation
overview: Complete the `add_peak_contours_layer` method in `DecodedTrajectoryNapariPlotter` to display peak contours that dynamically update when the user changes the epoch and time_bin sliders in Napari.
todos:
  - id: create_shapes_layer
    content: Create initial Napari shapes layer with empty data and appropriate shape_type
    status: pending
  - id: extract_contours_helper
    content: Create helper function to extract contours from peaks_dict and convert matplotlib Path to Napari shape format
    status: pending
    dependencies:
      - create_shapes_layer
  - id: create_update_callback
    content: Create callback function that updates shapes layer when epoch/time_bin sliders change
    status: pending
    dependencies:
      - extract_contours_helper
  - id: connect_callback
    content: Connect the update callback to viewer.dims.events.current_step
    status: pending
    dependencies:
      - create_update_callback
  - id: initial_display
    content: Display initial contours for current epoch/time_bin when layer is first created
    status: pending
    dependencies:
      - connect_callback
  - id: handle_edge_cases_contours
    content: Add error handling for missing keys, empty peaks_dict, and None contours
    status: pending
    dependencies:
      - initial_display
---

# Complete Peak Contours Layer Implementation

## Overview

Complete the `add_peak_contours_layer` method to create a Napari shapes layer that displays peak contours from `PosteriorPeaksPeakProminence2dResult`. The contours should update dynamically when the user changes the epoch and time_bin sliders.

## Implementation Details

### Current State

The method currently:

- Extracts `peaks_dict` from the results for a single (epoch_idx, time_bin_idx) tuple
- Has a TODO to finish the implementation

### Required Implementation

1. **Create Napari Shapes Layer**: Create an initial shapes layer with empty data that will be populated dynamically.
2. **Extract Contours from peaks_dict**: 

- For each peak in `peaks_dict`, extract contours from `level_slices`
- Convert matplotlib `Path` objects to Napari shape format (array of vertices)
- Handle the case where contours may be None or missing

3. **Create Update Callback Function**:

- Connect to `viewer.dims.events.current_step` (similar to `_on_current_step_change` in `build_ui`)
- Extract the current epoch_idx and time_bin_idx from the slider values
- Look up the corresponding peaks_dict from `peak_prominence_result.results[(epoch_idx, time_bin_idx)]`
- Extract all contours from the peaks_dict's level_slices
- Convert matplotlib Path contours to Napari shape format (vertices arrays)
- Update the shapes layer data with the new contours

4. **Shape Format Conversion**:

- Matplotlib Path objects have `.vertices` attribute (Nx2 array)
- Napari shapes expect a list of arrays, where each array is a shape's vertices
- For closed contours, ensure the first and last points match (or use Napari's closed shape handling)
- Use shape type 'path' or 'polygon' depending on Napari version

5. **Handle Edge Cases**:

- Missing (epoch_idx, time_bin_idx) keys in results dict
- Empty peaks_dict
- Missing or None contours
- Initial display when viewer is first created

### Key Implementation Points

- **Store Reference**: Store `peak_prominence_result` as an instance attribute or in the callback closure so it can be accessed when sliders change
- **Shape Type**: Use `shape_type='path'` or `shape_type='polygon'` for Napari shapes layer