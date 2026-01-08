---
name: Make sliders snap to integers
overview: Modify the sliders in `DecodedTrajectoryPyVistaPlotter` to snap to whole integer values by setting the VTK slider representation resolution and updating callbacks to round and set the slider value to the nearest integer.
todos:
  - id: set_epoch_slider_resolution
    content: Set resolution on slider_epoch after creation to enable integer snapping
    status: pending
  - id: set_timebin_slider_resolution
    content: Set resolution on slider_epoch_time_bin after creation to enable integer snapping
    status: pending
  - id: modify_epoch_callback
    content: Modify _on_slider_callback_epoch_idx to round values and snap slider to integer
    status: pending
  - id: modify_timebin_callback
    content: Modify _on_slider_callback_epoch_time_bin to round values and snap slider to integer
    status: pending
---

# Optional Minor: Make Sliders Snap to Integer Values in DecodedTrajectoryPyVistaPlotter

## Overview

The `DecodedTrajectoryPyVistaPlotter` class has two sliders (`slider_epoch` and `slider_epoch_time_bin`) that currently behave as continuous sliders. We need to make them snap to whole integer values instead of allowing smooth sliding.

## Implementation Plan

### File to Modify

- [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py)

### Changes Required

1. **Set slider resolution after creation** (lines ~3243 and ~3277):

- After creating `slider_epoch`, access its VTK representation and set the resolution to `(num_filter_epochs - 1)` to allow only integer steps
- After creating `slider_epoch_time_bin`, set its resolution to `(curr_num_epoch_time_bins - 1)`

2. **Modify callbacks to snap values** (lines ~3217-3243 and ~3248-3277):

- Update `_on_slider_callback_epoch_idx` to round the value to the nearest integer and programmatically set the slider to that value before processing
- Update `_on_slider_callback_epoch_time_bin` similarly to round and snap to integer values

### Technical Details

The VTK slider representation (accessed via `slider.GetRepresentation()`) has a `SetResolution()` method that controls the number of discrete steps. Setting it to `(max - min)` will create integer steps.Additionally, the callbacks should round incoming values and update the slider's displayed value to ensure visual snapping, even if the user drags between integer positions.

### Implementation Approach

1. After creating each slider widget, immediately set its resolution:
   ```python
                              self.slider_epoch.GetRepresentation().SetResolution(num_filter_epochs - 1)
                              self.slider_epoch_time_bin.GetRepresentation().SetResolution(curr_num_epoch_time_bins - 1)
            
         
      
   
   ```