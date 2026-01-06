---
name: Reset time_bin on epoch change and update maximum
overview: Modify `DecodedTrajectoryNapariPlotter` to reset the time_bin slider to minimum and update its maximum range when the epoch slider changes, since different epochs may have different numbers of time bins.
todos: []
---

# Reset time_bin slider to minimum and update maximum when epoch slider changes

## Overview

The `DecodedTrajectoryNapariPlotter` class uses Napari's dimension sliders to navigate through a 4D posterior volume (epoch, time_bin, xbin, ybin). Currently, when the epoch slider changes, the time_bin slider retains its previous value and maximum range. Since different epochs may have different numbers of time bins, this change will:

1. Automatically reset the time_bin slider to 0 (minimum) whenever the epoch slider is adjusted
2. Update the time_bin slider's maximum value to match the number of time bins available in the selected epoch

## Implementation

### File to modify

- [`pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`](pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py)

### Changes

#### 1. Store epoch time bin counts (lines 3693-3696)

In `build_posterior_volume`, store `epoch_time_bin_counts` as an instance variable `self.epoch_time_bin_counts` so it can be accessed in the event handler.

#### 2. Modify event handler (lines 3788-3797)

Modify the `_on_current_step_change` function in the `build_ui` method to:

1. **Track previous epoch index**: Store the current epoch index before processing the event
2. **Detect epoch changes**: Compare the new epoch index from the event with the previous one
3. **Update time_bin maximum**: When epoch changes, update `viewer.dims.range[1]` to set the maximum to the number of time bins for that epoch
4. **Reset time_bin**: When epoch changes, update `viewer.dims.current_step` to set time_bin to 0
5. **Update internal state**: Continue updating `self.curr_epoch_idx` and `self.curr_time_bin_index` as before

### Code changes

**In `build_posterior_volume` method:**

- After computing `epoch_time_bin_counts`, store it: `self.epoch_time_bin_counts = epoch_time_bin_counts`

**In `build_ui` method, `_on_current_step_change` function:**

- Store `old_epoch_idx = self.curr_epoch_idx` before processing
- Extract `new_epoch_idx = int(curr_step[0])` from the event
- If `old_epoch_idx != new_epoch_idx`:
- Get the number of time bins for the new epoch: `n_time_bins = self.epoch_time_bin_counts[new_epoch_idx]`
- Get current range for time_bin axis: `current_range = viewer.dims.range[1]` (time_bin is at index 1)
- Update the range: `viewer.dims.range[1] = (current_range[0], n_time_bins - 1, current_range[2])`
- Reset time_bin to 0: `viewer.dims.current_step = (new_epoch_idx, 0, curr_step[2], curr_step[3])`
- The recursion is safe because on the second event trigger (from our update), the epoch won't have changed, so the reset logic won't execute again

### Notes

- The time_bin axis is at index 1 in the dims.range array (original axis order: epoch=0, time_bin=1, xbin=2, ybin=3)