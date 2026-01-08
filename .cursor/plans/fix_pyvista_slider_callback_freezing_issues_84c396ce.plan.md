---
name: Fix PyVista slider callback freezing issues
overview: Fix critical callback recursion and double-update issues in DecodedTrajectoryPyVistaPlotter that cause notebook freezing when moving sliders. Add callback blocking, remove problematic double-update code, add execution guards, and improve cleanup.
todos:
  - id: add-callback-blocking
    content: Add callback blocking mechanism in update_ui() to prevent recursion when programmatically setting slider values
    status: pending
  - id: remove-double-update
    content: Remove problematic double-update code at lines 3307-3309 that uses potentially stale data_dict
    status: pending
  - id: add-execution-guards
    content: Add _update_in_progress guards to prevent nested callback execution in on_update_slider_epoch_idx, on_update_slider_epoch_time_bin, and perform_update_plot methods
    status: pending
  - id: improve-cleanup
    content: Enhance perform_clear_existing_decoded_trajectory_plots() to ensure proper cleanup of actors and data_dict before new updates
    status: pending
  - id: verify-callback-registration
    content: Verify build_ui() properly prevents duplicate slider creation and callback registration
    status: pending
---

# Fix PyVista Slider Callback Freezing Issues

## Problem Summary

The `DecodedTrajectoryPyVistaPlotter` freezes the notebook when sliders are moved, especially on the second interaction. Root causes:

1. **Callback recursion**: `update_ui()` programmatically sets slider values, triggering nested callbacks
2. **Double update**: Lines 3307-3309 attempt a second update using potentially stale `data_dict`
3. **No execution guards**: Multiple callbacks can execute simultaneously
4. **Stale state**: `data_dict` may contain references from previous updates

## Implementation Plan

### 1. Add callback blocking mechanism

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Add instance variable `_updating_slider_programmatically: bool = False` to `DecodedTrajectoryPyVistaPlotter` class (around line 3187)
- Modify `update_ui()` (line 3270) to:
- Set `_updating_slider_programmatically = True` before `SetValue()`
- Temporarily disable slider callbacks using `RemoveAllObservers()` / re-add after
- Or check the flag in callbacks to skip execution when programmatic update is in progress
- Reset flag after `SetValue()` completes

### 2. Remove problematic double-update code

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Remove or comment out lines 3305-3309 (the `update_plot_fn` code block)
- This code attempts a second update using `data_dict` which may be stale or from previous state
- The main update at lines 3300/3303 already handles the plotting

### 3. Add execution guards to prevent nested callbacks

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Add instance variable `_update_in_progress: bool = False` to track active updates
- Modify `on_update_slider_epoch_idx()` (line 3286) to:
- Check `_update_in_progress` at start, return early if True
- Set `_update_in_progress = True` at start
- Reset to `False` in finally block or at end
- Apply same guard to `on_update_slider_epoch_time_bin()` (line 3313)
- Apply same guard to `perform_update_plot_single_epoch_time_bin()` (line 3322) and `perform_update_plot_epoch_time_bin_range()` (line 3339)

### 4. Improve actor cleanup to prevent stale state

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Enhance `perform_clear_existing_decoded_trajectory_plots()` (line 3357):
- Ensure `data_dict` is cleared BEFORE new actors are created (already done, but verify order)
- Add explicit cleanup of any update functions stored in `data_dict`
- Ensure `plotActors_CenterLabels` are properly removed from plotter before clearing dict

### 5. Fix callback registration to prevent duplicates

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Ensure `build_ui()` (line 3190) only creates sliders once (already has `if self.slider_epoch is None` checks)
- Add cleanup in `perform_clear_existing_decoded_trajectory_plots()` if needed to prevent duplicate observers

## Implementation Details

### Callback Blocking Pattern

Use a context manager or flag-based approach similar to:

```python
def update_ui(self):
    if self.slider_epoch_time_bin is not None:
        # Block callbacks during programmatic update
        self._updating_slider_programmatically = True
        try:
            self.slider_epoch_time_bin.GetRepresentation().SetMaximumValue(...)
            self.slider_epoch_time_bin.GetRepresentation().SetValue(...)
        finally:
            self._updating_slider_programmatically = False
```



### Execution Guard Pattern

```python
def on_update_slider_epoch_idx(self, value: int):
    if self._update_in_progress:
        return  # Prevent nested execution
    self._update_in_progress = True
    try:
        # ... existing update code ...
    finally:
        self._update_in_progress = False
```



## Testing

After implementation, verify:

1. Moving epoch slider doesn't freeze
2. Moving time_bin slider doesn't freeze  
3. Multiple rapid slider movements don't cause issues