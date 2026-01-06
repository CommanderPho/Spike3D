---
name: Add Napari Console Logging
overview: Add a helper method to output messages to both stdout and Napari Console, then replace all print statements in DecodedTrajectoryNapariPlotter with this helper method.
todos:
  - id: add_log_helper
    content: Add _log_to_console() helper method to DecodedTrajectoryNapariPlotter class that outputs to both stdout and Napari Console
    status: completed
  - id: replace_prints_add_peak_contours
    content: Replace print statements in add_peak_contours_layer() method with _log_to_console() calls
    status: completed
    dependencies:
      - add_log_helper
  - id: replace_prints_nested_functions
    content: Replace print statements in nested functions (extract_contours_from_peaks_dict, update_contours_for_current_indices, _on_current_step_change_contours) with _log_to_console() calls, ensuring proper self access
    status: completed
    dependencies:
      - add_log_helper
---

# Add Napari Console Logging to DecodedTrajectoryNapariPlotter

## Overview

All `print()` statements in the `DecodedTrajectoryNapariPlotter` class (lines 3651-4228) should also output to the Napari Console. This will be accomplished by:

1. Creating a helper method that outputs to both stdout and Napari Console
2. Replacing all print statements in the class with calls to this helper

## Implementation Details

### 1. Add Helper Method

Add a new method `_log_to_console()` to the `DecodedTrajectoryNapariPlotter` class that:

- Takes a message string as input
- Outputs to stdout via `print()` (maintains existing behavior)
- Also outputs to Napari Console if viewer exists and console is available
- Handles cases where viewer/console might not be available gracefully

The method should be placed early in the class, after `build_ui()` method (around line 3810).

### 2. Replace Print Statements

Replace all `print()` calls in the following methods with `self._log_to_console()`:

- `add_peak_contours_layer()` - lines 4036, 4089, 4096, 4107, 4109, 4111, 4112
- `extract_contours_from_peaks_dict()` (nested function) - lines 4089, 4096, 4107, 4109, 4111, 4112
- `update_contours_for_current_indices()` (nested function) - lines 4118, 4120, 4124, 4127, 4128, 4129, 4134, 4138, 4141, 4142, 4146, 4151, 4153, 4155, 4157
- `_on_current_step_change_contours()` (nested function) - lines 4169, 4171, 4174, 4177, 4181, 4184, 4221

**Note**: For nested functions that don't have access to `self`, we'll need to either:

- Pass `self` as a parameter to the nested functions, OR
- Capture `self._log_to_console` in a closure variable before defining the nested functions

### 3. Napari Console Access

The Napari console can be accessed via `viewer.window._qt_viewer.dockConsole` (private API). We'll:

- Check if viewer exists
- Check if console widget is available
- Use the console's IPython kernel to execute print statements, or write directly to console output if possible
- Fall back gracefully if console is not available

## Files to Modify

- `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py` (lines 3651-4228)

## Implementation Approach

1. Add `_log_to_console()` method that safely accesses Napari console