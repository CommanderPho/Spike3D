---
name: Replace PyVista sliders with Qt sliders
overview: Replace the existing PyVista VTK slider widgets with Qt QSlider widgets in a thin horizontal bar at the bottom of the PyVista plotter window. The sliders will be embedded in the plotter's window layout and will update the plot when moved. The playback checkbox will be reimplemented as a Qt checkbox in the same bar.
todos:
  - id: add_qt_fields
    content: Add Qt widget fields to DecodedTrajectoryPyVistaPlotter class (qt_slider_epoch, qt_slider_epoch_time_bin, labels, checkbox, bar widget)
    status: completed
  - id: create_slider_bar_method
    content: Create _build_qt_slider_bar() method that builds the horizontal slider bar widget with labels, sliders, and checkbox
    status: completed
    dependencies:
      - add_qt_fields
  - id: modify_build_ui
    content: Modify build_ui() to call _build_qt_slider_bar() instead of creating PyVista sliders, remove all VTK slider code
    status: completed
    dependencies:
      - create_slider_bar_method
  - id: update_ui_methods
    content: Update update_ui() and perform_programmatic_slider_epoch_update() to work with Qt sliders instead of VTK
    status: in_progress
    dependencies:
      - modify_build_ui
  - id: connect_signals
    content: Connect Qt slider valueChanged signals to existing callback methods, ensuring callback blocking logic still works
    status: pending
    dependencies:
      - modify_build_ui
  - id: reimplement_playback
    content: Reimplement playback checkbox functionality using Qt QCheckBox, connect to PhoInteractivePlotter or implement playback directly
    status: pending
    dependencies:
      - create_slider_bar_method
---

# Replace PyVista Sliders with Qt Sliders in Bottom Bar

## Overview

Replace the existing PyVista VTK slider widgets with Qt `QSlider` widgets embedded in a thin horizontal bar at the bottom of the PyVista plotter window. The playback checkbox will be reimplemented as a Qt `QCheckBox` in the same bar.

## Architecture

The implementation will:

1. Access the PyVista plotter's window (`self.p.app_window`) which is a Qt widget
2. Create a horizontal layout bar at the bottom containing:

- "Epoch Idx" label + QSlider + value label
- "Timebin IDX" label + QSlider + value label  
- Playback QCheckBox

3. Remove all existing PyVista slider widget code
4. Connect Qt slider signals to existing update methods

## Implementation Details

### Files to Modify

**`pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`**

1. **Modify `build_ui()` method** (lines ~3201-3320):

- Remove all `self.p.add_slider_widget()` calls
- Remove VTK text actor creation code
- Add new method `_build_qt_slider_bar()` that:
    - Accesses `self.p.app_window` (the Qt window)
    - Gets or creates a bottom layout/widget
    - Creates Qt sliders and labels
    - Connects signals to existing callback methods

2. **Add new fields** to `DecodedTrajectoryPyVistaPlotter` class (around line 3177):

- `qt_slider_epoch: Optional[QtWidgets.QSlider] = field(default=None)`
- `qt_slider_epoch_time_bin: Optional[QtWidgets.QSlider] = field(default=None)`
- `qt_slider_epoch_label: Optional[QtWidgets.QLabel] = field(default=None)`
- `qt_slider_timebin_label: Optional[QtWidgets.QLabel] = field(default=None)`
- `qt_playback_checkbox: Optional[QtWidgets.QCheckBox] = field(default=None)`
- `qt_slider_bar_widget: Optional[QtWidgets.QWidget] = field(default=None)`

3. **Add new method `_build_qt_slider_bar()`**:

- Check if `self.p.app_window` exists (for BackgroundPlotter)
- Create a thin horizontal QWidget bar
- Use QHBoxLayout to arrange: labels, sliders, value displays, checkbox
- Set fixed height (e.g., 40-50px) for the bar
- Add bar to bottom of window layout
- Connect slider `valueChanged` signals to existing update methods
- Connect checkbox to playback functionality

4. **Modify `update_ui()` method** (line ~3284):

- Update Qt slider ranges instead of VTK slider representation
- Update Qt slider values programmatically

5. **Modify `perform_programmatic_slider_epoch_update()`** (line ~3297):

- Update Qt slider value instead of VTK slider

6. **Update callback methods**:

- `_on_slider_callback_epoch_idx`: Connect to Qt slider's `valueChanged` signal
- `_on_slider_callback_epoch_time_bin`: Connect to Qt slider's `valueChanged` signal
- Ensure callback blocking logic still works with Qt sliders

7. **Handle playback checkbox**:

- Reimplement playback functionality using Qt QCheckBox
- Connect to `PhoInteractivePlotter` or implement playback logic directly

### Key Implementation Points

- **Window Access**: Use `self.p.app_window` for BackgroundPlotter (from pyvistaqt)
- **Layout Management**: Access the window's central widget layout or create a wrapper layout
- **Signal Connections**: Use Qt's signal/slot mechanism instead of VTK callbacks
- **Thread Safety**: Ensure slider updates happen on the main Qt thread
- **Backward Compatibility**: Handle cases where plotter might not be a BackgroundPlotter

### Dependencies

- `qtpy.QtWidgets` (already imported) for QSlider, QLabel, QCheckBox, QHBoxLayout, QWidget
- Access to `self.p.app_window` from pyvistaqt BackgroundPlotter

## Testing Considerations

- Verify sliders appear at bottom of window
- Test slider value updates trigger plot updates
- Test playback checkbox functionality
- Ensure no VTK slider artifacts remain