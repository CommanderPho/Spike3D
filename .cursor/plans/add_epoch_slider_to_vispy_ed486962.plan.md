---
name: Add Epoch Slider to Vispy
overview: Add a Qt slider control below the vispy canvas in `render_predictive_decoding_with_vispy` that allows users to navigate epochs. The slider only updates the display when released (not during dragging) to avoid expensive redraws.
todos:
  - id: wrap-canvas-in-window
    content: Create QMainWindow with QVBoxLayout to host vispy canvas and slider controls
    status: completed
  - id: add-slider-controls
    content: Add epoch slider with label and value display below the canvas
    status: completed
  - id: connect-slider-events
    content: Connect valueChanged for label update and sliderReleased for display update
    status: completed
  - id: update-return-value
    content: Update return statement to include main_window and state references
    status: completed
isProject: false
---

# Add Epoch Slider to Vispy Renderer

## Overview

Add a Qt horizontal slider below the vispy canvas to control the epoch index, matching the behavior in `PredictiveDecodingDisplayWidget` (lines 5200-5215). The slider updates a label while dragging, but only triggers the expensive display update when the slider is released.

## Implementation

### 1. Wrap Canvas in QMainWindow

In [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) around line 6561-6563, after creating the canvas:

- Import Qt modules (already available via `from qtpy import QtWidgets, QtCore`)
- Create a `QMainWindow` to host the vispy canvas
- Create a `QVBoxLayout` in a central widget
- Add `canvas.native` (the Qt widget for the vispy canvas) to the layout
- Create a bottom row with the slider controls

### 2. Add Slider Widget (similar to lines 5198-5215)

Create a horizontal layout at the bottom containing:

- Label: "Epoch:"
- `QSlider` (horizontal) with range 0 to num_epochs-1
- Value label showing current slider position

### 3. Connect Slider Events

Two signal connections matching the reference implementation:

- `valueChanged` -> Updates the label text only (no display update while dragging)
- `sliderReleased` -> Calls `update_epoch_display()` to update the visualization

### 4. Store References in State

Add to the state dictionary:

- `epoch_slider`: The QSlider widget
- `epoch_value_label`: The QLabel showing current value
- `main_window`: The QMainWindow containing everything

### 5. Update Return Value

Return a dict containing both `canvas` and `main_window` (or just `main_window` since the canvas is accessible from it).

## Key Code Changes

### Location: Line ~6561 (canvas creation)

**Before:**

```python
canvas = scene.SceneCanvas(keys='interactive', show=True, size=(1400, 900), title='...')
```

**After:**

```python
from qtpy import QtWidgets, QtCore

canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1400, 900), title='...')

# Create main window to host canvas and slider
main_window = QtWidgets.QMainWindow()
main_window.setWindowTitle('Predictive Decoding Display - Vispy')
central_widget = QtWidgets.QWidget()
main_layout = QtWidgets.QVBoxLayout(central_widget)

# Add vispy canvas to layout
main_layout.addWidget(canvas.native, stretch=1)

# Create slider controls
slider_widget = QtWidgets.QWidget()
slider_layout = QtWidgets.QHBoxLayout(slider_widget)
slider_label = QtWidgets.QLabel("Epoch:")
epoch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
epoch_slider.setMinimum(0)
epoch_slider.setMaximum(max(0, num_epochs - 1))
epoch_slider.setValue(min(active_epoch_idx, num_epochs - 1))
epoch_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
epoch_slider.setTickInterval(1)
epoch_value_label = QtWidgets.QLabel(f"{active_epoch_idx}")

slider_layout.addWidget(slider_label)
slider_layout.addWidget(epoch_slider, stretch=1)
slider_layout.addWidget(epoch_value_label)
main_layout.addWidget(slider_widget)

main_window.setCentralWidget(central_widget)
main_window.resize(1400, 950)
main_window.show()
```

### Location: After state dict (line ~6640)

Add slider references and event handlers:

```python
state['epoch_slider'] = epoch_slider
state['epoch_value_label'] = epoch_value_label
state['main_window'] = main_window

def on_slider_value_changed(value):
    state['epoch_value_label'].setText(f"{value}")

def on_slider_released():
    value = state['epoch_slider'].value()
    update_epoch_display(value)

epoch_slider.valueChanged.connect(on_slider_value_changed)
epoch_slider.sliderReleased.connect(on_slider_released)
```

### Location: Return statement (line ~7157)

**Before:**

```python
return canvas
```

**After:**

```python
return main_window, canvas, state
```

