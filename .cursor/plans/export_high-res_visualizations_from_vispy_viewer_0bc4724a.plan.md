---
name: Export high-res visualizations from vispy viewer
overview: Add functionality to programmatically export high-resolution renderings of all epoch views from the vispy predictive decoding viewer by iterating through epoch indices and capturing screenshots.
todos:
  - id: "1"
    content: Add export_vispy_viewer_epochs function with proper imports (Path, QApplication, time, imageio/PIL)
    status: completed
  - id: "2"
    content: Implement epoch iteration loop with programmatic slider control and rendering synchronization
    status: completed
  - id: "3"
    content: Implement high-resolution canvas rendering using canvas.render() with scale factor
    status: completed
  - id: "4"
    content: Add file saving logic with proper naming convention and folder creation
    status: completed
  - id: "5"
    content: Add error handling and progress reporting
    status: completed
  - id: "6"
    content: Update render_predictive_decoding_with_vispy to expose update_epoch_display in state for easier access
    status: completed
isProject: false
---

# Plan: Enable High-Resolution Visualization Export from Vispy Viewer

## Overview

Add export functionality to the `render_predictive_decoding_with_vispy` viewer that programmatically iterates through all epoch indices, updates the display, and exports high-resolution renderings of all displayed views to an export folder.

## Implementation Details

### 1. Create Export Function

Add a new function `export_vispy_viewer_epochs` in `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)` that:

- **Parameters:**
  - `viewer_tuple`: Tuple returned from `render_predictive_decoding_with_vispy` containing `(main_window, canvas, state)`
  - `export_folder`: Path to folder where images will be saved
  - `resolution_scale`: Optional scale factor for high-res rendering (default: 2.0)
  - `export_individual_views`: Optional flag to export individual views separately (default: False)
  - `epoch_indices`: Optional list of specific epoch indices to export (default: None, exports all)
- **Functionality:**
  - Extract `main_window`, `canvas`, and `state` from viewer tuple
  - Get `num_epochs` from `state['num_epochs']`
  - Create export folder if it doesn't exist
  - Iterate through each epoch index:
    1. Programmatically set `state['epoch_slider'].setValue(epoch_idx)` (blocking signals to avoid recursive updates)
    2. Call `state['update_epoch_display'](epoch_idx)` directly or trigger via slider release
    3. Process Qt events with `QApplication.processEvents()` to ensure rendering completes
    4. Wait a small delay for rendering to stabilize
    5. Use `canvas.render(size=(width*scale, height*scale))` to capture high-res screenshot
    6. Save full canvas screenshot as `epoch_{epoch_idx:04d}_full.png`
    7. Optionally export individual views if `export_individual_views=True`

### 2. Modify Return Value

Update `render_predictive_decoding_with_vispy` to return a more accessible structure:

- Keep current return `(main_window, canvas, state)` for backward compatibility
- Add `update_epoch_display` function reference to state for easier access
- Consider returning a simple object/dict with methods for easier API

### 3. Export Implementation Details

**High-Resolution Rendering:**

- Use `canvas.render(size=(w*scale, h*scale), bgcolor='white')` where `w, h = canvas.size`
- Default scale factor of 2.0 (can be increased for very high-res)
- Handle RGBA to RGB conversion if needed
- Flip image vertically if required (vispy may return flipped)

**File Naming:**

- Full canvas: `epoch_{epoch_idx:04d}_full.png`
- Individual views (if enabled): `epoch_{epoch_idx:04d}_past.png`, `epoch_{epoch_idx:04d}_posterior.png`, etc.

**Rendering Synchronization:**

- Use `QApplication.processEvents()` multiple times to ensure all updates are processed
- Add small `time.sleep(0.1)` delay after setting epoch to allow rendering to complete
- Call `canvas.update()` before rendering to ensure latest state

### 4. Error Handling

- Validate export folder path and create if missing
- Handle invalid epoch indices gracefully
- Catch and log rendering errors for individual epochs
- Continue processing remaining epochs even if one fails

### 5. Progress Reporting

- Print progress messages: `"Exporting epoch {idx+1}/{num_epochs}..."`
- Optionally use tqdm if available for progress bar

## Code Structure

The export function will be placed after `render_predictive_decoding_with_vispy` in the same file, around line 7505. It will:

1. Import necessary modules: `Path`, `QApplication`, `time`, `numpy`, `imageio` or `PIL`
2. Validate inputs and create export directory
3. Extract viewer components from tuple
4. Loop through epochs with proper synchronization
5. Export images with appropriate naming

## Usage Example

```python
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.PredictiveDecodingComputations import render_predictive_decoding_with_vispy, export_vispy_viewer_epochs

viewer = render_predictive_decoding_with_vispy(...)
export_vispy_viewer_epochs(viewer, export_folder='./exports', resolution_scale=2.0)
```

## Files to Modify

- `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)`: Add export function after `render_predictive_decoding_with_vispy`

