---
name: Add Peak Prominence to Decoded Posteriors
overview: Extend the existing peak prominence plotting functionality from PyVista placefield tuning curves to also work with decoded posteriors. The implementation will render peak prominence contours, bounding boxes, text labels, and peak points for the currently displayed posterior time bin, automatically matching the posterior's visibility.
todos:
  - id: add_posterior_data_builder
    content: Add _build_pyvista_single_posterior_prominence_result_data() function to extract peak data from PosteriorPeaksPeakProminence2dResult for a specific (epoch_idx, time_bin_idx)
    status: completed
  - id: add_posterior_plotter
    content: Add _plot_pyvista_single_posterior_prominence_result() function to render peak contours/boxes/text/points on a posterior PyVista mesh
    status: completed
    dependencies:
      - add_posterior_data_builder
  - id: add_posterior_renderer
    content: Add _render_posterior_peak_prominence_2d_results_on_pyvista_plotter() function to orchestrate data building and plotting for posteriors
    status: completed
    dependencies:
      - add_posterior_plotter
  - id: add_public_api
    content: Add render_posterior_peak_prominence_2d_results_on_pyvista_plotter() public convenience function
    status: completed
    dependencies:
      - add_posterior_renderer
  - id: add_plotter_fields
    content: Add peak_prominence_result, peak_prominence_actors, peak_prominence_data fields to DecodedTrajectoryPyVistaPlotter class
    status: completed
  - id: integrate_update_method
    content: Modify perform_update_plot_single_epoch_time_bin() to render peak prominence when peak_prominence_result is set
    status: completed
    dependencies:
      - add_plotter_fields
      - add_posterior_renderer
  - id: integrate_clear_method
    content: Modify perform_clear_existing_decoded_trajectory_plots() to clear peak prominence actors
    status: completed
    dependencies:
      - add_plotter_fields
  - id: add_setter_method
    content: Add set_peak_prominence_result() method to DecodedTrajectoryPyVistaPlotter to set and trigger re-render
    status: completed
    dependencies:
      - add_plotter_fields
      - integrate_update_method
---

# Add Peak Prominence Plotting to Decoded Posteriors in PyVista

## Overview

Extend the peak prominence plotting chain in `peak_prominences.py` to work with decoded posteriors rendered by `DecodedTrajectoryPyVistaPlotter`. The current implementation works with placefield tuning curves (keyed by `neuron_id`), and needs to be adapted for posteriors (keyed by `(epoch_idx, time_bin_idx)` tuples).

## Key Differences to Handle

- **Data structure**: `PosteriorPeaksPeakProminence2dResult.results` is keyed by `(epoch_idx, time_bin_idx)` tuples instead of `neuron_id` integers
- **PyVista mesh access**: Need to get the PyVista mesh from the currently displayed posterior, not from `ipcDataExplorer.plots_data.tuningCurvePlotData[neuron_id]`
- **Integration point**: Need to hook into `DecodedTrajectoryPyVistaPlotter.perform_update_plot_single_epoch_time_bin()` to update peaks when time bin changes
- **Z-scaling**: Posteriors use different z-scaling than tuning curves (need to compute from posterior height)

## Implementation Plan

### 1. Create Posterior-Specific Data Building Function

**File**: `pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py`

- Add `_build_pyvista_single_posterior_prominence_result_data(epoch_idx, time_bin_idx, a_result, ...)` 
- Similar to `_build_pyvista_single_neuron_prominence_result_data` but:
- Takes `(epoch_idx, time_bin_idx)` tuple instead of `neuron_id`
- Accesses `PosteriorPeaksPeakProminence2dResult.results[(epoch_idx, time_bin_idx)]` instead of `active_peak_prominence_2d_results.results[neuron_id]`
- Returns same structure: `peak_locations, prominence_array, peak_labels, peak_levels, peak_level_bboxes`

### 2. Create Posterior-Specific Plotting Function

**File**: `pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py`

- Add `_plot_pyvista_single_posterior_prominence_result(plotter, posterior_pdata, peak_locations, peak_labels, peak_levels, peak_level_bboxes, ...)`
- Similar to `_plot_pyvista_single_neuron_prominence_result` but:
- Takes `plotter` (PyVista plotter) and `posterior_pdata` (PyVista mesh) directly instead of `ipcDataExplorer`
- Uses `posterior_pdata.contour()` to create contours (same as tuning curves)
- Computes z-scaling from the posterior mesh height instead of from `ipcDataExplorer.params.zScalingFactor`
- Returns same structure: `all_peaks_actors, all_peaks_data`

### 3. Create Main Rendering Function for Posteriors

**File**: `pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py`

- Add `_render_posterior_peak_prominence_2d_results_on_pyvista_plotter(plotter, posterior_pdata, peak_prominence_result, epoch_idx, time_bin_idx, ...)`
- Orchestrates the data building and plotting:
- Calls `_build_pyvista_single_posterior_prominence_result_data` to extract peak data
- Computes z-scaling from posterior mesh
- Calls `_plot_pyvista_single_posterior_prominence_result` to render
- Returns `all_peaks_data, all_peaks_actors`

### 4. Integrate into DecodedTrajectoryPyVistaPlotter

**File**: `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Add fields to `DecodedTrajectoryPyVistaPlotter`:
- `peak_prominence_result: Optional[PosteriorPeaksPeakProminence2dResult] = field(default=None)`
- `peak_prominence_actors = field(default=None)`
- `peak_prominence_data = field(default=None)`
- Modify `perform_update_plot_single_epoch_time_bin()`:
- After plotting the posterior, check if `peak_prominence_result` is set
- If set, get the PyVista mesh from `self.data_dict` (the posterior mesh)
- Call `_render_posterior_peak_prominence_2d_results_on_pyvista_plotter`
- Store results in `self.peak_prominence_actors` and `self.peak_prominence_data`
- Set visibility to match the posterior visibility
- Modify `perform_clear_existing_decoded_trajectory_plots()`:
- Clear peak prominence actors before clearing posterior actors
- Remove peak prominence actors from plotter
- Add method `set_peak_prominence_result(peak_prominence_result: PosteriorPeaksPeakProminence2dResult)`:
- Sets the peak prominence result
- Triggers re-render if plotter is already built

### 5. Add Convenience Function

**File**: `pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py`

- Add `render_posterior_peak_prominence_2d_results_on_pyvista_plotter(decoded_trajectory_plotter, peak_prominence_result, ...)`:
- High-level function that sets the peak prominence result on the plotter
- Calls `decoded_trajectory_plotter.set_peak_prominence_result()`
- Similar API to `render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter`

## Files to Modify

1. **`pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py`**

- Add `_build_pyvista_single_posterior_prominence_result_data()`
- Add `_plot_pyvista_single_posterior_prominence_result()`
- Add `_render_posterior_peak_prominence_2d_results_on_pyvista_plotter()`
- Add `render_posterior_peak_prominence_2d_results_on_pyvista_plotter()` (public API)

2. **`pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`**

- Add fields to `DecodedTrajectoryPyVistaPlotter` for peak prominence
- Modify `perform_update_plot_single_epoch_time_bin()` to render peaks
- Modify `perform_clear_existing_decoded_trajectory_plots()` to clear peaks
- Add `set_peak_prominence_result()` method

## Usage Example

```python
from pyphoplacecellanalysis.Pho3D.PyVista.peak_prominences import render_posterior_peak_prominence_2d_results_on_pyvista_plotter

# After creating DecodedTrajectoryPyVistaPlotter and building UI
a_decoded_trajectory_pyvista_plotter.build_ui()

# Add peak prominence results
render_posterior_peak_prominence_2d_results_on_pyvista_plotter(
    a_decoded_trajectory_pyvista_plotter,
    all_posteriors_pf_promenence_result_obj,
    promenence_plot_threshold=0.2,
    included_level_indicies=[1],
    include_contour_bounding_box=False,
    include_text_labels=False
)
```



## Notes

- Peak prominence actors will automatically update when sliders change (via `perform_update_plot_single_epoch_time_bin`)