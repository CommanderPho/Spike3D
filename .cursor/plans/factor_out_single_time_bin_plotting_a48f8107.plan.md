---
name: Factor out single time bin plotting
overview: Extract the core plotting logic from `_update_plot_for_time_bin` into a new standalone method that plots a single time bin without slider infrastructure, then refactor the original method to use it.
todos: []
---

#Factor Out Core Plotting Code for Single Time Bin

## Overview

Extract the core plotting logic from the nested `_update_plot_for_time_bin` function into a new standalone class method that can plot a single time bin independently, without the slider infrastructure.

## Changes

### 1. Create New Method: `_plot_single_time_bin_pyvista`

- **Location**: [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py) (before `plot_prominence_peaks_3d_pyvista`)
- **Purpose**: Core plotting logic for a single time bin
- **Parameters**:
- `plotter`: PyVista plotter instance (required)
- `posterior_peaks_result`: DynamicParameters with results
- `p_x_given_n_list`: List of posterior arrays
- `epoch_idx`: int
- `time_bin_idx`: int
- `xx`, `yy`: bin centers
- `show_col_contours`: bool
- `show_probe_level_contours`: bool
- `probe_level_to_show`: float or None
- `opacity`: float
- `cmap`: str
- `z_axis_scale`: float
- `show_scalar_bar`: bool (default True) - controls scalar bar visibility
- **Returns**: List of mesh actors created
- **Logic**: 
- Extract result for the specified epoch/time_bin
- Create posterior surface mesh
- Add col contours if enabled
- Add probe level contours if enabled
- Add peak centers as spheres
- Return list of actors for potential cleanup

### 2. Refactor `plot_prominence_peaks_3d_pyvista`

- **Location**: [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py) (lines 1701-1871)
- **Changes**:
- Replace `_update_plot_for_time_bin` nested function with a wrapper that:
    - Clears existing mesh actors
    - Calls `_plot_single_time_bin_pyvista` with `show_scalar_bar=first_update`
    - Updates `first_update` flag
    - Stores returned actors in `mesh_actors` list
- Keep all existing functionality (slider, plotter setup, etc.)
- Maintain backward compatibility (same return signature)

## Implementation Details

The new `_plot_single_time_bin_pyvista` method will contain the plotting logic from lines 1766-1838, but:

- Accepts `plotter` as a parameter instead of using closure
- Accepts `show_scalar_bar` parameter to control scalar bar visibility