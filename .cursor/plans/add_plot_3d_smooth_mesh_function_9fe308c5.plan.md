---
name: Add plot_3d_smooth_mesh function
overview: Implement a new `plot_3d_smooth_mesh` function in `graphs.py` that renders data as a smooth mesh similar to placefield tuning curves, following the same pattern as `plot_3d_binned_bars` and `plot_3d_stem_points`.
todos:
  - id: "1"
    content: Add `plot_3d_smooth_mesh` function to `graphs.py` after `plot_3d_stem_points`, implementing smooth mesh rendering with `smooth_shading=True` and opacity mapping
    status: completed
  - id: "2"
    content: Update import statements in `decoder_plotting_mixins.py` to include `plot_3d_smooth_mesh` in the relevant import locations
    status: completed
    dependencies:
      - "1"
---

# Implementation Plan: Add `plot_3d_smooth_mesh` Function

## Overview

Create a new plotting function `plot_3d_smooth_mesh` that renders 3D data as a smooth mesh surface, similar to how placefield tuning curves are rendered in `plot_placefields2D`. This function will be analogous to `plot_3d_binned_bars` and `plot_3d_stem_points` but with smooth shading enabled.

## Key Files to Modify

### 1. `pyphoplacecellanalysis/Pho3D/PyVista/graphs.py`

- Add the new `plot_3d_smooth_mesh` function after `plot_3d_stem_points` (around line 298)
- The function should follow the same signature pattern as `plot_3d_binned_bars` and `plot_3d_stem_points`:
     ```python
                              def plot_3d_smooth_mesh(p, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold: float=None, **kwargs):
     ```




- Key implementation details:
    - Use `prepare_binned_data_for_3d_bars` helper (like `plot_3d_binned_bars` does) to prepare the data
    - Create a `pv.StructuredGrid` from the prepared data
    - Set `smooth_shading=True` (unlike `plot_3d_binned_bars` which uses `False`)
    - Use `opacity='sigmoid'` or similar for smooth opacity transitions (like placefield tuning curves)
    - Support the same `drop_below_threshold` parameter
    - Return the same structure: `plotActors, data_dict` with the plot name and mesh data

### 2. `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`

- Update the import statement (around line 98) to include `plot_3d_smooth_mesh`
- Update the import statement (around line 3735) in `add_decoded_posterior_bars` to include `plot_3d_smooth_mesh`
- The function should work with the existing `active_plot_fn` parameter system, so no changes needed to the calling code

## Implementation Details

### Function Signature

- Follow the exact same pattern as `plot_3d_binned_bars`:
- Parameters: `p, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold=None, **kwargs`
- Uses `xbin` and `ybin` (edges) like `plot_3d_binned_bars`, not centers like `plot_3d_stem_points`

### Rendering Style

- Use `smooth_shading=True` (key difference from `plot_3d_binned_bars`)
- Use `opacity='sigmoid'` or `opacity='linear'` for smooth opacity transitions based on elevation
- Keep `show_edges=True` with configurable edge color (default 'k' for black)
- Use `nan_opacity=0.0` to hide NaN values
- Set `scalars='Elevation'` to use elevation for opacity mapping

### Data Preparation

- Use `prepare_binned_data_for_3d_bars(xbin, ybin, data, mask2d)` helper function
- Apply `drop_below_threshold` masking if provided
- Create meshgrid from modified bins
- Create `pv.StructuredGrid` with elevation data

### Return Structure

- Return `plotActors` dict with structure: `{plot_name: {'main': plotActor}}`
- Return `data_dict` with structure: `{plot_name: {'name': plot_name, 'grid': mesh, 'twoDimGrid_x': ..., 'twoDimGrid_y': ..., 'active_data': ...}}`