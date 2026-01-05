---
name: Fix PeakPromenenceDisplay coordinate and scaling issues
overview: "Fix multiple issues in PeakPromenenceDisplay plotting methods that cause rendered data to not match input posterior heatmaps: (1) Use the original posterior data instead of slab for the surface mesh, (2) Scale probe_level and height z-coordinates to match the scaled surface, (3) Ensure coordinate system alignment between meshgrid and data arrays."
todos:
  - id: fix_surface_data_source
    content: Fix _plot_single_time_bin_pyvista to use original posterior_2d.T instead of slab, or verify slab matches coordinate system
    status: completed
  - id: scale_probe_levels
    content: Scale probe_level by z_axis_scale when creating probe level contour polylines (line 1803)
    status: completed
  - id: scale_peak_heights
    content: Scale height by z_axis_scale when creating peak sphere centers (line 1813)
    status: completed
  - id: fix_grid_method
    content: Apply same scaling fixes to plot_prominence_peaks_3d_pyvista_grid method (lines 2041, 2049)
    status: completed
    dependencies:
      - scale_probe_levels
      - scale_peak_heights
---

# Fix PeakPromenenceDisplay Coordinate and Scaling Issues

## Problem Analysis

The new `PeakPromenenceDisplay` implementations have several issues causing rendered data to not match input posterior heatmaps:

1. **Data source mismatch**: The code extracts `posterior_2d` from the original data but never uses it, instead using `slab` which is already transposed. This can cause misalignment.
2. **Z-scaling inconsistency**: 

- The surface `ZZ` is scaled by `z_axis_scale` (line 1763)
- `col_level` is scaled correctly (line 1778)
- But `probe_level` is NOT scaled when creating polylines (line 1803)
- And `height` is NOT scaled when creating peak spheres (line 1813)

3. **Coordinate system**: Need to ensure the meshgrid matches the data orientation correctly.

## Solution

### File: `pyphoplacecellanalysis/External/peak_prominence2d.py`

#### Fix 1: Use original posterior data for surface (lines 1753-1763)

- Currently extracts `posterior_2d` but doesn't use it
- Should use `posterior_2d.T` to match the transposed slab convention, OR verify slab matches expectations
- Ensure the meshgrid coordinate system aligns with the data

#### Fix 2: Scale probe_level z-coordinates (line 1803)

- Change: `probe_polyline = cls.path_to_pyvista_polyline(contour, probe_level)`
- To: `probe_polyline = cls.path_to_pyvista_polyline(contour, probe_level * z_axis_scale)`
- This matches how `col_level` is scaled on line 1778

#### Fix 3: Scale peak sphere height (line 1813)

- Change: `peak_sphere = pv.Sphere(radius=0.1, center=(center[0], center[1], height))`
- To: `peak_sphere = pv.Sphere(radius=0.1, center=(center[0], center[1], height * z_axis_scale))`
- This ensures peak centers appear at the correct z-height on the scaled surface

#### Fix 4: Apply same fixes to grid plot method (lines 2008-2050)

- The `plot_prominence_peaks_3d_pyvista_grid` method has the same issues
- Apply scaling fixes to probe_level (line 2041) and height (line 2049)