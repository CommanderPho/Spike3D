---
name: Refactor posterior pane to use _helper_add_heatmap
overview: Convert the middle pane (decoded posterior display) in PredictiveDecodingDisplayWidget to use DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap instead of direct imshow calls, ensuring consistent display with past/future panes.
todos: []
---

# Refactor Posterior Pane to Use _helper_add_heatmap

## Overview

The middle pane currently uses direct `imshow` calls with manual extent calculations. This refactor will replace those with `DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap` to ensure consistent display (axes, extent, colormap handling) with the past/future panes.

## Files to Modify

- [`pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)

## Changes Required

### 1. Update `_update_posterior_plot` method (lines 4530-4573)

   - Replace direct `imshow` call for main posterior with `DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap`
   - Use `xbin_centers` and `ybin_centers` (or `xbin`/`ybin` if centers not available)
   - Map `should_use_flipped_images` to `rotate_to_vertical` parameter
   - Use `custom_image_extent` parameter to pass `self.extent`
   - Set `should_perform_reshape=False` to match past/future panes (line 4650)
   - Use appropriate `posterior_masking_value` (consider matching past/future: `1e-3`)

### 2. Update `_subfn_update_posterior_plot` nested function (lines 4971-5020)

   - Replace `imshow` calls for main posterior and time bin posteriors with `_helper_add_heatmap`
   - For main posterior: use 2D collapsed posterior with `a_time_bin_centers=None`
   - For time bin tiny heatmaps: iterate through time bins and call `_helper_add_heatmap` for each
   - Maintain the GridSpec layout (2 rows: main + time bins)
   - Keep shared color scale calculation for time bins (vmin/vmax)
   - Remove manual `.T` transpose operations - let `_helper_add_heatmap` handle reshaping via `should_perform_reshape`

### 3. Parameter Consistency

   - Main posterior: Use `time_cmap='viridis'` (keep current) or match past/future (`'Greens'`)
   - Time bins: Use `time_cmap='viridis'` with shared vmin/vmax
   - Extent: Use `custom_image_extent=self.extent` where `self.extent = (self.xbin[0], self.xbin[-1], self.ybin[0], self.ybin[-1])`
   - Handle `should_use_flipped_images` → `rotate_to_vertical` mapping
   - Set `posterior_masking_value` appropriately (default 0.0025 or match past/future 1e-3)

### 4. Import Requirements

   - Ensure `DecodedTrajectoryMatplotlibPlotter` is imported (already imported at line 4124)

## Implementation Details

### Main Posterior Plot

```python
# Instead of:
im = ax_main.imshow(active_posterior, aspect='equal', origin='lower', extent=active_extent, cmap='viridis', interpolation='none')

# Use:
heatmaps, image_extent, plots_data = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
    an_ax=ax_main,
    xbin_centers=self.xbin_centers if self.xbin_centers is not None else self.xbin,
    ybin_centers=self.ybin_centers if self.ybin_centers is not None else self.ybin,
    a_p_x_given_n=posterior_2d,
    a_time_bin_centers=None,
    rotate_to_vertical=self.should_use_flipped_images,
    custom_image_extent=self.extent,
    time_cmap='viridis',
    should_perform_reshape=False,
    posterior_masking_value=0.0025,  # or 1e-3 to match past/future
    full_posterior_opacity=1.0
)
```

### Time Bin Tiny Heatmaps

```python
# Instead of:
im_tiny = ax_tiny.imshow(time_bin_posteriors[t_bin_idx].T, ...)

# Use:
heatmaps_tiny, _, _ = DecodedTrajectoryMatplotlibPlotter._helper_add_heatmap(
    an_ax=ax_tiny,
    xbin_centers=self.xbin_centers if self.xbin_centers is not None else self.xbin,
    ybin_centers=self.ybin_centers if self.ybin_centers is not None else self.ybin,
    a_p_x_given_n=time_bin_posteriors[t_bin_idx],
    a_time_bin_centers=None,
    rotate_to_vertical=self.should_use_flipped_images,
    custom_image_extent=self.extent,
    time_cmap='viridis',
    should_perform_reshape=False,
    posterior_masking_value=0.0025,
    full_posterior_opacity=1.0
)
# Then apply vmin/vmax to the returned heatmap image
if heatmaps_tiny and len(heatmaps_tiny) > 0:
    heatmaps_tiny[0].set_clim(vmin=vmin_shared, vmax=vmax_shared)
```

### Overlay on Time Bins

The current code overlays the main posterior on time bins with low alpha. This can be handled by:

- Option A: Call `_helper_add_heatmap` twice (posterior + overlay) with different alphas
- Option B: Keep overlay logic but use `_helper_add_heatmap` for the overlay too

## Notes

- The `_helper_add_heatmap` method handles aspect ratio, origin, and extent internally
- Remove manual `.T` operations - `should_perform_reshape` controls this
- The method returns `heatmaps` (list of image handles), `image_extent`, and `plots_data`
- Maintain existing GridSpec structure for layout
- Keep axis labels, titles, and tick management as-is