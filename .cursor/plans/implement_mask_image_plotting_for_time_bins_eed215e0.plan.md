---
name: Implement mask image plotting for time bins
overview: Complete the `_add_contours_mask_images_for_current_epoch` function to plot 2D boolean masks as Silx ImageItem objects positioned correctly for each time bin in the 3D scene.
todos:
  - id: fix_mask_handling
    content: Fix mask data extraction - convert boolean mask to float array and remove incorrect num_masks calculation
    status: completed
  - id: implement_imageitem
    content: Implement ImageItem.setData() call with correct parameters (mask_image, x_coords, y_coords, z_offset)
    status: completed
  - id: configure_imageitem
    content: Configure ImageItem properties (visibility, bounding box, translation, scale) following existing pattern
    status: completed
  - id: update_debug_message
    content: Update debug print statement to reflect mask image items instead of contour line items
    status: completed
---

## Implementation Plan

Complete the `_add_contours_mask_images_for_current_epoch` function in [`EpochTimeBinViewerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\EpochTimeBinViewerWidget.py) (lines 1849-1925) to display masks as image planes.

### Current State

- Function skeleton exists with data extraction logic
- Missing: proper ImageItem setup and data assignment
- Issues: undefined variables (`x_coords_translated`, `y_coords_translated`, `z_coords`, `values`) and incorrect API usage

### Implementation Details

1. **Fix mask data handling** (line 1886-1890):

- `a_t_bin_masks` is already a 2D array `(n_x_bins, n_y_bins)` from the list comprehension
- Remove incorrect `num_masks` calculation
- Convert boolean mask to float array (0.0/1.0) for ImageItem compatibility

2. **Implement ImageItem setup** (lines 1895-1919):

- Use Silx ImageItem API: `setData(image, x, y, z)` where:
- `image`: 2D float array of the mask
- `x`, `y`: 1D coordinate arrays (`x_coords`, `y_coords`)
- `z`: scalar z position (`effective_z_offset`)
- Apply translation using `setTranslation()` method (already handled by group items or direct translation)
- Set appropriate colormap for mask visualization (optional, can use default)
- Configure visibility and bounding box settings

3. **Key changes**:

- Replace line 1888: Remove `num_masks` calculation (not needed)
- Replace line 1896: Use `line_item.setData(mask_image, x_coords, y_coords, effective_z_offset)`
- Add mask conversion: `mask_image = a_t_bin_masks.astype(float)` before setData
- Keep existing translation/group item logic (lines 1905-1917)
- Update debug message to say "mask image items" instead of "contour line items"

### Notes

- Follows the same pattern as `_add_contours_for_current_epoch` for translation and group item handling
- Each time bin gets one ImageItem showing the mask as a 2D image plane
- Masks are positioned at `effective_z_offset` above the height map surfaces
- No changes to other functions required