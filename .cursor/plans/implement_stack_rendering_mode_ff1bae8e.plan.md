---
name: Implement stack rendering mode
overview: Replace the grid-based 3D array rendering with `get_array_as_image_stack` approach, making it the default and only rendering mode. Convert numpy array slices to PIL Images, handle ellipsis with placeholder images, and convert the final stacked image to BytesIO for display.
todos:
  - id: add_imports
    content: Add imports for get_array_as_image, get_array_as_image_stack, and PIL.Image to display_helpers.py
    status: completed
  - id: create_ellipsis_helper
    content: Create _create_ellipsis_placeholder_image() helper function to generate placeholder PIL Images for ellipsis
    status: completed
  - id: modify_heatmap_function
    content: Replace grid-based 3D array handling in _subfn_create_heatmap with stack-based approach using get_array_as_image_stack
    status: completed
  - id: update_function_signatures
    content: Add stack-related parameters (stack_offset, stack_alpha, thumbnail_height) to _subfn_create_heatmap and _subfn_display_heatmap
    status: completed
  - id: update_main_function
    content: Update single_NDArray_array_preview_with_heatmap_repr_html to extract and pass through stack parameters from kwargs
    status: completed
  - id: handle_2d_arrays
    content: Ensure 2D arrays still render correctly (either keep matplotlib or convert to single-image stack)
    status: completed
---

# Implement `get_array_as_image_stack` Rendering Mode

## Overview

Replace the current matplotlib grid-based 3D array rendering with the `get_array_as_image_stack` approach. This will use PIL Images and create a visually appealing offset-stacked visualization.

## Changes Required

### 1. Update Imports in `display_helpers.py`

- Add import for `get_array_as_image` and `get_array_as_image_stack` from `pyphocorehelpers.plotting.media_output_helpers`
- Add import for `PIL.Image` if not already present

### 2. Modify `_subfn_create_heatmap` Function

**Location:** [`display_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\pho_jupyter_preview_widget\display_helpers.py) lines 233-348

**Changes:**

- Remove the matplotlib grid-based 3D array handling (lines 276-327)
- Add new stack-based 3D array handling:
- For 3D arrays:

1. Determine slice indices with ellipsis logic (keep existing logic for first/last slices)
2. For each slice index:

- If `None` (ellipsis): Create a placeholder PIL Image with "..." text
- Otherwise: Convert numpy slice to PIL Image using `get_array_as_image()` with appropriate parameters (colormap='viridis', desired_height/width for thumbnail size)

1. Pass list of PIL Images to `get_array_as_image_stack()` with configurable parameters (offset, alpha, etc.)
2. Convert final PIL Image to BytesIO using `img.save(buf, format='png')`

- For 2D arrays: Keep existing matplotlib behavior OR convert to PIL Image and use stack (single image)
- Add parameters: `stack_offset: int = 10`, `stack_alpha: float = 0.7`, `thumbnail_height: Optional[int] = None` to function signature

### 3. Update `_subfn_display_heatmap` Function

**Location:** [`display_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\pho_jupyter_preview_widget\display_helpers.py) lines 350-362

**Changes:**

- Update call to `_subfn_create_heatmap` to pass through new stack parameters
- Extract stack parameters from kwargs if provided

### 4. Update `single_NDArray_array_preview_with_heatmap_repr_html` Function

**Location:** [`display_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\pho_jupyter_preview_widget\display_helpers.py) lines 379-441

**Changes:**

- Extract stack-related parameters from kwargs (`stack_offset`, `stack_alpha`, `thumbnail_height`) before passing to `_subfn_display_heatmap`
- Pass these parameters through the call chain

### 5. Helper Function for Ellipsis Placeholder

Create a helper function to generate a placeholder PIL Image for ellipsis:

- Function: `_create_ellipsis_placeholder_image(width: int, height: int, text: str = "...") -> Image.Image`
- Creates a PIL Image with gray background and centered text
- Should match the size of other slice images

## Implementation Details

### Ellipsis Placeholder Image

- Use PIL ImageDraw to create text on a gray background
- Match dimensions of actual slice images
- Include text showing number of hidden slices if possible

### Numpy to PIL Conversion

- Use `get_array_as_image()` with:
- `colormap='viridis'` (matching current implementation)
- `desired_height` from kwargs or default thumbnail size
- `skip_img_normalization=True` for performance
- `export_kind=HeatmapExportKind.COLORMAPPED`

### Stack Parameters

- Default `offset=10` for subtle stacking effect
- Default `single_image_alpha_level=0.7` for good visibility with transparency
- Disable transparency by default (`single_image_alpha_level=1.0`) for better performance unless explicitly requested

### Error Handling

- Handle cases where `get_array_as_image()` fails for individual slices
- Fall back gracefully if PIL Image operations fail
- Maintain existing error handling structure

## Testing Considerations

- Test with 3D array (41, 63, 4) - should show 4 stacked images
- Test with large 3D array exceeding max_slices - should show first, ellipsis placeholder, and last slices
- Test with 2D array - should work as before or use single-image stack
- Verify performance with reasonable number of slices