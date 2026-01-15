---
name: Refactor print_helpers to use pho_jupyter_preview_widget implementations
overview: Replace the duplicate, less complete implementations in print_helpers.py (lines 1904-2157) with imports from the more complete pho_jupyter_preview_widget versions. Update all references in preferences_helpers.py to use the new import paths.
todos: []
---

# Refactor print_helpers to use pho_jupyter_preview_widget implementations

## Overview

The `print_helpers.py` file contains duplicate implementations of array preview functions that are less complete than the versions in `pho_jupyter_preview_widget/display_helpers.py`. This plan replaces the duplicate code with imports from the more complete implementations.

## Key Differences Identified

### Functions to Replace:

1. **`array_preview_with_shape`** (lines 1906-1928) - Identical in both, safe to replace
2. **`array_preview_with_graphical_shape_repr_html`** (lines 1931-1960) - `pho_jupyter_preview_widget` version uses custom `array_repr_html` instead of Dask
3. **`array_preview_with_heatmap_repr_html`** (lines 2045-2155) - `pho_jupyter_preview_widget` version:

   - Handles lists of arrays (not just single arrays)
   - Uses custom `array_repr_html` instead of Dask's `da.array(arr)._repr_html_()`
   - Uses `smart_array2string` instead of `np.array2string`

4. **`_subfn_create_heatmap`** (lines 1967-2029) - Identical, can be removed
5. **`_subfn_display_heatmap`** (lines 2032-2043) - Identical, can be removed

### Improvements in pho_jupyter_preview_widget versions:

- Better shape visualization (custom `array_repr_html` vs Dask)
- Handles lists of arrays
- Better string formatting (`smart_array2string` vs `np.array2string`)
- More complete implementation overall

## Implementation Steps

### 1. Replace functions in [print_helpers.py](H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\print_helpers.py)

**Lines 1904-2157**: Replace all duplicate implementations with imports:

```python
# Import from pho_jupyter_preview_widget instead of duplicating
from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import (
    array_preview_with_shape,
    array_preview_with_graphical_shape_repr_html,
    array_preview_with_heatmap_repr_html,
    _subfn_create_heatmap,
    _subfn_display_heatmap,
)
```

**Note**: The helper functions `_subfn_create_heatmap` and `_subfn_display_heatmap` are used internally by `array_preview_with_heatmap_repr_html`, so they may not need to be exported. Check if they're used elsewhere.

### 2. Update [preferences_helpers.py](H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\preferences_helpers.py)

**Lines 79, 88, 98**: Update imports to use `pho_jupyter_preview_widget` instead of `print_helpers`:

- Line 79: `from pyphocorehelpers.print_helpers import array_preview_with_graphical_shape_repr_html`

→ `from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_graphical_shape_repr_html`

- Line 88: Same change

- Line 98: `from pyphocorehelpers.print_helpers import array_preview_with_heatmap_repr_html`

→ `from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_heatmap_repr_html`

### 3. Verify Dependencies

Check if `print_helpers.py` imports `dask.array as da` only for these functions. If so, the import can potentially be removed if not used elsewhere.

## Considerations

- The `pho_jupyter_preview_widget` version preserves the TODO comment about `get_array_as_image_stack` for 3D arrays (as requested)
- All function signatures match, so this should be a drop-in replacement
- The `pho_jupyter_preview_widget` versions are more feature-complete (handle lists, better formatting)