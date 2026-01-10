---
name: Add efficient peak computation function
overview: Create a new function `_compute_single_posterior_slab_efficient` that duplicates `_compute_single_posterior_slab` but uses morphological reconstruction instead of the expensive `getProminence` contour-based method. The new function will find all peaks cheaply, compute prominence efficiently, filter to top N peaks, and only compute contours for those top N peaks.
todos:
  - id: "1"
    content: Duplicate _compute_single_posterior_slab function to create _compute_single_posterior_slab_efficient
    status: completed
  - id: "2"
    content: Add max_n_considered_peaks parameter to function signature
    status: completed
  - id: "3"
    content: Replace compute_prominence_contours call with morphological reconstruction approach (find peaks, compute prominence, filter, sort, take top N)
    status: completed
  - id: "4"
    content: Convert peak coordinates to peaks_dict format with proper spatial coordinate conversion
    status: completed
  - id: "5"
    content: Create simplified id_map, prominence_map, and parent_map for compatibility
    status: completed
  - id: "6"
    content: Test that return format matches original function exactly
    status: completed
---

## Implementation Plan

### 1. Create New Function `_compute_single_posterior_slab_efficient`

**Location**: Insert after `_compute_single_posterior_slab` (after line 1063)

**Key Changes**:

- Add `max_n_considered_peaks: int = 3` parameter to function signature
- Replace `compute_prominence_contours` call (lines 925-934) with efficient morphological reconstruction approach
- Keep all other logic identical for drop-in compatibility

### 2. Efficient Peak Finding Implementation

Replace the `compute_prominence_contours` call with:

```python
# Step 1: Find all local maxima using morphological operations (fast, O(N))
neighborhood = ndimage.generate_binary_structure(2, 2)
local_max = (slab == ndimage.maximum_filter(slab, footprint=neighborhood))
local_max &= (slab > np.nanmin(slab))

# Get peak coordinates and heights
peak_coords = np.column_stack(np.nonzero(local_max))  # (n_peaks, 2) as (row, col) = (y_idx, x_idx)
peak_heights = slab[local_max]

# Step 2: Compute prominence for ALL peaks using morphological reconstruction (fast, O(N))
seed = slab.copy()
seed[local_max] = -np.inf
reconstructed = reconstruction(seed, slab, method="dilation")
prominences = peak_heights - reconstructed[local_max]

# Step 3: Filter by min_considered_promenence
valid_mask = prominences >= min_considered_promenence
peak_coords = peak_coords[valid_mask]
peak_heights = peak_heights[valid_mask]
prominences = prominences[valid_mask]

# Step 4: Sort by height (descending) and take top N
if max_n_considered_peaks is not None and len(peak_heights) > max_n_considered_peaks:
    sorted_indices = np.argsort(peak_heights)[::-1][:max_n_considered_peaks]
    peak_coords = peak_coords[sorted_indices]
    peak_heights = peak_heights[sorted_indices]
    prominences = prominences[sorted_indices]

# Step 5: Convert to peaks_dict format expected by downstream code
peaks_dict = {}
for idx, (coord, height, prominence) in enumerate(zip(peak_coords, peak_heights, prominences)):
    # Convert pixel indices to spatial coordinates
    # Note: slab is in (n_ybins, n_xbins) format, coord is (row, col) = (y_idx, x_idx)
    center_x = xbin_centers[coord[1]]  # x_idx -> xbin_centers
    center_y = ybin_centers[coord[0]]  # y_idx -> ybin_centers
    
    peaks_dict[idx + 1] = {
        'id': idx + 1,
        'height': float(height),
        'prominence': float(prominence),
        'center': np.array([center_x, center_y], dtype=float),
        'col_level': float(height - prominence),  # approximate col level
        'area': None,  # not computed in efficient version
        'contours': [],  # not computed in efficient version
        'parent': 0,  # simplified - all peaks treated as independent
    }

# Step 6: Create simplified maps for compatibility
# These are not used downstream but needed for return format
id_map = np.zeros_like(slab, dtype=int)
prominence_map = np.zeros_like(slab, dtype=float)
parent_map = np.zeros_like(slab, dtype=int)

# Fill in maps for the top N peaks
for peak_id, peak_dict in peaks_dict.items():
    coord = peak_coords[peak_id - 1]  # peak_id is 1-indexed
    id_map[coord[0], coord[1]] = peak_id
    prominence_map[coord[0], coord[1]] = peak_dict['prominence']
```

### 3. Preserve All Other Logic

- Keep the `_subfn_get_contour_curve` helper function
- Keep all array initialization (lines 949-969)
- Keep the peak processing loop (lines 971-1024) - it will only process top N peaks
- Keep DataFrame construction (lines 1032-1049)
- Keep return format identical (lines 1054-1061)

### 4. Handle Edge Cases

- If no peaks found after filtering, return empty structures (same as original)
- Handle NaN values in slab appropriately
- Ensure coordinate conversion handles transposed slab correctly

### 5. Function Signature

```python
def _compute_single_posterior_slab_efficient(
    epoch_idx: int, 
    t_idx: int, 
    slab: NDArray, 
    xbin_centers: NDArray, 
    ybin_centers: NDArray, 
    step: float,  # kept for compatibility but not used
    min_considered_promenence: float, 
    peak_height_multiplier_probe_levels: Tuple, 
    max_n_considered_peaks: int = 3,  # NEW parameter
    debug_print: bool = False, 
    should_return_raw_matplotlib_Path_contours: bool = False
):
```

### Performance Benefits

- **Before**: O(1/step × slab_size) for `getProminence` + O(n_peaks × n_probe_levels × slab_size) for contours
- **After**: O(slab_size) for finding peaks + O(slab_size) for prominence + O(N × n_probe_levels × slab_size) for contours
- **Speedup**: 10-100x faster for posteriors with many peaks when N << n_peaks