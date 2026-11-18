---
name: Add 3D posterior support to mask function
overview: Add support for 3D posteriors (len(num_spatial_dims_list) == 3) to the mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin function by handling the 3D case in dimension unpacking and fixing dimension-specific indexing to use ellipsis for dimension-agnostic operations.
todos:
  - id: add_3d_dimension_unpacking
    content: Add elif branch for len(num_spatial_dims_list) == 3 case to unpack num_positions, num_y_bins, num_z_bins
    status: completed
  - id: fix_most_likely_position_indicies_indexing
    content: "Replace hardcoded : indexing with ... ellipsis in most_likely_position_indicies_list operations (lines 1705, 1718) to support both 2D and 3D"
    status: completed
  - id: fix_most_likely_positions_indexing
    content: "Replace hardcoded : indexing with ... ellipsis in most_likely_positions_list operations (line 1719) to support both 2D and 3D"
    status: completed
  - id: update_error_message
    content: Update NotImplementedError message to reflect that 3D is now supported (only unsupported cases should raise)
    status: completed
    dependencies:
      - add_3d_dimension_unpacking
  - id: update_comments
    content: Update comments at lines 1704 and 1707 to reflect dimension-agnostic shapes
    status: completed
---

# Add 3D Posterior Support to mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin

## Overview

The function `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` currently only handles 1D and 2D posteriors. It needs to support 3D posteriors where `p_x_given_n_list[i] `has shape `(x_bins, y_bins, z_bins, time_bins)`.

## Current Issues

1. **Dimension unpacking (lines 1652-1659)**: Only handles `len(num_spatial_dims_list) == 1` and `len(num_spatial_dims_list) == 2`, raises `NotImplementedError` for 3D.
2. **Dimension-specific indexing**: Several places use hardcoded `:` indexing that assumes 2D:

- Line 1705: `[:, all_time_bin_indicies]` assumes shape `(2, num_time_bins)` but should be `(3, num_time_bins)` for 3D
- Line 1718: `[:, is_time_bin_active]` assumes 2D
- Line 1719: `[is_time_bin_active, :]` assumes shape `(num_time_bins, 2)` but should be `(num_time_bins, 3)` for 3D

## Implementation Plan

### 1. Add 3D case to dimension unpacking (lines 1652-1659)

- Add `elif len(num_spatial_dims_list) == 3:` branch
- Unpack as `num_positions, num_y_bins, num_z_bins = num_spatial_dims_list`
- Update the error message to reflect that 3D is now supported

### 2. Fix dimension-specific indexing to be dimension-agnostic

- **Line 1705**: Change `[:, all_time_bin_indicies]` to `[..., all_time_bin_indicies]` 
    - Works for both 2D `(2, num_time_bins)` and 3D `(3, num_time_bins)`
- **Line 1718**: Change `[:, is_time_bin_active]` to `[..., is_time_bin_active]`
    - Works for both 2D and 3D `most_likely_position_indicies_list`
- **Line 1719**: Change `[is_time_bin_active, :]` to `[is_time_bin_active, ...]`
    - Works for both 2D `(num_time_bins, 2)` and 3D `(num_time_bins, 3)`

### 3. Update comments (optional but helpful)

- Line 1704: Update comment from "shape: 2, num_time_bins" to "shape: (n_dims, num_time_bins) where n_dims=2 for 2D, 3 for 3D"
- Line 1707: Update comment from "shape: num_time_bins, 2" to "shape: (num_time_bins, n_dims) where n_dims=2 for 2D, 3 for 3D"

## Files to Modify

- [`pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (lines 1652-1719)

## Notes

- The rest of the function already uses dimension-agnostic approaches (e.g., `...` ellipsis) for `p_x_given_n_list` operations, so those don't need changes