---
name: Complete downsampling_spatial_data implementation
overview: Finish implementing the `downsampling_spatial_data` method to properly update all `DecodingLocalityMeasures` fields after spatial downsampling, including bin centers, bin edges, PDF data, and dependent computed fields.
todos:
  - id: update_pdf_data
    content: Update p_x_given_n with coarse_pdf from downsampling
    status: pending
  - id: update_bin_centers
    content: Update xbin_centers and ybin_centers from coarse_bins[0] and coarse_bins[1]
    status: pending
  - id: compute_bin_edges
    content: Compute xbin and ybin edges from centers using get_bin_edges() helper function
    status: pending
    dependencies:
      - update_bin_centers
  - id: update_sigma
    content: Recompute sigma based on new bin sizes
    status: pending
    dependencies:
      - update_bin_centers
  - id: reset_computed_fields
    content: Reset gaussian_volume, p_x_given_n_dict, and locality_measures_df to None
    status: pending
  - id: cleanup_debug_code
    content: Remove or comment out debug statements like np.shape() calls
    status: pending
---

# Complete `downsampling_spatial_data` Implementation

## Overview

The `downsampling_spatial_data` method currently performs the downsampling computation but doesn't update the `DecodingLocalityMeasures` instance with the downsampled data. This plan completes the implementation by updating all relevant fields.

## Current State

The function at lines 748-780 in [PredictiveDecodingComputations.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py):

- Creates a deep copy of self
- Performs downsampling using `RigorousPDFDownsampler`
- Gets `coarse_pdf`, `coarse_bin_sizes`, and `coarse_bins` from the downsampler
- Validates normalization
- **Missing**: Updates the instance fields with downsampled data

## Implementation Steps

### 0. Add Import

- Add import at the top of the function (or check if already imported): `from neuropy.utils.mixins.binning_helpers import get_bin_edges`

### 1. Update PDF Data

- Replace `decoding_locality_measures.p_x_given_n` with `coarse_pdf`

### 2. Update Spatial Bin Centers

- Extract x-axis bin centers: `decoding_locality_measures.xbin_centers = coarse_bins[0]`
- Extract y-axis bin centers: `decoding_locality_measures.ybin_centers = coarse_bins[1]`
- Note: `coarse_bins[2]` and `coarse_bins[3]` remain unchanged (context and time axes)

### 3. Compute Bin Edges from Centers

- Import helper functions: `from neuropy.utils.mixins.binning_helpers import get_bin_edges`
- Compute `xbin` edges from `xbin_centers` using: `decoding_locality_measures.xbin = get_bin_edges(decoding_locality_measures.xbin_centers)`
- Compute `ybin` edges from `ybin_centers` using: `decoding_locality_measures.ybin = get_bin_edges(decoding_locality_measures.ybin_centers)`
- Note: `get_bin_edges()` automatically handles the conversion assuming fixed bin width

### 4. Update Sigma

- Recompute `sigma` based on new bin sizes:
- `x_step = coarse_bin_sizes[0] `(or `np.nanmean(np.diff(xbin_centers))`)
- `y_step = coarse_bin_sizes[1] `(or `np.nanmean(np.diff(ybin_centers))`)
- `decoding_locality_measures.sigma = np.nanmax([x_step, y_step]) * 5.0`

### 5. Reset Computed Fields

- Set `decoding_locality_measures.gaussian_volume = None` (will be recomputed when needed)
- Set `decoding_locality_measures.p_x_given_n_dict = None` (will be recomputed when needed)
- Optionally reset `decoding_locality_measures.locality_measures_df = None` if it depends on spatial bins

### 6. Clean Up Debug Code

- Remove or comment out debug statements like `np.shape(coarse_pdf)` and the commented normalization warning

## Key Implementation Details

**Bin Edges Computation:**Use the existing helper function from `neuropy.utils.mixins.binning_helpers`:

```python
from neuropy.utils.mixins.binning_helpers import get_bin_edges

# Convert bin centers to bin edges
decoding_locality_measures.xbin = get_bin_edges(decoding_locality_measures.xbin_centers)
decoding_locality_measures.ybin = get_bin_edges(decoding_locality_measures.ybin_centers)
```

The `get_bin_edges()` function automatically computes edges from centers assuming fixed bin width.**Field Updates:**

- `p_x_given_n`: Direct assignment from `coarse_pdf`
- `xbin_centers`, `ybin_centers`: Direct assignment from `coarse_bins[0]`, `coarse_bins[1]`
- `xbin`, `ybin`: Computed from centers and bin sizes
- `sigma`: Recomputed from new bin sizes
- Computed fields: Set to `None` to trigger recomputation