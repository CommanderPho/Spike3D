---
name: Refactor Sequentiality Computations
overview: Extract the sequentiality computation logic from PredictiveDecodingComputations into a standalone function in SequenceBasedComputationsGlobalComputationFunctions that operates on generic spatial-temporal data (masks and peak locations) without depending on predictive decoding results.
todos:
  - id: "1"
    content: Add compute_temporal_sequentiality_measures static method to SequenceBasedComputationsGlobalComputationFunctions class in SequenceBasedComputations.py
    status: completed
  - id: "2"
    content: Add necessary imports (scipy.ndimage) to SequenceBasedComputations.py if not already present
    status: completed
  - id: "3"
    content: Replace inline sequentiality computation (lines 688-816) in PredictiveDecodingComputations.py with call to new function
    status: completed
    dependencies:
      - "1"
  - id: "4"
    content: Add import for SequenceBasedComputationsGlobalComputationFunctions in PredictiveDecodingComputations.py
    status: completed
    dependencies:
      - "1"
---

# Refactor Sequentiality Computations

## Overview

Refactor the sequentiality computation code (lines 688-816 in `PredictiveDecodingComputations.py`) into a new standalone function in `SequenceBasedComputationsGlobalComputationFunctions` that accepts generic inputs (spatial masks and peak locations) rather than depending on predictive decoding results.

## Changes Required

### 1. Add new function to `SequenceBasedComputations.py`

Create a new static method `compute_temporal_sequentiality_measures` in the `SequenceBasedComputationsGlobalComputationFunctions` class that:

- **Inputs:**
- `alpha_epoch_masks`: 3D boolean array `(n_x_bins, n_y_bins, n_t_bins)` - prominence masks for each time bin
- `epoch_promenence_tuples`: List of tuples, each containing `(peak_coords, prominences, peak_heights)` for each time bin
- `xbin_centers`: 1D array of x-bin center coordinates (for converting bin space to cm space)
- `ybin_centers`: 1D array of y-bin center coordinates (for converting bin space to cm space)
- `computation_measure_name`: Optional string (default: `'temporal_sequentiality'`) for naming the output keys
- **Outputs:**
- Dictionary containing:
    - `f"{computation_measure_name}_mask_overlap"`: Sequentiality mask overlap (Jaccard index) between subsequent time bins
    - `f"{computation_measure_name}_peak_change_vector_bin_space"`: Change vectors in bin space
    - `f"{computation_measure_name}_peak_change_vector_cm_space"`: Change vectors in cm space
    - `f"{computation_measure_name}_peak_change_vector_magnitude"`: Magnitude of change vectors
    - `f"{computation_measure_name}_direction_change_angle_rad"`: Direction change angles in radians
    - `f"{computation_measure_name}_direction_change_angle_deg"`: Direction change angles in degrees
- **Implementation details:**
- Use `scipy.ndimage.binary_dilation` with a 3x3 kernel for mask dilation
- Compute Jaccard index (intersection/union) for mask overlap
- Extract top peak locations from prominence tuples
- Compute change vectors and their magnitudes
- Compute direction change angles using dot product formula

### 2. Update `PredictiveDecodingComputations.py`

Replace the inline sequentiality computation (lines 688-816) with:

- Import the new function from `SequenceBasedComputations`
- Call `SequenceBasedComputationsGlobalComputationFunctions.compute_temporal_sequentiality_measures(...)` with:
- `an_alpha_epoch_masks` (already computed)
- `epoch_promenence_tuples` (already computed)
- `self.xbin_centers` and `self.ybin_centers` (from the class)
- `a_computation_measure_name='temporal_sequentiality'`
- Store the returned dictionary values into `self.locality_measures_dict_dict[an_epoch_name]` using the same keys as before

### 3. Add necessary imports

- In `SequenceBasedComputations.py`: Add `import scipy.ndimage as ndimage` if not already present
- Ensure proper type hints using `NDArray` from `nptyping`

## Files to Modify

1. **[pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/SequenceBasedComputations.py](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/SequenceBasedComputations.py)**

- Add new static method `compute_temporal_sequentiality_measures` to `SequenceBasedComputationsGlobalComputationFunctions` class (after line 1457, before the display functions section)

2. **[pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)**

- Replace lines 688-816 with a call to the new function
- Add import for the new function

## Key Design Decisions

- The function is **static** (doesn't require class instance) to make it reusable and independent