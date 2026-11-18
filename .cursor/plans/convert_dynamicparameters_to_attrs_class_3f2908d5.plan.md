---
name: Convert DynamicParameters to attrs class
overview: Convert the return type of `_perform_find_posterior_peaks_peak_prominence2d_computation` from `DynamicParameters` to a custom attrs-based class with proper type hints, following the user's typical style using `@define` decorator.
todos:
  - id: "1"
    content: Add attrs imports (define, field) to peak_prominence2d.py if not already present
    status: completed
  - id: "2"
    content: Create nested PeakCounts attrs class with raw, uniform_blurred, gaussian_blurred fields
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Create PosteriorPeaksPeakProminence2dResult attrs class with all return fields
    status: completed
    dependencies:
      - "2"
  - id: "4"
    content: Update _perform_find_posterior_peaks_peak_prominence2d_computation return type annotation
    status: completed
    dependencies:
      - "3"
  - id: "5"
    content: Update both return statements to use new class instead of DynamicParameters
    status: completed
    dependencies:
      - "4"
---

# Convert DynamicParameters Output to attrs-based Class

## Overview

Replace the `DynamicParameters` return object in `_perform_find_posterior_peaks_peak_prominence2d_computation` with a custom attrs-based class that has defined fields and type hints.

## Implementation Steps

### 1. Create Nested `PeakCounts` Class

Create a nested attrs class for the `peak_counts` field in [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py):

- `raw`: NDArray - raw peak counts map
- `uniform_blurred`: NDArray - uniform filtered peak counts
- `gaussian_blurred`: NDArray - Gaussian filtered peak counts

### 2. Create Main Result Class

Create `PosteriorPeaksPeakProminence2dResult` class in the same file with fields:

- `xx`: NDArray - xbin_centers
- `yy`: NDArray - ybin_centers  
- `results`: Dict[Tuple[int, int], Dict[str, Any]] - nested dict keyed by (epoch_idx, time_bin_idx) containing peak analysis results
- `flat_peaks_df`: pd.DataFrame - concatenated DataFrame of all peaks
- `filtered_flat_peaks_df`: pd.DataFrame - filtered subset used for peak-count maps
- `peak_counts`: PeakCounts - nested class containing raw and blurred peak count maps

### 3. Update Function Return Type

- Change return type annotation from `'DynamicParameters'` to `'PosteriorPeaksPeakProminence2dResult'`
- Update both return statements (lines 1046 and 1076-1079) to instantiate the new class instead of `DynamicParameters`

### 4. Add Required Imports

- Ensure `from attrs import define, field` is imported at the top of the file
- Ensure `pandas as pd` is imported for DataFrame type hints

## Style Guidelines

- Use `@define(slots=False, eq=False)` decorator (matching user's style from `decoder_plotting_mixins.py`)
- Use `field()` for all fields