---
name: Convert classes to ComputedResult subclasses
overview: Convert `PeakCounts` and `PosteriorPeaksPeakProminence2dResult` to inherit from `ComputedResult` with proper `@define(slots=False, repr=False, eq=False)` decorator and `serialized_field` support, following the pattern from `DecodingLocalityMeasures`.
todos:
  - id: "1"
    content: Add ComputedResult and serialized_field imports to peak_prominence2d.py
    status: completed
  - id: "2"
    content: Update PeakCounts to inherit from ComputedResult with serialized_field and proper decorator
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Update PosteriorPeaksPeakProminence2dResult to inherit from ComputedResult with serialized_field and proper decorator
    status: completed
    dependencies:
      - "1"
---

# Convert Classes to ComputedResult Subclasses

## Overview

Convert `PeakCounts` and `PosteriorPeaksPeakProminence2dResult` to inherit from `ComputedResult` with proper serialization support, following the pattern used in `DecodingLocalityMeasures`.

## Implementation Steps

### 1. Add Required Imports

Add to [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py):

- `from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult`
- `from neuropy.utils.mixins.AttrsClassHelpers import serialized_field, non_serialized_field`

### 2. Update PeakCounts Class

- Change decorator from `@define(slots=False, eq=False)` to `@define(slots=False, repr=False, eq=False)`
- Make it inherit from `ComputedResult`
- Replace `field()` with `serialized_field()` for all fields:
- `raw: NDArray = serialized_field()`
- `uniform_blurred: NDArray = serialized_field()`
- `gaussian_blurred: NDArray = serialized_field()`
- Add `_VersionedResultMixin_version: str = "2025.12.21_0"` field (following pattern from DecodingLocalityMeasures)

### 3. Update PosteriorPeaksPeakProminence2dResult Class

- Change decorator from `@define(slots=False, eq=False)` to `@define(slots=False, repr=False, eq=False)`
- Make it inherit from `ComputedResult`
- Replace `field()` with `serialized_field()` for all fields:
- `xx: NDArray = serialized_field()`
- `yy: NDArray = serialized_field()`
- `results: Dict[Tuple[int, int], Dict[str, Any]] = serialized_field()`
- `flat_peaks_df: pd.DataFrame = serialized_field()`
- `filtered_flat_peaks_df: pd.DataFrame = serialized_field()`
- `peak_counts: PeakCounts = serialized_field()`