---
name: Simplify PredictiveDecoding Input Interface
overview: Remove dependency on `directional_decoders_decode_result` from `PredictiveDecoding.init_from_decode_result`, simplifying it to work with just `locality_measures` and `a_result_decoded` parameters. Clean up all related cruft from the old implementation.
todos:
  - id: update_init_signature
    content: Update PredictiveDecoding.init_from_decode_result method signature to remove directional_decoders_decode_result and extant_decoded_time_bin_size parameters
    status: completed
  - id: simplify_init_logic
    content: Remove all logic related to directional_decoders_decode_result from init_from_decode_result method body
    status: completed
    dependencies:
      - update_init_signature
  - id: update_perform_compute
    content: Review and update _perform_compute_predictive_decoding to ensure it works with simplified inputs (may need to extract data from locality_measures if a_result_decoded is not provided)
    status: completed
    dependencies:
      - simplify_init_logic
  - id: clean_imports
    content: Remove unused imports and update type hints related to DirectionalDecodersContinuouslyDecodedResult
    status: completed
    dependencies:
      - simplify_init_logic
  - id: update_docstrings
    content: Update docstrings and usage examples to reflect the new simplified interface
    status: completed
    dependencies:
      - simplify_init_logic
---

# Simplify PredictiveDecoding Input Interface

## Overview

Refactor `PredictiveDecoding.init_from_decode_result` to remove the dependency on `directional_decoders_decode_result`, making it work with just `locality_measures` (which contains all decoder info) and `a_result_decoded`.

## Changes Required

### 1. Update `PredictiveDecoding.init_from_decode_result` method signature

**File**: [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)

- Remove `directional_decoders_decode_result` parameter
- Remove `extant_decoded_time_bin_size` parameter (only used to extract from `directional_decoders_decode_result`)
- Keep `locality_measures` as optional but make it the primary source of decoder information
- Keep `a_result_decoded` as optional
- If `a_result_decoded` is not provided but `locality_measures` is, we may need to handle this case (or make `a_result_decoded` required when `locality_measures` is provided)

### 2. Simplify initialization logic in `init_from_decode_result`

**Lines**: ~1047-1079

- Remove the conditional logic that creates `DecodingLocalityMeasures` from `directional_decoders_decode_result` (lines 1052-1060)
- Remove the logic that extracts `a_result_decoded` from `directional_decoders_decode_result` (lines 1062-1067)
- Remove assertions related to `directional_decoders_decode_result`
- Simplify to: if `locality_measures` is provided, use it; otherwise, this is an error (since we can't create it without decoder info)
- If `a_result_decoded` is not provided, we may need to extract `p_x_given_n` from `locality_measures.p_x_given_n`, but check if `_perform_compute_predictive_decoding` can work with that

### 3. Update `_perform_compute_predictive_decoding` if needed

**Lines**: ~992-1044

- Check if this method can work with data extracted from `locality_measures` instead of requiring a full `DecodedFilterEpochsResult`
- Currently uses: `a_result_decoded.time_window_centers`, `a_result_decoded.time_bin_container.centers`, `a_result_decoded.p_x_given_n`
- `locality_measures` has: `time_window_centers`, `p_x_given_n`
- May need to make this method more flexible to accept either `DecodedFilterEpochsResult` or data from `locality_measures`

### 4. Clean up imports and type hints

- Remove any unused imports related to `DirectionalDecodersContinuouslyDecodedResult` if no longer needed
- Update type hints to reflect the simplified interface

### 5. Update docstrings and usage examples

- Update the docstring for `init_from_decode_result` to reflect the new simplified interface
- Update any usage examples in comments to show the new pattern

## Implementation Notes

- The key insight is that `locality_measures` already contains all the decoder information (xbin, ybin, xbin_centers, ybin_centers, epoch_names) that was previously extracted from `directional_decoders_decode_result`
- `_perform_compute_predictive_decoding` may need to be updated to handle the case where we're working with data from `locality_measures` rather than a full `DecodedFilterEpochsResult` object
- The method signature should become: `init_from_decode_result(cls, curr_active_pipeline, locality_measures: DecodingLocalityMeasures, a_result_decoded: Optional[DecodedFilterEpochsResult] = None, window_size: int = 200, sigma: Optional[float] = None)`