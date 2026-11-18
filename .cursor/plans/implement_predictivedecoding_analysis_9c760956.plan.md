---
name: Implement PredictiveDecoding Analysis
overview: "Implement the `perform_predictive_decoding_analysis` function by following the notebook usage pattern: create DecodingLocalityMeasures, compute it, create PredictiveDecoding, compute it, and store the result. Remove outdated WCorrShuffle code."
todos:
  - id: remove_wcorr_code
    content: Remove outdated WCorrShuffle code (lines 1740-1762) that references wcorr_ripple_shuffle and shuffle computations
    status: completed
  - id: fix_container_init
    content: Fix PredictiveDecodingComputationsContainer initialization to use predictive_decoding=None instead of wcorr_ripple_shuffle=None
    status: completed
  - id: add_locality_compute
    content: Add locality_measures.compute() call after creating DecodingLocalityMeasures to ensure it is fully computed
    status: completed
  - id: fix_sigma_computation
    content: Update sigma calculation and use locality_measures.sigma in compute() call instead of hardcoded 25.0
    status: completed
  - id: add_window_size_param
    content: Add window_size parameter to function signature with default value (90 or 200 based on usage)
    status: completed
  - id: fix_result_storage
    content: Fix line 1809 to store _obj (PredictiveDecoding instance) instead of wcorr_tool in predictive_decoding field
    status: completed
---

# Implement `perform_predictive_decoding_analysis` Function

## Overview

The current implementation has outdated code referencing `WCorrShuffle` and shuffles. Based on the notebook usage pattern in `2025-12-18_NotebookDumpToScript.py`, the function should create and compute `DecodingLocalityMeasures`, then create and compute `PredictiveDecoding`, and store the result.

## Key Changes

### 1. Remove Outdated WCorrShuffle Code

- Remove lines 1740-1762 that deal with `wcorr_ripple_shuffle` and shuffle computations
- The `PredictiveDecodingComputationsContainer` has a `predictive_decoding` field (line 1612), not `wcorr_ripple_shuffle`

### 2. Initialize Container Correctly

- Change line 1742 to initialize `PredictiveDecodingComputationsContainer` with `predictive_decoding=None` instead of `wcorr_ripple_shuffle=None`

### 3. Ensure DecodingLocalityMeasures is Computed

- After creating `DecodingLocalityMeasures` (lines 1767-1772), call `locality_measures.compute()` to ensure it's fully computed before use
- The notebook shows this pattern at line 4772: `decoding_locality_measures.compute()`

### 4. Fix PredictiveDecoding Creation and Computation

- Keep the `PredictiveDecoding.init_from_decode_result` call (lines 1778-1783)
- Update sigma calculation (lines 1786-1790) - the current approach computes sigma from xbin/ybin steps
- Call `_obj.compute(sigma=...)` after creation (notebook line 5590 shows: `pred_decoding_obj.compute(sigma=pred_decoding_obj.locality_measures.sigma)`)
- The current code at line 1796 calls `_obj.compute(sigma=25.0)` but should use the computed sigma or `locality_measures.sigma`

### 5. Store Result Correctly

- Line 1809 currently assigns `wcorr_tool` to `predictive_decoding` - this is wrong
- Should assign `_obj` (the `PredictiveDecoding` instance) to `global_computation_results.computed_data['PredictiveDecoding'].predictive_decoding`

### 6. Window Size Parameter

- The function should accept a `window_size` parameter (default 90 based on line 1782, but notebook uses 200)
- Make it configurable via function parameter with a sensible default

## Implementation Pattern from Notebook

The notebook shows this sequence (lines 4770-5590):

1. Create `DecodingLocalityMeasures`: `DecodingLocalityMeasures.init_from_decode_result(...)`
2. Compute it: `decoding_locality_measures.compute()`
3. Create `PredictiveDecoding`: `PredictiveDecoding.init_from_decode_result(..., locality_measures=..., a_result_decoded=..., window_size=...)`
4. Compute it: `pred_decoding_obj.compute(sigma=pred_decoding_obj.locality_measures.sigma)`
5. Store result in pipeline

## Files to Modify