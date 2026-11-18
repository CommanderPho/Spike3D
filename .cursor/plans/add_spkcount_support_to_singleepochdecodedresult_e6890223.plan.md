---
name: Add spkcount support to SingleEpochDecodedResult
overview: Add optional `spkcount` field to `SingleEpochDecodedResult` class and update `get_result_for_epoch` and `init_from_single_epoch_result` methods to preserve `spkcount` data when extracting/reconstructing single epoch results.
todos:
  - id: add_spkcount_field
    content: Add optional spkcount field to SingleEpochDecodedResult class definition (around line 449)
    status: pending
  - id: update_get_result_for_epoch
    content: Add 'spkcount' to single_epoch_field_names and fields_to_single_epoch_fields_dict in get_result_for_epoch() method (lines 943-945)
    status: pending
  - id: update_init_from_single_epoch_result
    content: Update init_from_single_epoch_result() to check for and use single_epoch_result.spkcount if parameter is not provided (lines 1008-1012)
    status: pending
    dependencies:
      - add_spkcount_field
---

# Add spkcount Support to SingleEpochDecodedResult



## Overview

Currently, `spkcount` (spike count arrays) exists on `DecodedFilterEpochsResult` but is not preserved when extracting a single epoch via `get_result_for_epoch()`. This plan adds optional `spkcount` support to `SingleEpochDecodedResult` to maintain data completeness.

## Current State

1. **`DecodedFilterEpochsResult`** has `spkcount: list` (line 900) - a list of arrays, one per epoch
2. **`SingleEpochDecodedResult`** does NOT have a `spkcount` field (lines 429-449)
3. **`get_result_for_epoch()`** (line 943) omits `'spkcount'` from `single_epoch_field_names`
4. **`init_from_single_epoch_result()`** (line 1008-1012) accepts optional `spkcount` parameter but cannot use stored value from `SingleEpochDecodedResult`

## Implementation Plan

### 1. Add optional `spkcount` field to `SingleEpochDecodedResult` class

- **File**: [`pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (around line 449)
- Add: `spkcount: Optional[NDArray] = non_serialized_field()`
- Make it optional to maintain backward compatibility with existing code

### 2. Update `get_result_for_epoch()` to include `spkcount`

- **File**: [`pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (lines 943-945)
- Add `'spkcount'` to `single_epoch_field_names` list
- Add `'spkcount'` mapping to `fields_to_single_epoch_fields_dict` (maps to itself since no plural/singular conversion needed)
- The existing validation loop (lines 947-959) will automatically handle `spkcount` since it iterates over `single_epoch_field_names`

### 3. Update `init_from_single_epoch_result()` to use stored `spkcount` if available

- **File**: [`pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (lines 1008-1012)