---
name: Refactor mask function to use generalized version
overview: Refactor `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` to use `mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks` internally, eliminating code duplication while preserving exact output behavior.
todos:
  - id: "1"
    content: Read the full implementation of both functions to understand all edge cases and ensure complete understanding
    status: completed
  - id: "2"
    content: Refactor mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin to compute is_time_bin_active_list and delegate to the generalized function
    status: completed
  - id: "3"
    content: Verify the refactored function returns the exact same tuple structure as before
    status: completed
  - id: "4"
    content: Test that all existing usages continue to work (check for any callers that might depend on internal implementation details)
    status: completed
---

# Refactor mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin

## Analysis

The function `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` (line 1808 in `reconstruction.py`) currently duplicates most of the masking logic that exists in the generalized `mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks` (line 1612). The only difference is that the first function computes `is_time_bin_active` arrays from spike counts, while the second accepts them as input.

## Current Structure

**`mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks`** (generalized):

- Input: `is_time_bin_active_list: List[NDArray]` (pre-computed masks)
- Performs all masking operations
- Returns: `(DecodedFilterEpochsResult, (is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list))`

**`mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin`** (specific):

- Input: `spikes_df`, `min_num_spikes_per_bin_to_be_considered_active`, `min_num_unique_active_neurons_per_time_bin`
- Computes `is_time_bin_active` for each epoch using `spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask`
- Then duplicates all the masking logic from the generalized function
- Returns: Same structure as generalized function

## Refactoring Plan

### Step 1: Compute is_time_bin_active_list

In `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin`:

- Loop through each epoch (same as current code, lines 1849-1873)
- For each epoch, extract `time_bin_edges` and compute spatial dimensions
- Call `spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask` with the appropriate parameters
- Extract `is_time_bin_active` from the return tuple
- Append to `is_time_bin_active_list`

### Step 2: Delegate to generalized function

- Call `self.mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks(is_time_bin_active_list=is_time_bin_active_list, masked_bin_fill_mode=masked_bin_fill_mode)`
- Return the result directly

### Step 3: Preserve function signature and behavior

- Keep the exact same function signature
- Ensure the return tuple structure matches exactly (same variable names in the tuple)
- Maintain all parameter defaults
- Keep the docstring (may update it to mention it uses the generalized function internally)

## Files to Modify

- `h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py`
- Refactor `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` (lines 1808-1999)
- Remove duplicated masking logic (~170 lines)
- Replace with spike count computation loop + single call to generalized function

## Implementation Details

The refactored function will:

1. Validate `masked_bin_fill_mode` (keep existing assertion)
2. Get `num_filter_epochs` from `self.num_filter_epochs`
3. Loop through epochs to compute `is_time_bin_active_list`:

- Extract shape and dimensions (same logic as current lines 1852-1863)
- Extract `time_bin_edges` (same logic as current lines 1865-1873)
- Call `compute_unit_time_binned_spike_counts_and_mask` (current line 1875-1877)
- Extract `is_time_bin_active` from the tuple
- Append to list

4. Call `self.mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks(is_time_bin_active_list=is_time_bin_active_list, masked_bin_fill_mode=masked_bin_fill_mode)`
5. Return the result

This reduces the function from ~190 lines to ~30-40 lines while preserving exact behavior.