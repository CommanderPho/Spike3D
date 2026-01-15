---
name: Optimize is_included filtering
overview: Refactor the inefficient `is_included` column-based filtering approach to use direct boolean masks and vectorized operations, eliminating unnecessary dataframe copies and redundant filtering steps.
todos:
  - id: refactor_detect_matching
    content: Replace is_included column-based filtering with direct boolean mask in detect_matching_past_future_positions method (lines 2211-2219)
    status: completed
  - id: update_compute_matching
    content: Update compute_matching_pos_epochs_df to remove redundant is_included filtering (line 1316) and update docstring
    status: completed
    dependencies:
      - refactor_detect_matching
---

# Optimize is_included Filtering Implementation

## Current Issues

The current implementation in `detect_matching_past_future_positions` (lines 2211-2219) has several inefficiencies:

1. **Unnecessary full dataframe copy**: Line 2211 copies the entire `measured_positions_df` when only past/future positions are needed
2. **Inefficient column initialization**: Line 2213 initializes `is_included` to `False` for all rows
3. **Multiple `.loc[]` operations**: Lines 2215-2216 use two separate `.loc[]` calls instead of a single vectorized operation
4. **Redundant filtering**: The full dataframe is passed to `compute_matching_pos_epochs_df` (line 2219), which then filters it again (line 1316) using `measured_positions_df[measured_positions_df['is_included']]`

## Proposed Solution

### 1. Replace Column-Based Filtering with Direct Boolean Mask

**Location**: [PredictiveDecodingComputations.py:2211-2219](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)

**Current approach**:

```python
measured_positions_df_copy = measured_positions_df.copy()
measured_positions_df_copy['is_included'] = False
measured_positions_df_copy.loc[(measured_positions_df_copy['t'] < curr_epoch_start_t), 'is_included'] = True
measured_positions_df_copy.loc[(measured_positions_df_copy['t'] > curr_epoch_stop_t), 'is_included'] = True
a_matching_pos_epochs_df = MatchingPastFuturePositionsResult.compute_matching_pos_epochs_df(measured_positions_df=measured_positions_df_copy, ...)
```

**Improved approach**:

```python
# Create boolean mask directly (single vectorized operation)
is_included_mask = (measured_positions_df['t'] < curr_epoch_start_t) | (measured_positions_df['t'] > curr_epoch_stop_t)
# Filter once before passing to function
filtered_positions_df = measured_positions_df[is_included_mask].copy()
a_matching_pos_epochs_df = MatchingPastFuturePositionsResult.compute_matching_pos_epochs_df(measured_positions_df=filtered_positions_df, ...)
```

### 2. Update `compute_matching_pos_epochs_df` Method

**Location**: [PredictiveDecodingComputations.py:1294-1329](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)

**Current approach** (line 1316):

```python
a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls._custom_build_sequential_position_epochs(matching_past_positions_df=measured_positions_df[measured_positions_df['is_included']])
```

**Improved approach**:

Since the dataframe is already filtered before being passed, remove the redundant filtering:

```python
a_matching_pos_epochs_df, curr_matching_positions_df_dict = cls._custom_build_sequential_position_epochs(matching_past_positions_df=measured_positions_df)
```

Update the method docstring to clarify that `measured_positions_df` should already be filtered to only include past/future positions.

## Benefits

1. **Performance**: Eliminates unnecessary full dataframe copy and redundant filtering operations
2. **Memory efficiency**: Only creates a copy of the filtered subset, not the entire dataframe
3. **Code clarity**: Single vectorized boolean mask is more readable than multiple `.loc[]` operations
4. **Robustness**: Direct boolean operations are less error-prone than column-based filtering

## Implementation Details

- The boolean mask `(measured_positions_df['t'] < curr_epoch_start_t) | (measured_positions_df['t'] > curr_epoch_stop_t)` combines both conditions in a single vectorized operation
- The `.copy()` is only called on the filtered subset, not the full dataframe
- The `compute_matching_pos_epochs_df` method receives a pre-filtered dataframe, eliminating the need for the `is_included` column check

## Safety Analysis: Compatibility with Later Centroid Trajectory Angle Filtering

**Question**: Will this optimization interfere with the later `recompute_relevant_position_active_mask_centroid_traj_angle` filtering?

**Answer**: **No, the optimization is safe and will not interfere.** Here's why:

### Data Flow Analysis

1. **`relevant_positions_df` is created independently** (lines 2171-2183):
   - Created by spatial filtering (matching epoch mask) from `measured_positions_df`
   - Stored directly in `MatchingPastFuturePositionsResult` (line 2241)
   - **Not affected by `is_included` filtering** - it's created before the `is_included` step

2. **`is_included` filtering is only used for epoch detection** (lines 2211-2219):
   - Filters `measured_positions_df` to past/future positions
   - Used ONLY to create `a_matching_pos_epochs_df` (detected epochs)
   - The filtered dataframe is a temporary variable, not stored

3. **Epoch matching is time-based, not index-based** (line 2231):
   - `_recompute_relevant_pos_epoch_position_df_index_column` uses `adding_epochs_identity_column`
   - Matching is done by TIME (`override_time_variable_name='t'`), not by dataframe indices
   - The epochs in `a_matching_pos_epochs_df` are matched to positions in `relevant_positions_df` based on time overlap

4. **Later centroid filtering uses stored dataframes** (line 1241):
   - Uses `self.relevant_positions_df` (already stored, independent of `is_included`)
   - Uses `self.matching_pos_epochs_df` (already stored, created from filtered positions but matched by time)
   - Filters to positions within epochs using the same time-based matching

### Key Points

- **`relevant_positions_df`**: Created independently, stored separately, never modified by `is_included` filtering
- **`matching_pos_epochs_df`**: Created from filtered positions, but matched back to `relevant_positions_df` by TIME, not indices
- **Time-based matching**: The epoch-to-position matching uses the 't' column, so it works regardless of which dataframe subset was used for epoch detection
- **No index dependencies**: The optimization doesn't change any stored indices or dataframes - it only changes how the temporary filtering is done

### Optional Centroid Trajectory Angle Refinement

The `recompute_relevant_position_active_mask_centroid_traj_angle` method (lines 1218-1271) is an **optional refinement step** that:

1. **Operates on already-filtered data**:
   - Uses `self.relevant_positions_df` (already filtered by spatial mask and time)
   - Uses `self.matching_pos_epochs_df` (already filtered epochs)
   - Further filters to positions within epochs (line 1241: `drop_non_epoch_events=True`)

2. **Is computationally expensive** (hence optional):
   - Controlled by `should_defer_extended_computations` flag (default: `True`)
   - Only runs when explicitly called via `recompute_all()` or directly
   - Performs trajectory segmentation and angle matching calculations

3. **Produces fully filterable output**:
   - Adds `centroid_pos_traj_matching_angle_idx` column to `relevant_positions_df` (lines 1264-1267)
   - Positions with `-1` are not matched; positions with valid index (>= 0) are matched
   - Can be used to create fully filtered output:
     ```python
     # Get positions that match centroid trajectory angles
     fully_filtered_positions_df = result.relevant_positions_df[
         result.relevant_positions_df['centroid_pos_traj_matching_angle_idx'] != -1
     ]
     # Or get positions within epochs that match
     epoch_filtered_positions_df = result.relevant_positions_df[
         (result.relevant_positions_df['matching_found_relevant_pos_epoch'] != -1) &
         (result.relevant_positions_df['centroid_pos_traj_matching_angle_idx'] != -1)
     ]
     ```

4. **Is unaffected by the optimization**:
   - The optimization only changes how `is_included` filtering is done for epoch detection
   - The stored `relevant_positions_df` and `matching_pos_epochs_df` remain identical
   - The refinement step operates on these stored dataframes, which are unchanged

### Conclusion

The optimization is **completely safe** because:
1. It only affects a temporary filtering step used for epoch detection
2. The stored dataframes (`relevant_positions_df`, `matching_pos_epochs_df`) are independent and unchanged
3. Epoch matching is time-based, so it works regardless of the filtering method
4. The optional centroid trajectory angle refinement operates on the stored dataframes, which are unchanged
5. The refinement step can still produce fully filterable output using the `centroid_pos_traj_matching_angle_idx` column

## Testing Considerations

- Verify that the filtered results match the original implementation
- Ensure that `_custom_build_sequential_position_epochs` works correctly with the pre-filtered dataframe
- Check that edge cases (empty dataframes, no past/future positions) are handled correctly
- **Verify that `relevant_positions_df` and `matching_pos_epochs_df` remain unchanged** (they should be identical to the original implementation)
- **Test that `recompute_relevant_position_active_mask_centroid_traj_angle` produces identical results** after the optimization