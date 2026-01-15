---
name: Merge centroid_pos_traj_matching_angle_idx to relevant_positions_df
overview: Replace the code that aggregates and merges into matching_pos_epochs_df with code that merges the centroid_pos_traj_matching_angle_idx column directly from epoch_only_relevant_positions_df into self.relevant_positions_df using index-based matching.
todos:
  - id: remove_epoch_aggregation
    content: Remove the groupby aggregation and merge into matching_pos_epochs_df (lines 1191-1206)
    status: completed
  - id: add_positions_merge
    content: Add code to merge centroid_pos_traj_matching_angle_idx from epoch_only_relevant_positions_df into self.relevant_positions_df using index-based matching
    status: completed
---

# Merge centroid_pos_traj_matching_angle_idx to relevant_positions_df

## Current Issue

The code currently aggregates `'centroid_pos_traj_matching_angle_idx'` by epoch and merges it into `self.matching_pos_epochs_df`, but it should instead be merged into `self.relevant_positions_df`.

## Solution

Since `epoch_only_relevant_positions_df` is a filtered subset of `self.relevant_positions_df` (created with `drop_non_epoch_events=True`), we can merge the column back using the DataFrame index.

## Implementation Steps

1. **Remove the aggregation and merge into matching_pos_epochs_df** (lines 1191-1206)

   - Remove the groupby aggregation by epoch
   - Remove the merge into `self.matching_pos_epochs_df`
   - Remove the label column handling for epochs

2. **Merge column into self.relevant_positions_df** (replace lines 1191-1206)

   - Initialize the column in `self.relevant_positions_df` if it doesn't exist (set to -1)
   - Extract just the `'centroid_pos_traj_matching_angle_idx'` column from `epoch_only_relevant_positions_df` along with its index
   - Update `self.relevant_positions_df` by matching on index (using `.loc` or merge on index)
   - Rows not in `epoch_only_relevant_positions_df` will keep their -1 value

## Code Changes

In [`PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py) at lines 1191-1206:

Replace the aggregation/merge logic with:

```python
# Assign the 'centroid_pos_traj_matching_angle_idx' column from epoch_only_relevant_positions_df to self.relevant_positions_df
# Initialize the column if it doesn't exist
if 'centroid_pos_traj_matching_angle_idx' not in self.relevant_positions_df.columns:
    self.relevant_positions_df['centroid_pos_traj_matching_angle_idx'] = -1
# Update values from epoch_only_relevant_positions_df by matching on index
self.relevant_positions_df.loc[epoch_only_relevant_positions_df.index, 'centroid_pos_traj_matching_angle_idx'] = epoch_only_relevant_positions_df['centroid_pos_traj_matching_angle_idx']
```

This approach:

- Preserves all rows in `self.relevant_positions_df` (those not in epochs keep -1)
- Updates only the rows that exist in `epoch_only_relevant_positions_df` with their computed values
- Uses index-based matching which is efficient and preserves the original DataFrame structure