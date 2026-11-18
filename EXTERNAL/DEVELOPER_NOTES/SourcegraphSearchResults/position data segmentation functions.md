---
title: position data segmentation functions
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/6fc8e5c5-7ea3-490e-8b1c-62454535c84a
url: https://phohale.sourcegraph.app/deepsearch/324
created: 2026-01-15T11:27:09Z
updated: 2026-07-22T17:43:07Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/diba-lab/NeuroPy github.com/CommanderPho/pyPhoCoreHelpers github.com/CommanderPho/Spike3D Search my entire codebase for any functions related to segmentation/splitting of position data into separate run trajectories

I found several functions related to segmentation and splitting of position data into separate run trajectories across your codebase. Here are the main ones:

## 1. **DirectionalLapsHelpers** ([DirectionalPlacefieldGlobalComputationFunctions.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py))

Main functions for splitting trajectories by direction:

- **`update_lap_directions_properties()`** ([L1275-L1293](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1275-L1293)) - Updates laps with directional information
- **`build_global_directional_result_from_natural_epochs()`** ([L1296-L1400](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1296-L1400)) - Main function to extract directional laps from existing epochs
- **`split_to_directional_laps()`** - Splits position data into odd/even laps (LR/RL directions)

These functions create a `DirectionalLapsResult` ([L880](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L880)) object that contains:
- `split_directional_laps_dict` - Laps partitioned by direction
- `split_directional_laps_contexts_dict` - Context information for each split
- Direction-specific decoders (long_LR, long_RL, short_LR, short_RL)

## 2. **SubsequencesPartitioningResult** ([heuristic_replay_scoring.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py))

Sophisticated trajectory partitioning for decoded position sequences ([L133-500](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L133-L500)):

**Key Function:**
- **`init_from_positions_list()`** ([L489](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L489)) - Main initializer that:
  - Analyzes first-order differences in position
  - Identifies direction changes exceeding `same_thresh` (default 4cm)
  - Splits sequences at direction changes
  - Merges small intrusions/gaps (up to `max_ignore_bins`, default 2 bins)

**Key Parameters:**
```python
same_thresh: float = 4  # threshold (cm) for treating sequential positions as unchanged
max_ignore_bins: int = 2  # max sequential bins to bridge over
max_jump_distance_cm: Optional[float]  # max allowed distance between adjacent bins
```

**Outputs:**
- `split_positions_arrays` - Initial subsequences split by direction changes
- `merged_split_positions_arrays` - Subsequences after merging small gaps
- `split_indicies` - Indices where splits occur
- `position_bins_info_df` - Per-bin metadata
- `position_changes_info_df` - Information about each position change
- `subsequences_df` - Properties of each subsequence

## 3. **Lap Direction Determination** ([DefaultComputationFunctions.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/DefaultComputationFunctions.py?L36-L45))

```python
def _perform_lap_direction_determination(computation_result, **kwargs):
    """Adds the 'is_LR_dir' column to laps dataframe"""
    computation_result.sess.laps.update_lap_dir_from_net_displacement(
        pos_input=computation_result.sess.position
    )
```

This function uses net displacement to determine if each lap is left-to-right (LR) or right-to-left (RL).

## Usage Examples

### For Directional Laps:
```python
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

# Update lap directions
DirectionalLapsHelpers.update_lap_directions_properties(curr_active_pipeline)

# Get directional result
directional_laps_result = DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs(
    curr_active_pipeline
)
```

### For Position Sequence Partitioning:
```python
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequencesPartitioningResult

partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    a_most_likely_positions_list=decoded_positions,
    pos_bin_edges=position_bin_edges,
    max_ignore_bins=2,
    same_thresh=4.0,  # 4 cm threshold
    max_jump_distance_cm=None
)

# Access results
split_sequences = partition_result.split_positions_arrays
merged_sequences = partition_result.merged_split_positions_arrays
longest_sequence = partition_result.longest_sequence_subsequence
```

The diagram above shows the overall flow of trajectory segmentation in your codebase.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D position trajectory segment split](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20position%20trajectory%20segment%20split)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D run trajectory separate](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20run%20trajectory%20separate)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D) laps](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%29%20laps)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D laps epochs partition](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20laps%20epochs%20partition)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D) file:.*.py def split](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%29%20file:.%2A.py%20def%20split)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D) file:.*.py partition](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%29%20file:.%2A.py%20partition)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy) file:.*.py DirectionalLaps](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%29%20file:.%2A.py%20DirectionalLaps)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy) file:.*.py SubsequencesPartitioningResult](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%29%20file:.%2A.py%20SubsequencesPartitioningResult)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy) file:.*.py lap_direction](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%29%20file:.%2A.py%20lap_direction)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L1095-1400](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1095-1400)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L133-500](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L133-500)
- [(repo:github.com/diba-lab/NeuroPy) file:.*.py class Laps](/search?q=patterntype:keyword%20%28repo:github.com%2Fdiba-lab%2FNeuroPy%29%20file:.%2A.py%20class%20Laps)
- [github.com/diba-lab/NeuroPy](/github.com/diba-lab/NeuroPy)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py as_epoch_obj OR get_lap_times OR to_dataframe](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20as_epoch_obj%20OR%20get_lap_times%20OR%20to_dataframe)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py update_lap_dir displacement trajectory](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20update_lap_dir%20displacement%20trajectory)

## Suggested Follow-ups

- Can you elaborate on the criteria used to determine `same_thresh` and `max_ignore_bins` in the `SubsequencesPartitioningResult` function, and how they affect the resulting trajectory segmentation?
- Are there any visualization tools or methods available in the codebase to inspect the results of the trajectory segmentation performed by `DirectionalLapsHelpers` or `SubsequencesPartitioningResult`, such as plotting the split trajectories on the environment?


---

# What about methods to detect consecutive runs of values in a pandas dataframe column or something similar? For example I have a dataframe of measured/sampled animal positions over the course of an experiment. I want to find the List of time ranges (start_t, stop_t) where the animal's position satisfies a specific set of conditions -- e.g. it falls within a specific list of position bins and has a specific heading direction.

Perfect! Here are the key methods for detecting consecutive runs and finding time ranges where conditions are met:

## 1. **`contiguous_regions(condition)`** - Main Function for Consecutive Runs ([mathutil.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/utils/mathutil.py?L241-L266))

This is the primary function for finding consecutive runs of `True` values in a boolean array:

```python
from neuropy.utils.mathutil import contiguous_regions
import numpy as np

# Example: Find runs where position is in specific bins
position_df = sess.position.to_dataframe()

# Create condition: position between 50-100cm AND moving left-to-right
condition = (position_df['x'] >= 50) & (position_df['x'] <= 100) & (position_df['is_LR_dir'])

# Get indices of contiguous True regions
regions = contiguous_regions(condition.values)
# Returns: [[start_idx0, end_idx0], [start_idx1, end_idx1], ...]

# Convert indices to time ranges
time_ranges = [(position_df.iloc[start]['t'], position_df.iloc[end-1]['t']) 
               for start, end in regions]
```

**How it works:**
- Input: Boolean array/Series
- Output: NDArray of shape `(N, 2)` where each row is `[start_index, end_index]`
- Uses `np.diff()` to find transitions in the boolean array
- Handles edge cases (starts/ends with True)

## 2. **`add_epochs_id_identity()`** - Add Epoch Labels to DataFrame

Adds an epoch ID column to your dataframe based on which epoch each row falls into:

```python
from neuropy.utils.mixins.time_slicing import add_epochs_id_identity

# Create epochs from your conditions first
epochs_df = pd.DataFrame({
    'start': start_times,
    'stop': stop_times,
    'label': ['run_1', 'run_2', ...]
})

# Add epoch IDs to position data
position_df = add_epochs_id_identity(
    position_df, 
    epochs_df=epochs_df,
    epoch_id_key_name='run_id',  # new column name
    epoch_label_column_name='label',  # which column has labels
    no_interval_fill_value=-1,  # value for data outside epochs
    override_time_variable_name='t'  # time column name
)

# Now you can filter by run
run_1_data = position_df[position_df['run_id'] == 0]
```

## 3. **`find_data_indicies_from_epoch_times()`** - Find Matching Row Indices

Find which rows in your dataframe correspond to specific epoch times:

```python
from neuropy.core.epoch import find_data_indicies_from_epoch_times

# Find indices where position data matches epoch start times
epoch_start_times = np.array([10.5, 25.3, 40.1])
matching_indices = find_data_indicies_from_epoch_times(
    position_df,
    epoch_times=epoch_start_times,
    t_column_names=['t'],
    atol=0.01,  # tolerance for matching
    not_found_action='skip_index',  # or 'raise'
    debug_print=False
)
```

## Complete Workflow Example

Here's a complete example for your use case:

```python
import numpy as np
import pandas as pd
from neuropy.utils.mathutil import contiguous_regions
from neuropy.core.epoch import Epoch

# Step 1: Load your position data
position_df = sess.position.to_dataframe()

# Step 2: Define your conditions
# Example: Position in specific bins [40-60, 80-100] and heading left-to-right
target_bins = [(40, 60), (80, 100)]
is_in_target_bins = np.zeros(len(position_df), dtype=bool)
for start_pos, end_pos in target_bins:
    is_in_target_bins |= (position_df['x'] >= start_pos) & (position_df['x'] <= end_pos)

# Add direction condition (assuming you have this column)
is_correct_direction = position_df['is_LR_dir'] == True  # or your direction logic

# Combine conditions
condition = is_in_target_bins & is_correct_direction

# Step 3: Find contiguous regions
region_indices = contiguous_regions(condition.values)

# Step 4: Convert to time ranges
time_ranges = []
for start_idx, end_idx in region_indices:
    start_t = position_df.iloc[start_idx]['t']
    stop_t = position_df.iloc[end_idx - 1]['t']  # end_idx is exclusive
    time_ranges.append((start_t, stop_t))

print(f"Found {len(time_ranges)} qualifying runs")
for i, (start, stop) in enumerate(time_ranges):
    print(f"Run {i}: {start:.2f}s - {stop:.2f}s (duration: {stop-start:.2f}s)")

# Step 5: Create an Epoch object for further analysis
runs_df = pd.DataFrame({
    'start': [t[0] for t in time_ranges],
    'stop': [t[1] for t in time_ranges],
    'label': [f'run_{i}' for i in range(len(time_ranges))]
})
runs_epochs = Epoch(runs_df)

# Now you can use this to filter spikes, decode, etc.
```

## Additional Helper: Building Epochs from Multiple Conditions

If you want more complex logic:

```python
def find_runs_matching_conditions(position_df, position_ranges, direction='LR', 
                                  min_duration=0.5, min_speed=None):
    """
    Find time ranges where animal satisfies multiple conditions.
    
    Parameters:
    -----------
    position_df : pd.DataFrame
        Position dataframe with columns ['t', 'x', 'y', 'speed', 'is_LR_dir']
    position_ranges : list of tuples
        List of (min_pos, max_pos) ranges
    direction : str
        'LR' for left-to-right, 'RL' for right-to-left
    min_duration : float
        Minimum duration (seconds) for a run to be included
    min_speed : float, optional
        Minimum speed threshold
    
    Returns:
    --------
    pd.DataFrame : Epochs dataframe with start, stop, label columns
    """
    from neuropy.utils.mathutil import contiguous_regions
    
    # Build position condition
    in_position = np.zeros(len(position_df), dtype=bool)
    for min_pos, max_pos in position_ranges:
        in_position |= (position_df['x'] >= min_pos) & (position_df['x'] <= max_pos)
    
    # Direction condition
    if direction == 'LR':
        in_direction = position_df['is_LR_dir'] == True
    else:
        in_direction = position_df['is_LR_dir'] == False
    
    # Speed condition (if specified)
    if min_speed is not None:
        above_speed = position_df['speed'] >= min_speed
    else:
        above_speed = np.ones(len(position_df), dtype=bool)
    
    # Combine all conditions
    full_condition = in_position & in_direction & above_speed
    
    # Find contiguous regions
    regions = contiguous_regions(full_condition.values)
    
    # Convert to time ranges and filter by duration
    valid_runs = []
    for i, (start_idx, end_idx) in enumerate(regions):
        start_t = position_df.iloc[start_idx]['t']
        stop_t = position_df.iloc[end_idx - 1]['t']
        duration = stop_t - start_t
        
        if duration >= min_duration:
            valid_runs.append({
                'start': start_t,
                'stop': stop_t,
                'label': f'{direction}_run_{len(valid_runs)}',
                'duration': duration,
                'start_idx': start_idx,
                'end_idx': end_idx - 1
            })
    
    return pd.DataFrame(valid_runs)

# Usage:
runs_df = find_runs_matching_conditions(
    position_df,
    position_ranges=[(40, 60), (80, 100)],
    direction='LR',
    min_duration=0.5,
    min_speed=10.0
)
runs_epochs = Epoch(runs_df[['start', 'stop', 'label']])
```

The diagram above shows the overall workflow for finding consecutive runs and converting them to time ranges/epochs. The `contiguous_regions()` function is the key building block for all of these operations.

## Sources

- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers) file:.*.py find_data_indicies_from_epoch_times OR epochs_from_dataframe](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%29%20file:.%2A.py%20find_data_indicies_from_epoch_times%20OR%20epochs_from_dataframe)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers consecutive runs boolean mask time ranges](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20consecutive%20runs%20boolean%20mask%20time%20ranges)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers) file:.*.py get_contiguous_groups OR contiguous_regions](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%29%20file:.%2A.py%20get_contiguous_groups%20OR%20contiguous_regions)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers) file:.*.py where condition epochs](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%29%20file:.%2A.py%20where%20condition%20epochs)
- [github.com/diba-lab/NeuroPy/neuropy/utils/mathutil.py@1d9688a?L240-280](/github.com/diba-lab/NeuroPy@1d9688a69aeaee2f11cd73b6c5541c1e6885144e/-/blob/neuropy/utils/mathutil.py?L240-280)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers) file:.*.py add_epochs_id_identity](/search?q=patterntype:keyword%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%29%20file:.%2A.py%20add_epochs_id_identity)
- [repo:github.com/diba-lab/NeuroPy file:.*.py class EpochsAccessor OR def label_slice](/search?q=patterntype:keyword%20repo:github.com%2Fdiba-lab%2FNeuroPy%20file:.%2A.py%20class%20EpochsAccessor%20OR%20def%20label_slice)
- [github.com/diba-lab/NeuroPy/neuropy/core/epoch.py@1d9688a?L1-180](/github.com/diba-lab/NeuroPy@1d9688a69aeaee2f11cd73b6c5541c1e6885144e/-/blob/neuropy/core/epoch.py?L1-180)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py class EpochsAccessor](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20class%20EpochsAccessor)

## Suggested Follow-ups

- How can I visualize the identified runs on the position data to verify their accuracy?
- Can you provide an example of how to integrate the identified time ranges/epochs with spike data for further analysis?

