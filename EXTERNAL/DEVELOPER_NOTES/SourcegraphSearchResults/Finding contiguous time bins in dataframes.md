---
title: Finding contiguous time bins in dataframes
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/afb0a184-2f65-4f6c-ba8e-2deb99ea2cd9
url: https://phohale.sourcegraph.app/deepsearch/307
created: 2025-12-15T10:39:32Z
updated: 2026-07-22T17:42:42Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/diba-lab/NeuroPy github.com/CommanderPho/pyPhoCoreHelpers github.com/CommanderPho/Spike3D Trying to find a function/functions that determine contiguous time bins within dataframes to produce epochs 

Now let me create a comprehensive summary with the key functions I found:

Based on my exploration, I found several key functions for determining contiguous time bins within dataframes to produce epochs across your repositories:

## Core Functions

### 1. **`contiguous_regions`** - [neuropy/utils/mathutil.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/utils/mathutil.py?L241-L266)

This is the fundamental function that finds contiguous True regions in a boolean array:

```python
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    
    # Find the indices of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()
    
    # Shift the index by 1 to the right
    idx += 1
    
    if condition[0]:
        idx = np.r_[0, idx]
    
    if condition[-1]:
        idx = np.r_[idx, condition.size]
    
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx
```

**Used by:**
- [neuropy/utils/position_util.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/utils/position_util.py?L9)
- [neuropy/io/usvio.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/io/usvio.py?L14)
- [neuropy/io/optitrackio.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/io/optitrackio.py?L177)

### 2. **`thresh_epochs`** - [neuropy/utils/mathutil.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/utils/mathutil.py?L199-L238)

Creates epochs based on threshold detection with merging and duration filtering:

```python
def thresh_epochs(arr: np.ndarray, thresh, length, sep=0, boundary=0, fs=1):
    """
    Parameters:
    - thresh: height threshold (min, max)
    - length: duration threshold (min, max)
    - sep: epochs separated by less than this will be merged
    - boundary: boundary threshold value
    - fs: sampling frequency
    """
    # Find peaks that exceed threshold
    # Merge overlapping epochs based on separation parameter
    # Filter by duration thresholds
    # Returns: starts, stops, peaks, peak_values (all in seconds)
```

**Used by:**
- [neuropy/utils/position_util.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/utils/position_util.py?L9)

### 3. **`Epoch.from_boolean_array`** - [neuropy/core/epoch.py](https://phohale.sourcegraph.app/github.com/diba-lab/NeuroPy/-/blob/neuropy/core/epoch.py?L755-L784)

Creates epochs from a boolean time series:

```python
@staticmethod
def from_boolean_array(arr, t=None):
    """Create epochs from a boolean array
    
    Parameters
    ----------
    arr : np.array
        timeseries of boolean values
    t : np.array, optional
        corresponding time in seconds
        
    Returns
    -------
    core.Epoch
        epochs where the arr is high
    """
    assert np.array_equal(arr, arr.astype(bool))
    int_arr = arr.astype("int")
    pad_arr = np.pad(int_arr, 1)
    diff_arr = np.diff(pad_arr)
    starts, stops = np.where(diff_arr == 1)[0], np.where(diff_arr == -1)[0]
    stops[stops == len(arr)] = len(arr) - 1
    
    if t is not None:
        starts, stops = t[starts], t[stops]
    
    return Epoch.from_array(starts, stops, "high")
```

### 4. **`partition_df_dict`** - [pyphocorehelpers/indexing_helpers.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoCoreHelpers/-/blob/src/pyphocorehelpers/indexing_helpers.py?L784-L794)

Splits a DataFrame into multiple DataFrames based on a column's unique values:

```python
def partition_df_dict(df: pd.DataFrame, partitionColumn: str) -> Dict[Any, pd.DataFrame]:
    """splits a DataFrame df on the unique values of a specified column 
    to return a unique DataFrame for each unique value in the column.
    
    Usage:
        from pyphocorehelpers.indexing_helpers import partition_df_dict
        partitioned_dfs = partition_df_dict(spikes_df, partitionColumn='new_epoch_IDX')
    """
    unique_values = np.unique(df[partitionColumn])
    grouped_df = df.groupby([partitionColumn])
    return dict(zip(unique_values, [grouped_df.get_group(aValue) for aValue in unique_values]))
```

**Used in:**
- [heuristic_replay_scoring.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1111) for subsequence partitioning

## Summary

The main workflow for finding contiguous time bins and creating epochs is:

1. **`contiguous_regions`** - Identifies contiguous regions in boolean arrays (most fundamental)
2. **`thresh_epochs`** - Finds epochs based on threshold crossing with merging/filtering
3. **`Epoch.from_boolean_array`** - Converts boolean time series to Epoch objects
4. **`partition_df_dict`** - Splits dataframes by contiguous groups for further analysis

These functions work together to identify contiguous time periods in your data and convert them into epoch representations for analysis.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D find_data_indicies_from_epoch_times](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20find_data_indicies_from_epoch_times)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy OR repo:github.com/CommanderPho/pyPhoCoreHelpers file:.*.py def find_contiguous](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20file:.%2A.py%20def%20find_contiguous)
- [repo:github.com/diba-lab/NeuroPy file:epoch.py](/search?q=patterntype:keyword%20repo:github.com%2Fdiba-lab%2FNeuroPy%20file:epoch.py)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy file:.*.py contiguous AND (bins OR epochs OR intervals)](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20file:.%2A.py%20contiguous%20AND%20%28bins%20OR%20epochs%20OR%20intervals%29)
- [github.com/diba-lab/NeuroPy/neuropy/utils/mathutil.py@5e2242a?L240-280](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/utils/mathutil.py?L240-280)
- [github.com/diba-lab/NeuroPy/neuropy/utils/mathutil.py@5e2242a?L195-240](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/utils/mathutil.py?L195-240)
- [github.com/diba-lab/NeuroPy/neuropy/core/epoch.py@5e2242a?L1-100](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/core/epoch.py?L1-100)
- [repo:github.com/diba-lab/NeuroPy file:epoch.py from_boolean_array](/search?q=patterntype:keyword%20repo:github.com%2Fdiba-lab%2FNeuroPy%20file:epoch.py%20from_boolean_array)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py consecutive contiguous threshold](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20consecutive%20contiguous%20threshold)
- [github.com/diba-lab/NeuroPy/neuropy/core/epoch.py@5e2242a?L750-780](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/core/epoch.py?L750-780)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers file:.*.py consecutive_differences find_data_indicies](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20file:.%2A.py%20consecutive_differences%20find_data_indicies)
- [github.com/diba-lab/NeuroPy](/github.com/diba-lab/NeuroPy)
- [github.com/diba-lab/NeuroPy/neuropy/core/epoch.py@5e2242a?L775-800](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/core/epoch.py?L775-800)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py def consecutive](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20def%20consecutive)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py split group dataframe epoch contiguous bins](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20split%20group%20dataframe%20epoch%20contiguous%20bins)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/diba-lab/NeuroPy file:.*.py np.diff contiguous consecutive](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20file:.%2A.py%20np.diff%20contiguous%20consecutive)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py epoch bin index contiguous gap break](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20epoch%20bin%20index%20contiguous%20gap%20break)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis file:.*.py partition_df file:heuristic_replay_scoring.py](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20file:.%2A.py%20partition_df%20file:heuristic_replay_scoring.py)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L1600-2000](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1600-2000)
- [repo:github.com/CommanderPho/pyPhoCoreHelpers def partition_df_dict](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20def%20partition_df_dict)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L1060-1200](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1060-1200)
- [github.com/CommanderPho/pyPhoCoreHelpers/src/pyphocorehelpers/indexing_helpers.py@066d6f3?L780-850](/github.com/CommanderPho/pyPhoCoreHelpers@066d6f3fe679ab0683169e3be844257d24301bf3/-/blob/src/pyphocorehelpers/indexing_helpers.py?L780-850)
- [github.com/diba-lab/NeuroPy/neuropy/utils/position_util.py@5e2242a?L80-120](/github.com/diba-lab/NeuroPy@5e2242ac1c9d6cac777b2db1dc4c7698169bb2b1/-/blob/neuropy/utils/position_util.py?L80-120)

## Suggested Follow-ups

- Can you provide a specific example of how to use `contiguous_regions` with a Pandas DataFrame to identify contiguous time bins based on a threshold?
- How can the `thresh_epochs` function be adapted to work directly with timestamps in a DataFrame instead of relying on sampling frequency?

