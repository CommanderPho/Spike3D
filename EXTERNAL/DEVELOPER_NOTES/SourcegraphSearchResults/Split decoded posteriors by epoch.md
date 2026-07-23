---
title: Split decoded posteriors by epoch
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/7d6910cc-a251-4353-bb5f-466428e63ad4
url: https://phohale.sourcegraph.app/deepsearch/297
created: 2025-12-11T08:38:14Z
updated: 2026-01-15T10:15:29Z
---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/diba-lab/NeuroPy  Search for a function that splits decoded posteriors (I specifically have a `SingleEpochDecodedResult` object) into separate results for each Epoch using an Epoch object or epochs_df

Perfect! I found the functions. Let me create a summary with the key information:

Based on my search through the repositories, I found the functions you need to split decoded posteriors by epochs. Here are the main methods available in the [`DecodedFilterEpochsResult`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L854-L1100) class:

## Main Functions for Splitting by Epochs

### 1. **`filtered_by_epoch_times(included_epoch_start_times)`** 
[Line 1182-1190](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1182-L1190)

Filters the decoded result by epoch start times from an `epochs_df`:

```python
# Filter using epoch start times from your epochs_df
filtered_result = single_epoch_decoded_result.filtered_by_epoch_times(
    included_epoch_start_times=epochs_df['start'].to_numpy()
)
```

### 2. **`filtered_by_epochs(included_epoch_indicies)`**
[Line 1193-1231](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1193-L1231)

Filters by specific epoch indices:

```python
# Filter using epoch indices
included_epoch_indicies = [1, 3, 4]
filtered_result = single_epoch_decoded_result.filtered_by_epochs(included_epoch_indicies)
```

### 3. **`get_result_for_epoch(active_epoch_idx)`**
[Line 936-967](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L936-L967)

Gets a `SingleEpochDecodedResult` for a specific epoch index:

```python
# Get result for a single epoch by index
single_result = decoded_result.get_result_for_epoch(active_epoch_idx=0)
```

### 4. **`get_result_for_epoch_at_time(epoch_start_time)`**
[Line 970-979](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L970-L979)

Gets a `SingleEpochDecodedResult` for a specific epoch start time:

```python
# Get result for a single epoch by start time
single_result = decoded_result.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
```

## Usage Examples from the Codebase

Here are real examples from the codebase:

```python
# Example 1: Filter by epoch times from a DataFrame
filtered_decoder_filter_epochs_decoder_result_dict = {
    a_name: a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) 
    for a_name, a_result in decoder_filter_epochs_decoder_result_dict.items()
}

# Example 2: Filter using start times only
included_epoch_start_times = replay_epochs_df['start'].to_numpy()
filtered_result = a_result.filtered_by_epoch_times(included_epoch_start_times)

# Example 3: Get single epoch result
a_result = v.get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
```

## Key Points

- Both `filtered_by_epoch_times()` and `filtered_by_epochs()` return a new `DecodedFilterEpochsResult` with only the selected epochs
- They properly slice all epoch-related fields: `p_x_given_n_list`, `most_likely_positions_list`, `time_bin_containers`, etc.
- The `filtered_by_epoch_times()` method internally calls `filtered_by_epochs()` after finding the matching indices
- These methods work with both `Epoch` objects and `pd.DataFrame` (they convert automatically)

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis split decoded posterior epoch SingleEpochDecodedResult](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20split%20decoded%20posterior%20epoch%20SingleEpochDecodedResult)
- [repo:github.com/CommanderPho/Spike3D split decoded posterior epoch SingleEpochDecodedResult](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FSpike3D%20split%20decoded%20posterior%20epoch%20SingleEpochDecodedResult)
- [repo:github.com/diba-lab/NeuroPy split decoded posterior epoch](/search?q=patterntype:nls%20repo:github.com%2Fdiba-lab%2FNeuroPy%20split%20decoded%20posterior%20epoch)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis SingleEpochDecodedResult](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20SingleEpochDecodedResult)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis subdivide split epochs_df DecodedFilterEpochsResult](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20subdivide%20split%20epochs_df%20DecodedFilterEpochsResult)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis DecodedFilterEpochsResult get_result_for_epoch](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20DecodedFilterEpochsResult%20get_result_for_epoch)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@26f673d?L854-1100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L854-1100)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis filtered_by_epoch](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20filtered_by_epoch)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@26f673d?L1181-1250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1181-1250)

## Suggested Follow-ups

- Can you provide an example of how to use `filtered_by_epoch_times` with a `SingleEpochDecodedResult` object and a pandas DataFrame containing epoch start and stop times?
- How do the filtering methods handle edge cases, such as when a time bin falls exactly on an epoch boundary?


---

# These are close, but they actually just filter `DecodedFilterEpochsResult`objects which are ALREADY split up by epochs. I need to get to a `DecodedFilterEpochsResult`-like form from a `SingleEpochDecodedResult` or its contents

Perfect! Now I have the complete picture. Let me provide you with the answer:

You're right that the filtering functions work on already-split `DecodedFilterEpochsResult` objects. To go from a `SingleEpochDecodedResult` (continuous decoding) to a `DecodedFilterEpochsResult` split by epochs, you need to:

## Solution: Use the Decoder's `decode_specific_epochs` Method

The key is that you **don't split an existing `SingleEpochDecodedResult`** - instead, you **re-decode using your epochs**:

```python
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

# Given:
# - a_decoder: your decoder object
# - spikes_df: your spikes dataframe  
# - epochs_df: your DataFrame or Epoch object with the epochs you want
# - decoding_time_bin_size: your desired time bin size (e.g., 0.025)

# Decode the specific epochs to get a DecodedFilterEpochsResult
decoded_epochs_result: DecodedFilterEpochsResult = a_decoder.decode_specific_epochs(
    spikes_df=spikes_df,
    filter_epochs=epochs_df,  # Can be pd.DataFrame or Epoch object
    decoding_time_bin_size=decoding_time_bin_size,
    debug_print=False
)

# Now decoded_epochs_result is split by your epochs!
# Each epoch can be accessed:
single_epoch_result = decoded_epochs_result.get_result_for_epoch(epoch_idx=0)
```

## Alternative: Build Pseudo-Epochs from Time Bins

If you already have a `SingleEpochDecodedResult` and want to create an epochs_df that represents each time bin as an epoch:

```python
# From a SingleEpochDecodedResult (continuous result)
single_continuous_result: SingleEpochDecodedResult = results.continuous_results['long'].get_result_for_epoch(0)

# Build an epochs_df where each time bin is treated as an epoch
decoding_bins_epochs_df: pd.DataFrame = single_continuous_result.build_pseudo_epochs_df_from_decoding_bins()

# This gives you a DataFrame with columns: ['start', 'stop', 'duration', 'label']
# where each row represents one time bin from your continuous decoding
```

## Key Points

1. **`decode_specific_epochs()`** ([Line 2122](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L2122)) is the main function to decode multiple epochs
2. **`build_pseudo_epochs_df_from_decoding_bins()`** ([Line 685](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L685)) creates an epochs_df from the time bins in a continuous result
3. You cannot directly "split" a `SingleEpochDecodedResult` - you need to re-decode with the desired epochs

The workflow is: **Decoder + Spikes + Epochs → `decode_specific_epochs()` → `DecodedFilterEpochsResult`**

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis decode_specific_epochs OR perform_decode_specific_epochs](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20decode_specific_epochs%20OR%20perform_decode_specific_epochs)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis split continuous result epochs subdivide time bins](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20split%20continuous%20result%20epochs%20subdivide%20time%20bins)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis SingleEpochDecodedResult to DecodedFilterEpochsResult](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20SingleEpochDecodedResult%20to%20DecodedFilterEpochsResult)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@26f673d?L2119-2250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L2119-2250)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis continuous_result split OR continuous flatten](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20continuous_result%20split%20OR%20continuous%20flatten)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis time_sliced time slice continuous posterior epochs](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20time_sliced%20time%20slice%20continuous%20posterior%20epochs)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@26f673d?L429-750](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L429-750)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis build_pseudo_epochs_df_from_decoding_bins](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20build_pseudo_epochs_df_from_decoding_bins)

## Suggested Follow-ups

- Can you provide a complete runnable example of using `decode_specific_epochs` with a sample decoder, spikes_df, and epochs_df?
- How does `decode_specific_epochs` handle overlapping epochs, and what are the implications for the resulting `DecodedFilterEpochsResult`?

