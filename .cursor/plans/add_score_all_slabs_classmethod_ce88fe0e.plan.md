---
name: Add score_all_slabs classmethod
overview: Add a classmethod to `PeakPromenenceMetrics` that takes a `PosteriorPeaksPeakProminence2dResult` object and calls `score_slab_quality` for each slab it contains, returning scores for all slabs.
todos: []
---

# Ad

d score_all_slabs_quality classmethod to PeakPromenenceMetrics

## Overview

Add a new classmethod `score_all_slabs_quality` to the `PeakPromenenceMetrics` class that:

1. Takes a `PosteriorPeaksPeakProminence2dResult` object as input
2. Iterates through all slabs in the result object
3. Calls `score_slab_quality` for each slab
4. Returns a dictionary of scores keyed by `(epoch_idx, time_bin_idx)` tuples

## Implementation Details

### Location

Add the method to [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py) in the `PeakPromenenceMetrics` class, after the existing `score_slab_quality` method (around line 2139).

### Method Signature

```python
@classmethod
def score_all_slabs_quality(cls, a_pf_promenence_result_obj: PosteriorPeaksPeakProminence2dResult, 
                            max_reasonable_peak_distance: float = None, 
                            min_contour_size_threshold: float = 0.5,
                            close_peak_distance_threshold: float = None) -> Dict[DecodedEpochTimeBinIndexTuple, dict]:
```



### Implementation Steps

1. Extract `xbin_centers` and `ybin_centers` from `a_pf_promenence_result_obj.xx` and `a_pf_promenence_result_obj.yy`
2. Extract `flat_peaks_df` from `a_pf_promenence_result_obj.flat_peaks_df`
3. Iterate through `a_pf_promenence_result_obj.results.items()` to get each `(epoch_idx, time_bin_idx)` tuple and its `slab_result_dict`
4. For each slab:

- Filter `flat_peaks_df` to get the `posterior_peaks_df` for that specific slab using:
     ```python
                    slab_posterior_peaks_df = flat_peaks_df[
                        (flat_peaks_df['epoch_idx'] == epoch_idx) & 
                        (flat_peaks_df['time_bin_idx'] == time_bin_idx)
                    ]
     ```




- Call `cls.score_slab_quality()` with:
    - `slab_result_dict` from the results dict
    - `slab_posterior_peaks_df` (filtered DataFrame)
    - `xbin_centers` and `ybin_centers` from the result object
    - The optional parameters passed to the method

5. Store each score result in a dictionary keyed by `(epoch_idx, time_bin_idx)`
6. Return the dictionary of all scores

### Return Value

Returns a `Dict[DecodedEpochTimeBinIndexTuple, dict]` where:

- Keys are `(epoch_idx, time_bin_idx)` tuples
- Values are the score dictionaries returned by `score_slab_quality` (containing `overall_score`, `is_well_localized`, and `score_components`)

### Documentation