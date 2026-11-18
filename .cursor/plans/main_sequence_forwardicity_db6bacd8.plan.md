---
name: Main sequence forwardicity
overview: Score forwardicity on the heuristic main sequence with intrusion bins removed, and show that score on the epoch-slice plots.
todos:
  - id: helper
    content: Add main_sequence_positions() that returns is_main, non-intrusion pos values in flat_idx order
    status: completed
  - id: callback
    content: Score the plot label from that array via decoded_sequence_and_heuristics_curves_data[epoch_start], falling back to the full posterior
    status: completed
isProject: false
---

# Main-sequence forwardicity

`forwardicity_score` in [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) already accepts `most_likely_positions_arr`. The notebook call is the right entry point, but `longest_sequence_subsequence` is the wrong array.

That property is the merged subsequence with the most bins (`np.nanargmax` of raw length). The merge still concatenates the short intrusion fragments into that array, so `np.diff` still counts the hops. The ranked main sequence is the `is_main` row (`main_subsequence_ranking_columns` starts with `len_excluding_intrusions`), and its positions live on `position_bins_info_df` (`pos`, `subsequence_idx`, `is_intrusion`, `flat_idx`). Those two subsequences can differ; the plot triangles follow raw length and already have a note that they can disagree with `is_main`.

## Selection

Add a helper next to `forwardicity_score` that returns time-ordered positions for the `is_main` subsequence with intrusion rows dropped:

- `subsequence_idx` from `partition_result.subsequences_df` where `is_main`
- rows of `position_bins_info_df` with that `subsequence_idx` and `is_intrusion == False`
- sort by `flat_idx`, take `pos`

Repeats stay in the array. `forwardicity_score` already drops zero steps. A main sequence shorter than 2 bins stays `NaN`.

## Plot callback

In `ForwardicityPaginatedPlotDataProvider._callback_update_curr_single_epoch_slice_plot`, look up the partition the same way the heuristic overlay does: `plots_data.decoded_sequence_and_heuristics_curves_data[epoch_slice[0]].partition_result` (keyed by epoch start in `decoder_build_single_decoded_sequence_and_heuristics_curves_data`). Pass those positions as `most_likely_positions_arr` and keep the existing LR=−1 / RL=+1 map.

If that overlay was not added, or the epoch key is missing, keep the current full-posterior MAP score. Label the main-sequence value so the two modes are distinguishable, for example `main fwdicty: 0.67`.

## Notebook call

The attached cell should pass the helper output, not `longest_sequence_subsequence`:

```python
positions = main_sequence_positions(a_seq_and_heuristics_result)
forwardicity, (ratio_major_aligned_bins, ratio_decoder_aligned_bins) = forwardicity_score(most_likely_positions_arr=positions, most_likely_decoder_direction=most_likely_decoder_direction, debug_print=False)
```

Index the dict by epoch start (`a_tuple.start`), not `list(values())[an_epoch_idx]`, when the start time is known. No notebook edit unless you want that cell updated in place.
