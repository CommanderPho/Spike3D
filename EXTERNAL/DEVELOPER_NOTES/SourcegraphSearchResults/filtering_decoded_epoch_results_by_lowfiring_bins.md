---
title: Filtering decoded epoch results by low-firing bins
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/798e8eea-aa95-403b-9856-7337e83cbe6e
url: https://phohale.sourcegraph.app/deepsearch/382
created: 2026-07-17T22:42:04Z
updated: 2026-07-17T22:42:42Z
---

# decoded epoch result filtering out low-firing bins

Here is a comprehensive breakdown of how `DecodedFilterEpochsResult` filters out low-firing bins:

---

## Low-Firing Bin Filtering in `DecodedFilterEpochsResult`

The core method is [`mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1541-1728) on `DecodedFilterEpochsResult`. It is **pure** (returns a deep copy, does not modify `self`).

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_num_spikes_per_bin_to_be_considered_active` | `1` | Min total spikes in a bin for it to be considered active |
| `min_num_unique_active_neurons_per_time_bin` | `3` | Min number of distinct neurons that must fire |
| `masked_bin_fill_mode` | `'last_valid'` | How to handle inactive bins (see modes below) |

### How It Works

For each epoch `i`, it calls `spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask(...)` on the epoch's time bin edges to get:
- `is_time_bin_active` — boolean array, `True` where bins meet the threshold
- `inactive_mask` — inverse mask
- `mask_rgba` — for visualization

Then it applies one of four **fill modes**:

#### `'last_valid'` (default)
- Sets inactive bins to `NaN` / `-1` in `p_x_given_n`, `most_likely_position_indicies`, `most_likely_positions`
- Builds a `last_valid_indices` lookup array (O(n)) and **back-fills** all inactive bins from the most recent active bin
- If _no_ bins are active, warns and leaves everything `NaN`

#### `'dropped'`
- Physically removes inactive bins from all result arrays (`p_x_given_n_list`, `most_likely_position_indicies_list`, `most_likely_positions_list`, `spkcount`, `time_bin_edges`, `time_bin_containers`, `nbins`)
- Rebuilds a new `BinningContainer` from the surviving centers
- Handles edge cases where 0 or 1 centers remain

#### `'nan_filled'`
- Just NaN-fills the inactive bins, no backfill or dropping

#### `'ignore'`
- Does nothing at all to the decoded output

### Return Value

```python
a_decoded_result, (is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list)
```

After masking, it recomputes marginals (`marginal_x_list`, `marginal_y_list`) from the modified posteriors via `BasePositionDecoder.perform_build_marginals(...)` ([line 1711](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1711)).

### Call Sites

- [`context_dependent.py` line ~992](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py) — used with `masked_bin_fill_mode='dropped'` for continuous decoding
- Visualization helper [`_plot_low_firing_time_bins_overlay_image`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L9611-9659) overlays the `mask_rgba` on matplotlib render widgets to visually dim those bins

## Sources

- [decoded epoch result filter low firing bins](/search?q=patterntype:nls%20decoded%20epoch%20result%20filter%20low%20firing%20bins)
- [filter low firing bins minimum spikes threshold decoded epoch](/search?q=patterntype:nls%20filter%20low%20firing%20bins%20minimum%20spikes%20threshold%20decoded%20epoch)
- [DecodedFilterEpochsResult filter low firing](/search?q=patterntype:nls%20DecodedFilterEpochsResult%20filter%20low%20firing)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@0676630?L1540-1650](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1540-1650)
- [MaskedTimeBinFillMode masked_bin_fill_mode low firing reconstruction](/search?q=patterntype:nls%20MaskedTimeBinFillMode%20masked_bin_fill_mode%20low%20firing%20reconstruction)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@0676630?L1650-1720](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1650-1720)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py@0676630?L1720-1760](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py?L1720-1760)

