---
name: Fix paginator DataFrame
overview: Fix AttributeError in custom-sort resync by making `_subfn_helper_build_paginator` accept a DataFrame (it already does later for slices, but the label bootstrap still assumes Epoch).
todos:
  - id: fix-paginator-df
    content: Update _subfn_helper_build_paginator to normalize Epoch/DataFrame before n_epochs/label access
    status: completed
isProject: false
---

# Fix paginator DataFrame AttributeError

## Cause

[`resync_pagination_controller_epoch_order`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) correctly passes a **DataFrame** into `_subfn_helper_build_paginator` so `Epoch()` cannot re-sort by `start`. That helper already handles DataFrame when building `epoch_slices`, but the **label bootstrap** still assumes Epoch:

```1336:1341:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py
        epoch_labels = filter_epochs_decoder_result.epoch_description_list.copy()
        if epoch_labels is None or len(epoch_labels) < active_filter_epochs.n_epochs:
            if 'label' not in active_filter_epochs._df.columns:
                active_filter_epochs._df['label'] = active_filter_epochs._df.index.to_numpy() # integer ripple indexing
            active_labels = active_filter_epochs._df['label'].to_numpy()
```

→ `AttributeError: 'DataFrame' object has no attribute 'n_epochs'`.

## Change

In [`stacked_epoch_slices.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py) `_subfn_helper_build_paginator`:

1. Normalize `active_filter_epochs` to `epochs_df` first (`DataFrame` as-is, `Epoch` via `.to_dataframe()`).
2. Use `n_epochs = len(epochs_df)` for the label-length check.
3. Read/write `label` on `epochs_df.columns` (not `._df`).
4. Build `epoch_slices` from that same `epochs_df` (existing logic).

No change needed in `PhoDiba2023Paper.py` resync path once the helper accepts DataFrame end-to-end.

## Verify

Re-run the notebook viewer init with `sort_by='best_overall_wcorr_z'`: no AttributeError, heatmaps visible, page 1 not strictly time-ordered.