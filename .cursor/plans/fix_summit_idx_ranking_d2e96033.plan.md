---
name: Fix summit_idx ranking
overview: Assign `summit_idx` in `_build_1d_peak_prominence_df` by ranking peaks within each (neuron/aclu, trial) by `peak_height` descending (0 = tallest), instead of using SciPy left-to-right peak order.
todos:
  - id: sort-summit-idx
    content: In _build_1d_peak_prominence_df, assign summit_idx by sorting peak_heights descending within each (neuron, trial)
    status: completed
  - id: test-summit-idx
    content: Add/extend unit test asserting tallest peak gets summit_idx=0
    status: completed
isProject: false
---

# Fix `summit_idx` to rank by peak height

## Problem

In [`_build_1d_peak_prominence_df`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py) (~2384–2394), `summit_idx` is set to loop index `peak_idx`. That follows SciPy `find_peaks` spatial order (left→right), not height.

```2384:2394:h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py
            for peak_idx in np.arange(n_peaks):
                _records.append({
                    neuron_id_column_name: neuron_identifier,
                    'time_bin_idx': int(trial_idx),
                    'summit_idx': int(peak_idx),
                    ...
                })
```

## Change (single place)

Edit [`_build_1d_peak_prominence_df`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py) so that **within each `(neuron_IDX, trial_idx)`** group:

1. `sort_order = np.argsort(peak_heights)[::-1]` (highest first; stable secondary order via existing peak order on ties is fine)
2. Emit rows in that order with `summit_idx = 0, 1, 2, ...`
3. Index `prominences` / coords / heights via `sort_order`

No change to `compute_1d_peak_prominence` / `compute_1d_dt_...` tuple order or dominant-peak masks (those already use `argmax(peak_heights)`).

`compute_1d_posterior_peak_promenences` already calls `_build_1d_peak_prominence_df`, so the returned DF picks this up automatically.

## Test

Extend [`tests/test_peak_prominence2d.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/tests/test_peak_prominence2d.py) (e.g. in `test_build_df_optional_ids_and_xbins` or a small new case): synthetic curve with two peaks of unequal height; assert taller peak has `summit_idx == 0` and shorter has `summit_idx == 1`.
