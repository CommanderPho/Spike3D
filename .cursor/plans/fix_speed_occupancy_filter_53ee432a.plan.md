---
name: Fix speed occupancy filter
overview: The zero-neuron crash is the inverted placefield speed cut emptying lap epochs during rebuild. Aclu slicing is not on this stack. Fix `PfND.filtered_by_speed` so it keeps running samples as separate intervals, and update the test that currently locks the slow-span result.
todos:
  - id: fix-speed-filter
    content: Keep speed >= speed_thresh and emit one interval per contiguous run in PfND.filtered_by_speed
    status: completed
  - id: update-rtc-test
    content: Update the RTC training-mask test to expect the running samples, not the slow span
    status: completed
isProject: false
---

# Fix the zero-neuron placefield rebuild

The traceback does not go through aclu slicing. `ComputeGlobalEpochBase.recompute` rebuilds each 1D decoder with `BasePositionDecoder.replacing_computation_epochs` on `.laps`, which constructs a new `PfND(..., compute_on_init=True)`. That calls `PfND.setup` → `PfND.filtered_by_speed`. `get_by_id` is not called.

`filtered_by_speed` keeps `speed < speed_thresh` (stationary). Laps are running, so the kept set is empty, `compute()` builds a ratemap with no cells, and `BasePositionDecoder._setup_computation_variables` raises. The loaded `pf1D` objects still have neurons; only this rebuild is empty. Leave the `.laps` line in [`EpochComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py) as it is. Leave `PfND.get_by_id` and `BasePositionDecoder.get_by_id` alone.

## Change `PfND.filtered_by_speed`

In [`placefields.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) around line 1517:

- Keep samples with `speed >= speed_thresh`. `speed_thresh` is a minimum running speed (KDIBA default 10 cm/s). `speed_thresh == 0` means no cut, and speed is non-negative, so `>= 0` keeps every sample. `None` still skips the cut. This matches the unused pre-2023-11-14 branch’s intent (`speed > speed_thresh`, and the comment that time below the threshold must be removed from occupancy).
- Stop collapsing each epoch with `groupby(...).agg(first, last)`. That span re-includes the gaps. After dropping non-epoch samples (`decoder_epoch_id != -1`), emit one `[start, stop]` per contiguous run of passing samples inside each epoch id: `start` is the first passing `t`, `stop` is the last passing `t`. Do not use `detect_epoch_satisfying_condition`’s falling-edge stop; that timestamp is the first failing sample.
- Return the same columns callers already use (`start`, `stop`, `label`, `duration`, `n_samples`). An empty pass returns an empty frame with those columns.
- Correct the comment that says the `-1` rows are “below the speed.” Those rows are outside the input epochs.

Callers that should keep working without edits:

- `PfND.setup` time-slices spikes and position with the returned `starts`/`stops`.
- [`_pfnd_speed_filtered_training_intervals`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\rtc_clusterless_adapters.py) only forwards this result into the RTC training mask.

## Update the test that locks the bug

[`test_build_is_training_mask_matches_pfnd_filtered_by_speed_intervals`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_rtc_clusterless_decoder.py) uses speeds `[0, 5, 15, 5, 20]` and `speed_thresh=10`, and expects `[0.0, 3.0]` with mask `[True, True, True, False]`. That is the slow span, and it includes the fast sample at `t=2` only because of the first/last collapse.

After the fix the passing samples are `t=2` and `t=4`, so the intervals are `[[2.0, 2.0], [4.0, 4.0]]`. Point the mask times at `0, 2, 3, 4` and expect `[False, True, False, True]`.
