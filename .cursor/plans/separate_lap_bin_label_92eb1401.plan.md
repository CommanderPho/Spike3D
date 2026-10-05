---
name: Separate lap bin label
overview: Stop requiring lap and ripple decoding bins to match in the stacked-epoch posterior export, and record each size on the output context when they differ.
todos:
  - id: drop-equal-bin-assert
    content: Remove the equal-bin assert and write laps_time_bin_size plus ripple_time_bin_size on the export context when the sizes differ
    status: completed
isProject: false
---

# Label laps with their own time bin

In [`_display_directional_merged_pf_decoded_stacked_epoch_slices`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (around lines 10366–10424), one context field is set from the ripple bin only:

```python
assert (ripple_decoding_time_bin_size == laps_decoding_time_bin_size), ...
active_context = deepcopy(active_context).overwriting_context(time_bin_size=ripple_decoding_time_bin_size)
```

The image export already writes laps and ripples from separate result dicts, each with its own `decoding_time_bin_size`. The assert only protects that one context field. The session folder then drops `time_bin_size` from `get_description`, so the field is not in the path today.

## Change

Remove the assert and the TODO that assumes the sizes are equal.

- When the two sizes are equal, keep the current `overwriting_context(time_bin_size=...)` and keep excluding `time_bin_size` from the session-folder description.
- When they differ, do not write a single `time_bin_size`. Write `laps_time_bin_size=laps_decoding_time_bin_size` and `ripple_time_bin_size=ripple_decoding_time_bin_size`. Exclude a leftover `time_bin_size` from the folder description so an incoming context cannot keep the ripple-only label, and leave the two new keys in the description so the output path records both.

No change to [`perform_export_all_decoded_posteriors_as_images`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\data_exporting.py). `laps/` and `ripple/` stay separate folders under that session path.
