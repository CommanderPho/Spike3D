---
name: Rename speed filter kwargs
overview: Rename `PfND.filtered_by_speed`'s ambiguous `speed_thresh` kwarg to `minimum_speed_thresh` so call sites and the body match keep-speed-at-or-above semantics, then update the two callers.
todos:
  - id: rename-kwarg-def
    content: Rename speed_thresh -> minimum_speed_thresh in filtered_by_speed signature, body, and docstring
    status: completed
  - id: update-callers
    content: Update PfND.setup and rtc_clusterless_adapters keyword args to minimum_speed_thresh=...
    status: completed
isProject: false
---

# Rename `filtered_by_speed` kwargs for keep-above semantics

## Intent

`PfND.filtered_by_speed` keeps samples with `speed >= thresh` (minimum running speed). The bare name `speed_thresh` is ambiguous next to replay APIs that use `maximum_speed_thresh`. Rename the function kwarg only — leave [`PlacefieldComputationParameters.speed_thresh`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py) unchanged.

## Rename

In [`placefields.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py) `PfND.filtered_by_speed`:

- `speed_thresh: Optional[float]` → `minimum_speed_thresh: Optional[float]`
- Leave `speed_column_override_name` and `debug_print` as-is (they do not encode keep-above vs keep-below)
- Update the method docstring to state: keep contiguous runs of samples with speed at or above `minimum_speed_thresh`; `None` skips the cut; `0` keeps non-negative speeds
- Replace internal uses / comments that say `speed_thresh` with `minimum_speed_thresh`

## Call sites (keyword-only updates)

- [`placefields.py` ~985](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py): `speed_thresh=self.config.speed_thresh` → `minimum_speed_thresh=self.config.speed_thresh`
- [`rtc_clusterless_adapters.py` ~175](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/rtc_clusterless_adapters.py): same keyword rename; still pass `pf.config.speed_thresh`

No other `filtered_by_speed(` call sites exist in the workspace.

## Out of scope

- Renaming `PlacefieldComputationParameters.speed_thresh` / config fields
- Changing inequality logic or contiguous-run behavior
- Adding a deprecated `speed_thresh=` alias
