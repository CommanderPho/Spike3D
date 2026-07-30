---
name: Lap spike counts df
overview: Add spike-level lap into the reliability matrix Polars aggregation so `per_tbin_aclu_per_lap_xy_spike_counts_df` is computed alongside the existing xy counts, returned as a 6th value, and stored on the decoder like its sibling fields.
todos:
  - id: compute-lap-xy
    content: "In compute_reliability_matrix: add lap Polars groupby when lap present; return as 6th value; update docstring"
    status: completed
  - id: mixin-store
    content: Unpack 6th value in mixin; store self.per_tbin_aclu_per_lap_xy_spike_counts_df; update docs/example unpack
    status: completed
  - id: decoder-field
    content: Add serialized_field + setup/post_load clear + neuron-slice in reconstruction.py and reconstruction_dst.py
    status: completed
isProject: false
---

# Add `per_tbin_aclu_per_lap_xy_spike_counts_df`

## Approach

Compute true lap-partitioned spike counts inside [`compute_reliability_matrix`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\reliability.py) from the same spike-level frame (with `t_bin_idx` / `binned_x` / `binned_y` already assigned). Keep the existing non-lap `per_tbin_aclu_xy_spike_counts_df` and confusion path unchanged. When `'lap'` is missing, return/store `None`.

```mermaid
flowchart LR
  spikes["spikes_df + t_bin_idx"] --> xyAgg["groupby aclu,t_bin,x,y"]
  spikes --> lapAgg["groupby aclu,t_bin,lap,x,y"]
  xyAgg --> existing["per_tbin_aclu_xy_spike_counts_df"]
  lapAgg --> lapDf["per_tbin_aclu_per_lap_xy_spike_counts_df"]
  existing --> confusion["perform_compute_confusion_matrix"]
```

## Changes

### 1. [`reliability.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\reliability.py) — `compute_reliability_matrix`

After the existing Polars block (~1035–1046), when `'lap' in spikes_df.columns`:

- Build a second Polars frame from `["t_bin_idx", "aclu", "lap", "binned_x", "binned_y"]`
- `group_by(["aclu", "t_bin_idx", "lap", "binned_x", "binned_y"]).agg(pl.len().alias("n_spikes"))`
- Else set `per_tbin_aclu_per_lap_xy_spike_counts_df = None`

Return it as the **6th** tuple element. Update docstring Returns. Leave the existing 5 outputs’ semantics identical (coarse `per_tbin_aclu_spike_counts_df` still sums the non-lap xy table).

Update the two unpack sites:

- Docstring example ~896
- Mixin `_perform_compute_unit_confusion_reliability_variables` ~2376–2383: unpack + `self.per_tbin_aclu_per_lap_xy_spike_counts_df = ...`; keep the public 5-tuple return of that method unchanged (storage on `self` is the API for this new table). Document the new attribute in the mixin class docstring Outputs list.

### 2. Decoder storage / slice / reset (mirror sibling field)

[`reconstruction.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py):

- Add `per_tbin_aclu_per_lap_xy_spike_counts_df: pd.DataFrame = serialized_field(default=None, ...)` next to `per_tbin_aclu_xy_spike_counts_df`
- Clear it in `setup` / `post_load`
- Neuron-slice filter by `aclu` beside the existing xy-counts block (~3862)

[`reconstruction_dst.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction_dst.py): same neuron-slice filter (~229).

## Resulting columns

`['aclu', 't_bin_idx', 'lap', 'binned_x', 'binned_y', 'n_spikes']` (spike `t_bin_idx` remains 1-based; no lap filtering — callers can drop `lap == -1` if needed).

## Non-goals

- Do not change confusion / `reliability_*` math
- Do not rewire `plot_reliability_maps_with_spikes` to use this table yet
- Do not change the mixin method’s 5-value return signature beyond storing the new attribute on `self`