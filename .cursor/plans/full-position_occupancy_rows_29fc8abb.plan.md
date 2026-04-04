---
name: Full-position occupancy rows
overview: Replace the stub block at lines 522–536 in `BinnedOccupancyComparisons.plot_decoded_and_measured_occupancies` with a loop over `decoder_names`, compute full-session binned occupancy (samples + seconds) per decoder using that decoder's grid, and add two new image rows to the existing pyqtgraph layout—mirroring the pattern of the measured decoder occupancy rows above.
todos:
  - id: loop-decoders
    content: Replace L522–536 with loop over decoder_names; build occ_samples/occ_sec per nm using curr_sess.position + pf1D_Decoder_dict[nm] bin edges
    status: completed
  - id: plot-two-rows
    content: Add _subfn_add_single_row calls for samples row and seconds row with shared scales and subtitles
    status: completed
  - id: remove-dead
    content: Drop duplicate a_decoder_name lines, unused laps_df in block, and dangling tuple display
    status: completed
isProject: false
---

# Full-position occupancy for all decoder columns

## Context

- The comparison figure is built in `[plot_decoded_and_measured_occupancies](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)` (~L352–543). `decoder_names` defaults to `list(pf1D_Decoder_dict.keys())` and drives column count and labels (L367–369, L432–433).
- Rows 472–479 and 512–519 show **decoder PF** occupancies (`pf1D_Decoder_dict[nm].pf.occupancy` and `.probability_normalized_occupancy`), which reflect how the place-field pipeline defined occupancy (epochs used for the decoder), not necessarily every position sample.
- `[compute_binned_position_occupancy](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py)` (L926–978) bins **all** rows in the position `DataFrame` after adding bin columns; it returns `(occupancy_n_samples, occupancy_seconds)` (tuple; docstring return type is stale). Bin alignment uses `xbin_edges` / `ybin_edges` from each decoder—same grid as the PF so maps are comparable.

## What “last two occupancies” means here

The stub unpacks two arrays from one call: `**occupancy_n_samples`** and `**occupancy_seconds`**. Implement **two new rows** in the grid:

1. **All-position occupancy (sample counts)** — same interpretation as “raw” counts on the decoder grid.
2. **All-position occupancy (seconds)** — counts × `1/position_sampling_rate_Hz` (per existing helper).

Use the same visualization helpers as neighboring rows: `_row_shared_limits_percentile_cap` (or `_row_shared_limits` if you prefer linear scaling) and `_subfn_add_single_row` (L397–413).

## Implementation steps (single file)

**File:** `[pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)`

1. **Remove** the duplicate `a_decoder_name` assignments and the unused `laps_df` in this block (L523–524, L526); laps are not needed for full-position occupancy.
2. **Loop** `for nm in decoder_names:` (not hard-coded `'roam'`/`'maze_GLOBAL'`) so the number of panels always matches `pf1D_Decoder_dict` and the figure stays internally consistent—when your dict keys are exactly those two names, behavior matches your intent.
3. **Per decoder name `nm`:**
  - `curr_sess = curr_active_pipeline.filtered_sessions[nm]`
  - `a_decoder = pf1D_Decoder_dict[nm]`
  - `position_sampling_rate_Hz = curr_sess.position_sampling_rate`
  - `pos_obj = curr_sess.position`
  - Update metadata as in the existing snippet (L530–532) so `compute_binned_position_occupancy` can resolve rate if needed (L960–963 in `position.py`).
  - `pos_df = pos_obj.to_dataframe()`
  - `occ_samples, occ_sec = pos_df.position.compute_binned_position_occupancy(xbin_edges=a_decoder.xbin, ybin_edges=a_decoder.ybin, position_sampling_rate_Hz=position_sampling_rate_Hz)`
  - Append to two lists/dicts keyed by column order (e.g. `all_pos_occ_n_samples[nm]`, `all_pos_occ_sec[nm]`).
4. **Add two rows** after the loop:
  - Build `column_data` as `[(array_for_nm, short subtitle) for nm in decoder_names]` for samples, then repeat for seconds.
  - Compute shared `vmin`/`vmax` per row from the list of arrays (same pattern as L476–478).
  - Call `_subfn_add_single_row` with clear `row_side_label` values (e.g. `"All-pos occupancy (n)"` and `"All-pos occupancy (s)"` or similar) and appropriate `colorbar_label` (`"Samples"` / `"Occupancy (sec)"`).
5. **Optional cleanup:** Delete the trailing bare expressions `occupancy_n_samples, occupancy_seconds` (L535–536) once values are consumed by plotting.

## Notes / non-goals

- No changes required in NeuroPy unless you want to fix the inaccurate `-> pd.DataFrame` annotation on `compute_binned_position_occupancy`; out of scope unless you ask.
- If a decoder lacks 2D `ybin`, `compute_binned_position_occupancy` already branches on columns in `perform_add_binned_position_columns`; keep passing `xbin`/`ybin` from `a_decoder` as today.

