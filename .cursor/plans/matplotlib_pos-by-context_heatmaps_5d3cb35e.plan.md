---
name: Matplotlib pos-by-context heatmaps
overview: Replace the PyQt `BasicBinnedImageRenderingWindow` + `.show()` block in `determine_decoded_context_uncertainty_as_fn_of_position` with a matplotlib heatmap helper patterned after existing `plot_*` utilities (optional `savefig`, `show_figure`, `plt.close`), and hook PNG export into the same `enable_export_path` / `register_output_file` flow as the CSV branch.
todos:
  - id: add-plot-helper
    content: Add plot_pos_by_ctxt_joint_heatmap (or similar) with save_path/show_figure/close semantics matching plot_linearized_position_prob_p
    status: completed
  - id: wire-export-loop
    content: Replace Qt viewer loop with helper calls; save+register+close when csv_export_out_dir set; optional show flag
    status: completed
  - id: cleanup-return
    content: Remove undefined a_laps_decoded_marginal_posterior_df line; add return a_decoded_marginal_posterior_df and docstring note for PNG
    status: completed
isProject: false
---

# Matplotlib batch output for `result_pos_by_ctxt_joint`

## What the Qt code does today

In [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py), `_subfn_compute_pre_post_delta_pos_by_ctxt_joint` builds `result_pos_by_ctxt_joint` with shape **`(n_contexts, n_pos_bins)`** (contexts = `P_Long`, `P_Short`; see lines 4151–4174). The viewer passes **`a_result_pos_by_ctxt_joint.T`**, i.e. **`(n_pos_bins, n_contexts)`**, into [`BasicBinnedImageRenderingWindow.__init__`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\BinnedImageRenderingWindow.py) as `matrix`, with default `xbins`/`ybins` (lines 905–944).

## Target pattern in this file

Match helpers like [`plot_linearized_position_prob_p`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) (~5941–6018): create a figure/axes, plot, optional **`savefig`**, **`show_figure`** gate, **`plt.close(fig)`** when not showing, **`return fig, ax, ...`**. This mirrors [`MeasuredVsDecodedOccupancy.plot_meas_vs_decoded_occupancy`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) (~3972–4064), which returns figures without forcing a GUI event loop.

## Implementation plan

1. **Add a small matplotlib helper** (new function in the same module, placed near other `plot_*` utilities—e.g. just above `determine_decoded_context_uncertainty_as_fn_of_position` ~4067, or next to `plot_linearized_position_prob_p` ~5940—your choice for locality vs. “all plot helpers together”).

   Suggested responsibilities (single-line signatures per your style rules):

   - Inputs: `result_pos_by_ctxt_joint` **`(n_contexts, n_pos_bins)`**, `context_labels` (e.g. `('P_Long', 'P_Short')`), `xbin_centers` from **`a_decoder.xbin_centers`**, `title`, plus `save_path: Optional[Path]=None`, `show_figure: bool=False`, `figsize`/`dpi` optional.
   - Build **`img = np.asarray(result_pos_by_ctxt_joint).T`** so the matplotlib image matches the Qt orientation (position on rows, context on columns), unless you prefer `origin='lower'` + `extent`—either way, **keep axes labels explicit** (position vs context).
   - Use **`ax.imshow`** (or `pcolormesh` if you want non-uniform bin edges) with **`aspect='auto'`**, **`interpolation='nearest'`**, colormap **`viridis`** (same default string as `BasicBinnedImageRenderingWindow`), and a **`colorbar`** for interpretability.
   - Set ticks: e.g. context indices on one axis, position bin indices or **`xbin_centers`** on the other (if using centers, use `extent` carefully with `imshow` pixel boundaries or prefer `pcolormesh` with bin edges).

2. **Replace lines ~4214–4225** in `determine_decoded_context_uncertainty_as_fn_of_position`:

   - Remove the TODO, the **inner** `from ... BasicBinnedImageRenderingWindow` import (redundant with the module-level import at [lines 66–67](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)), **`_out_viewer_dict`**, and **`.show()`**.
   - Loop over `pre_post_delta_result_pos_by_ctxt_joint_dict.items()` and call the helper per `a_pre_post_delta`.
   - **When `csv_export_out_dir is not None`**: write **`fig.savefig(...)`** to the same `output/` folder using the **same sanitization stem pattern** as the CSV loop (`sanitize_filename_for_Windows`, session, delta, date, `tbin`), e.g. `..._pos_by_ctxt_joint_{delta}_tbin{t}.png`; **`resolve()`** the path; **`curr_active_pipeline.register_output_file`** with metadata `kind='pos_by_ctxt_joint_png'` (or similar), `pre_post_delta`, `time_bin_size`, `source_fn='determine_decoded_context_uncertainty_as_fn_of_position'`; wrap in **`try`/`except`** with WARN like the CSV block; then **`plt.close(fig)`** so batch runs do not leak figures.
   - **When not exporting**: default **`show_figure=False`** and **`plt.close(fig)`** after creation (batch-safe). Optionally add a keyword on `determine_decoded_context_uncertainty_as_fn_of_position`, e.g. **`show_pos_by_ctxt_joint_figure: bool=False`**, passed through to the helper so notebooks can opt into display without Qt.

3. **Notebook residue at line ~4224**: `a_laps_decoded_marginal_posterior_df` is **undefined** in this function (only occurrence in the file). Remove that bare expression (and the stale `# _out_viewer_dict` comment) as part of the same minimal cleanup so the function body is executable.

4. **Return type / docstring (optional but recommended)**: The function is annotated `-> pd.DataFrame` but currently has **no `return`** before the next top-level section. When editing this region, add **`return a_decoded_marginal_posterior_df`** (or the frame you intend to expose) and extend the docstring to mention optional PNG output when `enable_export_path` is set.

## Files to touch

- Only [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py): new helper + replace Qt block + small cleanup/return/docstring as above.

## Out of scope

- Changing `_subfn_compute_pre_post_delta_pos_by_ctxt_joint` math or CSV export logic.
- Removing the **module-level** `BasicBinnedImageRenderingWindow` import unless you later confirm it is unused elsewhere in this huge module (it is still referenced elsewhere in the same file, e.g. ~8031).
