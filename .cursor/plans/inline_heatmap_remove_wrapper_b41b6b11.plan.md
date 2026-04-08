---
name: Inline heatmap remove wrapper
overview: "Remove `plot_pos_by_ctxt_joint_heatmap` and inline matplotlib heatmap output in `determine_decoded_context_uncertainty_as_fn_of_position` using the same pattern as `plot_peak_heatmap_test` in the same file: `mpl.colormaps.get_cmap('viridis')` with `set_bad`, an `imshow_kwargs` dict, and `visualize_heatmap(..., ax=..., defer_show=True, **imshow_kwargs)` plus manual axis labels. Fix the function return type annotation to match the existing dict return."
todos:
  - id: remove-wrapper
    content: Delete plot_pos_by_ctxt_joint_heatmap; inline visualize_heatmap + mpl cmap + save/close in determine_decoded_context_uncertainty_as_fn_of_position
    status: completed
  - id: fix-annotation-doc
    content: Update docstring (drop wrapper name); fix return type hint to Dict[str, pd.DataFrame]
    status: completed
isProject: false
---

# Inline heatmap (remove `plot_pos_by_ctxt_joint_heatmap`)

## Reference pattern (same file)

[`plot_peak_heatmap_test`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) (~12068–12111): `cmap = mpl.colormaps.get_cmap('viridis')`, `cmap.set_bad(color='black')`, build `imshow_kwargs` (`origin`, `cmap`, `interpolation`, `aspect`, `extent`, …), then call [`visualize_heatmap`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\matplotlib\visualize_heatmap.py) with `ax=curr_ax`, `defer_show=True`, and `**imshow_kwargs`. For a **new** figure, create `fig, ax = plt.subplots(...)` first, then pass `ax` into `visualize_heatmap` so figure size/dpi match your needs (the helper’s default `plt.subplots(figsize=(20,8))` when `ax is None` is too large).

## Code changes (single file)

**File:** [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)

1. **Delete** the standalone function `plot_pos_by_ctxt_joint_heatmap` and its `@function_attributes` block (currently ~4067–4093).

2. **Inside** `determine_decoded_context_uncertainty_as_fn_of_position`, in the existing `if (csv_export_out_dir is not None) or show_pos_by_ctxt_joint_figure:` loop (~4245–4263), **replace** the `plot_pos_by_ctxt_joint_heatmap(...)` call with inlined steps:
   - Local imports (consistent with neighbors): `import matplotlib.pyplot as plt` and `from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap`. The module already imports `matplotlib as mpl` at top level (~26–28); use that for `mpl.colormaps.get_cmap('viridis')` and `set_bad`, matching `plot_peak_heatmap_test`.
   - Compute `img = np.asarray(a_result_pos_by_ctxt_joint).T`, `n_pos` / `n_ctx`, `xc` from `a_decoder.xbin_centers` with the same fallback as today if length mismatch, `half_bin`, `extent = (-0.5, n_ctx - 0.5, y_bottom, y_top)`.
   - Build `imshow_kwargs`: `origin='lower'`, `interpolation='nearest'`, `aspect='auto'`, `extent`, `cmap` (with bad color), `animated=False` (not `True` unless you want blitting).
   - `fig, ax = plt.subplots(num=_plot_num, figsize=(10.0, 4.0), dpi=220, clear=True)`.
   - `fig, ax, im = visualize_heatmap(img, ax=ax, title=_plot_title, layout='none', defer_show=True, show_colorbar=True, show_xticks=False, show_yticks=False, **imshow_kwargs)`.
   - Set context axis explicitly (since `visualize_heatmap` with `show_xticks=False` clears ticks): `ax.set_xticks(np.arange(n_ctx))`, `ax.set_xticklabels(_pos_by_ctxt_context_labels)`, `ax.set_xlabel('Context')`, `ax.set_ylabel('Linearized position')`.
   - Preserve batch behavior: `if _png_save_path is not None: fig.savefig(_png_save_path, dpi=300, bbox_inches='tight')`; `if not show_pos_by_ctxt_joint_figure: plt.close(fig)`; if showing, rely on `defer_show=True` so `visualize_heatmap` does not call `plt.show()` (same idea as peak heatmap loop).

3. **Docstring** (~4103–4104): remove the parenthetical that names `plot_pos_by_ctxt_joint_heatmap`; describe optional PNGs as matplotlib heatmaps saved from this function.

4. **Return type** (~4097): the function currently returns `pre_post_delta_a_decoded_marginal_posterior_df_dict` while annotated `-> pd.DataFrame`. Per your choice, **keep returning the dict** and **update the annotation** to something accurate (e.g. `Dict[str, pd.DataFrame]`) and adjust the docstring “Returns” line if present so it matches.

## Verification

- Grep confirms `plot_pos_by_ctxt_joint_heatmap` only existed in this file; no other call sites after removal.
- No new markdown or plan files.
