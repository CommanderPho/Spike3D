---
name: Combined pos-by-context heatmaps
overview: Replace the per–pre/post-delta loop that opens separate figures with one 1×N subplot figure, shared y-axis, identical `vmin`/`vmax` on every `imshow`, and a single figure-level colorbar. Export and `register_output_file` become one combined PNG (plus a short docstring tweak).
todos:
  - id: replace-loop
    content: Replace 4206–4242 loop with ordered keys, shared vmin/vmax, subplots(1,n,sharey), visualize_heatmap(..., show_colorbar=False), fig.colorbar once, single save/register
    status: completed
  - id: docstring
    content: Update determine_decoded_context_uncertainty_as_fn_of_position docstring PNG sentence (~4074–4075)
    status: completed
isProject: false
---

# Side-by-side pre/post-delta `pos_by_ctxt_joint` heatmap

## Context

In [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py), `determine_decoded_context_uncertainty_as_fn_of_position` (lines ~4202–4242) currently iterates `pre_post_delta_result_pos_by_ctxt_joint_dict.items()` and, for each key, builds its own `plt.subplots`, calls [`visualize_heatmap`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\matplotlib\visualize_heatmap.py) with `show_colorbar=True`, and saves one PNG per delta.

[`visualize_heatmap`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\matplotlib\visualize_heatmap.py) forwards `**imshow_kwargs` to `ax.imshow` (line 34), so **`vmin` and `vmax` are already supported** without changing that helper. It adds a colorbar only when `show_colorbar=True` (lines 38–40); for a shared bar, call it with **`show_colorbar=False`** on every panel and attach **`fig.colorbar(..., ax=axes)`** once.

## Implementation (single block replacing the current `for` loop)

1. **Stable panel order**  
   Build an ordered key list so “pre” is left of “post” when both exist, without relying on dict iteration order:

   - Start with keys present in `('pre-delta', 'post-delta')` in that order.
   - Append any remaining keys from the dict (e.g. sorted) so behavior stays sensible if partitions change.

2. **Skip empty**  
   If the ordered list is empty, do nothing (same effective behavior as today when the dict is empty).

3. **Shared data limits**  
   For each key, compute `img = np.asarray(array).T` as today. Compute **`vmin` / `vmax`** once over the **finite** values of all panels, e.g. `np.nanmin` / `np.nanmax` on a stacked array (or min/max over per-panel stats). Use the same `_cmap` with `set_bad` once; pass **`vmin`, `vmax`, `cmap`** in `imshow_kwargs` for every panel so the colormap is identical.

4. **One figure, N columns**  
   - `fig, axes = plt.subplots(1, n, num='pos_by_ctxt_joint_combined', figsize=(10.0 * n, 4.0), dpi=220, sharey=True, clear=True)`  
   - Normalize `axes` to a length-`n` 1D array for indexing.

5. **Per-panel `visualize_heatmap`**  
   Reuse the existing extent / `xc` / `half_bin` logic (identical `n_pos` across partitions in normal use). For each `i, key`:

   - `visualize_heatmap(img, ax=axes[i], title=key, layout='none', defer_show=True, show_colorbar=False, show_xticks=False, show_yticks=False, **imshow_kwargs)`  
   - Set x-ticks/labels and x-label on **each** panel (`Context`).  
   - Set **`ax.set_ylabel('Linearized position')` only on `axes[0]`** so shared-y does not repeat the label.

6. **Single colorbar**  
   After the loop: `fig.colorbar(first_mappable, ax=axes.tolist(), shrink=...)` (optional `fraction`/`pad` if layout is tight). Any panel’s `ScalarMappable` is fine because all share the same `norm` via matching `vmin`/`vmax`/`cmap`.

7. **Optional overall title**  
   e.g. `fig.suptitle('result_pos_by_ctxt_joint per Pos X', y=1.02)` — optional; keep minimal if you prefer only per-panel titles.

8. **Save / close / register (one artifact)**  
   - When `csv_export_out_dir` is set, use **one** filename, e.g. include all sanitized delta keys joined (or a fixed `combined` slug plus ordered delta parts) plus existing `_date`, `_sess`, `_tbin` pattern.  
   - `fig.savefig` once; `plt.close(fig)` when `not show_pos_by_ctxt_joint_figure`.  
   - **`register_output_file` once** with metadata such as `pre_post_deltas` = ordered list/tuple of keys (and keep `kind='pos_by_ctxt_joint_png'`, `time_bin_size`, `source_fn`).

9. **Docstring**  
   Adjust the sentence at lines 4074–4075 that says PNGs are **per** pre/post-delta to describe a **single combined** side-by-side figure (minimal edit).

## Downstream

[`batch_user_completion_helpers.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) only calls the function; no change required unless something external globbed `*_pos_by_ctxt_joint_*` per delta — the new filename pattern should remain grep-friendly (`pos_by_ctxt_joint` substring preserved).

## Testing suggestion

Run `determine_decoded_context_uncertainty_as_fn_of_position` with `show_pos_by_ctxt_joint_figure=True` and two partitions; confirm one window, one colorbar, aligned y-axis, and that both panels use the same scale (e.g. one panel mostly “dim” if its values are lower but same colormap range).
