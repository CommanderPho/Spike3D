---
name: Plot 2D Transitions
overview: Update `plot_transition_matricies` so it accepts the single-decoder list returned by `_compute_position_transition_matrix_2d` as well as the existing decoder-name dictionary used by 1D transition matrices.
todos:
  - id: normalize-plot-input
    content: Add a small input-normalization block in `plot_transition_matricies` before calling `.items()`.
    status: completed
  - id: document-list-usage
    content: Update the plotter docstring to mention both dict and single list inputs.
    status: completed
  - id: validate-file
    content: Run a focused validation/lint check for the edited file.
    status: completed
isProject: false
---

# Plot 2D Transitions

I’ll make a minimal change in [`pyphoplacecellanalysis/Analysis/Decoder/transition_matrix.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/transition_matrix.py):

- Keep `_compute_position_transition_matrix_2d(...)` returning `List[NDArray]`, since that mirrors the 1D low-level helper and matches your current notebook code.
- In `TransitionMatrixComputations.plot_transition_matricies(...)`, normalize `binned_x_transition_matrix_higher_order_list_dict` before the `.items()` loop:
  - If it is already a dict, preserve the current 1D behavior unchanged.
  - If it is a list/tuple of matrices, wrap it into a single-entry dict using a decoder inferred from `decoders_dict`.
  - Prefer a decoder whose `(len(xbins) * len(ybins))` matches the flattened 2D transition matrix size; otherwise fall back to the sole decoder if there is only one.
- Preserve the existing 2D rendering logic, which already converts each flattened transition matrix into a destination probability map via `np.nansum(transition_matrix, axis=0).reshape(n_x_bins, n_y_bins)`.
- Add/update the docstring usage so your exact pattern is valid:

```python
binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix_2d(a_decoder.xbin_labels, a_decoder.ybin_labels, *[(pos_df[k].to_numpy() - 1) for k in binned_pos_column_labels])
out = TransitionMatrixComputations.plot_transition_matricies(decoders_dict=directional_decoders_decode_result.pf1D_Decoder_dict, binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list, power_step=3)
```

After editing, I’ll run a focused syntax/import check if feasible and check lints for `transition_matrix.py`.