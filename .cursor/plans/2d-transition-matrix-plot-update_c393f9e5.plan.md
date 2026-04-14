---
name: 2d-transition-matrix-plot-update
overview: Update transition-matrix plotting so 2D decoder transition matrices are visualized correctly by converting flattened state-to-state matrices into 2D destination probability maps.
todos:
  - id: inspect-plot-branch
    content: Add 1D/2D detection branch in `plot_transition_matricies` while preserving existing structure.
    status: completed
  - id: add-2d-transform
    content: Implement flattened transition-to-destination-map transform and choose correct x/y bins for 2D decoders.
    status: completed
  - id: verify-compatibility
    content: Confirm 1D plotting still works unchanged and 2D plotting passes shape constraints.
    status: completed
isProject: false
---

# Update `plot_transition_matricies` for 2D Transition Matrices

## Goal
Make [`h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/transition_matrix.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/transition_matrix.py) handle both 1D and 2D decoder transition matrices in the existing plotting flow.

## Planned changes
- In `TransitionMatrixComputations.plot_transition_matricies`, detect whether a decoder is 2D by checking for `ybin_centers` / `ybin_labels` and matrix shape consistency with `n_x * n_y`.
- For 2D decoders, convert each flattened transition matrix `T` of shape `(n_x*n_y, n_x*n_y)` into a destination-map image by summing over source states:
  - `dest_flat = np.nansum(T, axis=0)`
  - `dest_map = dest_flat.reshape(n_x, n_y)`
- Plot 2D `dest_map` with `xbins` from x centers and `ybins` from y centers/labels so axis ticks are physically meaningful.
- Keep the current 1D behavior unchanged (existing square matrix plot with x-vs-x bins).
- Preserve existing title/grid/row/column behavior and only branch matrix/axis selection logic where needed (minimal edit footprint).

## Validation plan
- Run a quick smoke test with a known 1D decoder and confirm output looks unchanged.
- Run a smoke test with a 2D decoder matrix list from `_compute_position_transition_matrix_2d` and verify each row/decoder cell renders without shape assertion errors.
- Confirm plotted 2D map dimensions align with `(len(xbins), len(ybins))` in `BasicBinnedImageRenderingWindow` expectations.

## Notes
- This implements your selected 2D behavior: **sum over source states to show destination probability map**.
- No notebook changes are needed.