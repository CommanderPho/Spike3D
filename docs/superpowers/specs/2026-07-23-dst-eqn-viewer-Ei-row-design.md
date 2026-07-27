# DST eqn viewer: per-cell `E_i(x)` row

## Goal

When `build_interactive_bayesian_2d_eqn_viewer` receives a `BayesianPlacemapPositionDecoderDST`, insert one mosaic row under each cell’s placefield showing that cell’s Shafer-discounted evidential mass map `E_i(x)`. Bayesian (non-DST) layout stays unchanged.

## Formula

Match `BayesianPlacemapPositionDecoderDST.compute_posterior`:

- Per-cell Poisson likelihood: `L_i(x) = (τ f_i(x))^{n_i} · exp(-τ f_i(x))` (factorial is constant in `x` and cancels under normalization)
- `p_i(x) = L_i(x) / Σ_x L_i(x)`
- `α = reliability_active[i]` if `n_i > 0`, else `reliability_silent[i]`
- `E_i(x) = α · p_i(x) + (1 − α)`

Recompute on every slider redraw.

## Layout (DST only)

```
row: cell_*_pf
row: cell_*_E          ← new
row: cell_*_exp_term
row: decoded_posterior | term0 | term1 | joint_likelihood
```

Non-DST: existing three-row mosaic (no `*_E` keys).

Bump `height_ratios` / `fig_h` slightly when the extra row is present.

## Drawing

- Axes: `ax_cell_E` list parallel to `ax_cell_pf`
- `imshow` via existing `_subfn_imshow_map`, same cell colormap
- Title e.g. `Cell {aclu}: E_i=α p_i+(1-α)  α={α:.2f}` (note active vs silent via current `n_i`)
- Clear `ax_cell_E` in `_subfn_redraw` with other axes
- Keep refs on `fig._bayes_eqn_ui['ax_cell_E']`

## Scope

- File: `PendingNotebookCode.py` → `build_interactive_bayesian_2d_eqn_viewer` only
- No changes to `reconstruction_dst.py` decode math
- Docstring: mention DST `E_i` row

## Out of scope

- In-field mask panels
- Scalar-only α gauges
- Changing Bayesian factor panels
