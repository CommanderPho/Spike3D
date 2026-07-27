---
name: DST Ei mosaic row
overview: Add a DST-only mosaic row under each placefield in `build_interactive_bayesian_2d_eqn_viewer` showing per-cell Shafer mass `E_i(x) = α p_i(x) + (1-α)`, recomputed on slider updates.
todos:
  - id: helper-Ei
    content: Add _subfn_dst_Ei_maps helper matching DST compute_posterior mass formula
    status: completed
  - id: mosaic-dst
    content: Conditionally insert cell_*_E mosaic row + height/fig sizing when is_dst
    status: completed
  - id: redraw-ui
    content: Wire ax_cell_E into _subfn_redraw, docstring, and fig._bayes_eqn_ui
    status: completed
isProject: false
---

# DST eqn viewer `E_i` row

> Spec: [docs/superpowers/specs/2026-07-23-dst-eqn-viewer-Ei-row-design.md](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/docs/superpowers/specs/2026-07-23-dst-eqn-viewer-Ei-row-design.md)

**Goal:** When the decoder is `BayesianPlacemapPositionDecoderDST`, show one extra subplot row under each PF with that cell’s discounted evidential mass map; Bayesian layout unchanged.

**File:** only [`PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) → `build_interactive_bayesian_2d_eqn_viewer` (~1516–1866).

## Changes

### 1. Helper: per-cell `E_i` maps

Add `_subfn_dst_Ei_maps(tuning_curves_xy, spike_counts, tau, reliability_active, reliability_silent)` near `_subfn_poisson_factor_maps`:

- Reuse clipped `F` / per-cell `L_i` (power×exp; factorial cancels under normalize)
- `p_i = L_i / sum(L_i)`; if sum is 0, uniform over bins
- `α = reliability_active[i]` if `n_i > 0` else `reliability_silent[i]`
- `E_i = α * p_i + (1 - α)`
- Return list of `(nx, ny)` maps and optionally the α used per cell

Match [`compute_posterior`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction_dst.py) lines ~444–460.

### 2. Conditional mosaic

After `is_dst` is known, build layout:

- **DST:** `pf` → `cell_*_E` → `exp_term` → joint factors; `height_ratios=[3.0, 2.8, 3.0, 3.2]`; bump `fig_h` by ~0.9
- **Bayesian:** keep current 3-row mosaic / ratios

```python
pf_row = [f"cell_{chr(97+i)}_pf" for i in range(n_cells)] + ["."] * pad
E_row = [f"cell_{chr(97+i)}_E" for i in range(n_cells)] + ["."] * pad  # DST only
L_row = [f"cell_{chr(97+i)}_exp_term" for i in range(n_cells)] + ["."] * pad
factor_row = ["decoded_posterior", "term0", "term1", "joint_likelihood"]
```

### 3. Redraw + UI refs

- `ax_cell_E = [...]` when DST else `[]`
- In `_subfn_redraw`: clear `ax_cell_E`; if DST, compute `E_i` and `_subfn_imshow_map` with cell cmap; title like `Cell {aclu}: $E_i=\alpha p_i+(1-\alpha)$  $\alpha$={α:.2f}`
- `fig._bayes_eqn_ui['ax_cell_E'] = ax_cell_E`
- Docstring: note DST `E_i` row under PFs

## Verify

Call with a DST decoder (pair of neurons); confirm four mosaic rows, `E_i` updates when toggling `n=0` vs `n>0` (α active vs silent), and Bayesian path still has three rows.
