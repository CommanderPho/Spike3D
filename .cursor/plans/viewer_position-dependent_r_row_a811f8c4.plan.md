---
name: Viewer position-dependent R row
overview: Add a POSITION_DEPENDENT reliability estimation toggle to `InteractiveBayesian2DEquationDebugger`, and when enabled show a per-cell `R_i(x)` heatmap row directly under each placefield.
todos:
  - id: estimation-toggle
    content: Add ReliabilityEstimationMode field, import, checkbox, setup sync + confusion ensure on toggle
    status: completed
  - id: r-row-redraw
    content: Add mosaic R_row under PF, ax_cell_R, redraw populate; fix 2D α in titles/_dst_Ei_maps; export/docs
    status: completed
isProject: false
---

# Viewer: position-dependent reliability row

## Scope

Only [`PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) `InteractiveBayesian2DEquationDebugger` (~1518–2499). Decoder estimation already exists; this wires it into the UI.

## UI control

Add a checkbox under the existing reliability radio (same left column as `drop n=0`):

- Label: `pos-dep R`
- Checked ↔ `ReliabilityEstimationMode.POSITION_DEPENDENT`
- Unchecked ↔ `PER_CELL` (default)

On check:

1. Set `self.reliability_estimation_mode` and `self.sliced.reliability_estimation_mode`
2. If confusion products missing, call `self.sliced.compute_unit_confusion_reliability_variables()` (auto-refreshes `reliability_*`); else `self.sliced._compute_reliability_metrics()`
3. `redraw()`

On uncheck: set `PER_CELL`, recompute metrics if confusion df present (else ones), `redraw()`.

## Mosaic row under PF

Always insert an `R` row immediately below `pf_row` (before DST `E_row` / `L_row`) so toggling does not rebuild the figure:

```text
pf_row
R_row          # cell_a_R, cell_b_R, ...
[E_row]        # DST only
L_row
factor_row
```

- Labels: `cell_{a..}_R`
- Field: `ax_cell_R: List`
- Height ratio ~2.4; bump `fig_h` slightly (like DST E bump)
- Include each `ax_cell_R[i]` in the per-cell group box (right after PF)
- In `redraw`: always `cla` these axes; if mode is `POSITION_DEPENDENT` and `reliability_active` is 2D, imshow cell `i` map reshaped to `(nx, ny)` with `vmin=0, vmax=1`, title `R[{aclu}](x)` (use active rates; silent map only if `n_i==0` and `should_discount_silence`); else leave axes empty with title `R (PER_CELL)` or clear title

Helper on the class:

```python
@classmethod
def _reliability_map_for_cell(cls, reliability_arr, cell_idx, spatial_shape) -> NDArray:
    R = np.asarray(reliability_arr, dtype=float)
    if R.ndim == 1:
        return np.full(spatial_shape, float(R[cell_idx]))
    return R[:, cell_idx].reshape(spatial_shape)
```

## Fix 2D reliability consumers in the same class

Current code assumes 1D `reliability_active[i]` as float — breaks for POSITION_DEPENDENT:

- PF title α display in `redraw` (~2235): use nanmean of the cell’s map (or omit scalar α when 2D and point at the R row)
- [`_dst_Ei_maps`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) (~1828): support 2D α like `_poisson_factor_maps` already does (`alpha_i` as spatial array)
- Fix `discount_silence` → `should_discount_silence` in the PF title branch

## Wiring checklist

- Import `ReliabilityEstimationMode`
- Fields: `reliability_estimation_mode`, `ax_cell_R`, `estimation_mode_check`
- `setup`: sync mode from field onto `sliced`; if already `POSITION_DEPENDENT`, ensure confusion/metrics
- `buildUI`: mosaic + checkbox + group boxes + export refs
- `on_estimation_mode` callback
- `_get_export_control_axes` / `_build_export_context` include estimation mode
- Class docstring layout note for the R row

## Out of scope

- No Zhang `neuropy_bayesian_prob` changes
- No new per-bin confusion aggregation
- No rebuild of figure on toggle (row always allocated)
