---
name: Posterior default single bin
overview: Restore viewport-driven posterior display to a single decoded bin at the viewport left edge (matching legacy `update(start_t)` / `searchsorted` behavior) by default, and keep viewport summed/renormalized posteriors only behind `use_all_active_viewport_timebins=True`.
todos:
  - id: init-flag
    content: Add use_all_active_viewport_timebins to __init__ (param_kwargs.pop, default False)
    status: completed
  - id: on-window-branch
    content: "Branch on_window_changed: left-edge single index vs _included_posterior_bin_indices_for_viewport"
    status: completed
  - id: title-gate
    content: Gate | n_bins title suffix on use_all_active_viewport_timebins
    status: completed
  - id: doc-helper
    content: Brief doc on _included_posterior_bin_indices_for_viewport optional mode
    status: completed
isProject: false
---

# Default single-bin posteriors; optional viewport aggregation

## Behavior


| Mode                                                   | `on_window_changed(start_t, end_t)` posterior indices                                                                                                                                                                                                                                                                                                  | Title `n_bins`                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Default** (`use_all_active_viewport_timebins=False`) | Single index: `clamp(searchsorted(centers, start_t, side='left'), 0, n-1)` — **ignores `end_t` for the heatmap**, same bin family as `[update](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)` for `t=start_t` | Omit `n_bins` (or only when flag is True — see below) |
| **Optional** (`use_all_active_viewport_timebins=True`) | Existing `[_included_posterior_bin_indices_for_viewport](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)`: strict bins in `[start_t,end_t]`, fallback to single clamped index                                   | Append `                                              |


## Implementation (one file)

**File:** [TimeSynchronizedPositionDecoderPlotter.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)

1. `**__init`__** — After other `param_kwargs.pop` calls, add:
  - `self.use_all_active_viewport_timebins = param_kwargs.pop('use_all_active_viewport_timebins', False)` (bool, default `False`).
2. `**on_window_changed`** — Branch on `self.use_all_active_viewport_timebins`:
  - **False:** Compute `idx` with the same logic as `update` but use `start_t` instead of `t` (`searchsorted` + clamp). Set `self.last_included_posterior_bin_indices = np.array([idx], dtype=np.intp)`, `last_window_index` / `last_window_time` from `idx` / `centers[idx]`.
  - **True:** Keep the current block calling `_included_posterior_bin_indices_for_viewport(...)` and the existing `fi = first index` assignment for `last_window`_*.
3. `**_update_plots` title** (~585–586) — Only append `| n_bins: {len(...)}` when `self.use_all_active_viewport_timebins` is True **and** `last_included_posterior_bin_indices is not None`. That removes the new suffix from default UI and keeps it as a signal for aggregation mode.
4. **Docs** — One-line note on `_included_posterior_bin_indices_for_viewport` that it is used when `use_all_active_viewport_timebins` is enabled.

## No changes required

- **[TimeSynchronizedGenericPlotterLayer.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedGenericPlotterLayer.py):** `_update_plots` already uses single-slice when `len(included)==1` and sum+normalize when `>1`. Default mode only passes length-1 indices.
- `**update(t)`:** Already sets `last_included_posterior_bin_indices` to one index; unchanged.

## Usage

Construct with `use_all_active_viewport_timebins=True` where summed posteriors are desired (e.g. notebooks in [PendingNotebookCode.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) or [helpers.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/helpers.py)). Existing call sites need no updates (default restores legacy behavior).

## Verification

- Default: widen/narrow the synced window — posterior and `t_idx` track the **left edge** only; title has no `n_bins`.
- Flag True: widening increases `n_bins` when more strict bins fit; heatmap is summed/renormalized as now.

