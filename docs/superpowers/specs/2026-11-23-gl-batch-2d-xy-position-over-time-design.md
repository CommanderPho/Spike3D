# Design: GL batch 2D (x, y) position over time (layout C)

**Date:** 2026-11-23  
**Status:** approved  
**Approach:** (1) new headless Agg completion + thin plot helper

## Goal

Export one figure per session showing the animal’s measured trajectory in the maze plane (`x` vs `y`) with **time encoded as color**, runnable on Great Lakes via the existing KDIBA batch / `--only` recipe path.

## Non-goals

- Stacked `x(t)` / `y(t)` timeseries (layout A/B)
- Per-lap paginated grids (`plot_lap_trajectories_2d`)
- PyQt / spike-raster `_render_export_all_time_tracks`
- Decoded posteriors, lap overlays, velocity panels
- Rewriting pipeline pickles

## Visual spec

- Single matplotlib axes, equal aspect
- Trajectory as a time-gradient line (prefer `DecodedTrajectoryMatplotlibPlotter._helper_add_gradient_line` with `t`, `x`, `y`)
- Colorbar labeled with time (`t`)
- Title / context identifying the session
- Optional max-point downsample (default on if `n_points` exceeds a high threshold, e.g. 50k) to keep Agg export fast on long sessions — preserve time order via stride, not random sample

## Data source

- Prefer global filtered session position: `curr_active_pipeline.filtered_sessions[global_epoch_name].position`
- Fallback: `curr_active_pipeline.sess.position` if global filter name unavailable
- Required columns / arrays: `t`, `x`, `y` (from position object / dataframe)

## Code layout

### 1. Pure plot helper (pyPhoPlaceCellAnalysis)

**Suggested location:** `PhoPositionalData/plotting/laps.py` (or a small sibling module if `laps.py` is already oversized)

**Suggested API:**

```python
def plot_2d_xy_position_over_time(position_or_df, *, ax=None, figsize=(10, 8), cmap='viridis', max_points: Optional[int] = 50000, add_colorbar: bool = True) -> Tuple[Figure, Axes, dict]:
    ...
```

Returns `(fig, ax, artists_dict)` with LineCollection / colorbar handles.

### 2. Batch completion (pyPhoPlaceCellAnalysis)

**File:** `General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py`

**Name:** `figures_export_2d_position_over_time_completion_function`

**Behavior (mirror `figures_export_laps_correct_decoder_side_by_side_completion_function`):**

- Force `MPLBACKEND=Agg`
- Build figure via the plot helper
- Save PNG to `self.collected_outputs_path`:
  - `{BATCH_DATE_TO_USE}-{session_name}_2d_xy_position_over_time.png`
- Record paths / errors in `across_session_results_extended_dict[...]`
- Catch exceptions; re-raise only if `fail_on_exception_for_debugging` / `self.fail_on_exception`
- Register in `MAIN_get_template_string` default catalog

### 3. GL `--only` recipe (Spike3D)

**File:** `ProcessBatchOutputs_qclus1246789_Only.ipy`

- Import the new completion function
- Add recipe `figures_export_2d_position_over_time` (+ `_completion_function` alias)
- `phases`: `[ProcessingScriptPhases.final_run]`
- `extended_computations_include_includelist`: `[]` (position already on loaded pipeline; no decoder prereqs)
- `should_freeze_pipeline_updates`: `True` or leave false with empty compute list — prefer freeze / empty list so jobs stay light
- Document CLI:

```text
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --date YYYY-MM-DD --computer GL --only figures_export_2d_position_over_time
```

## Success criteria

1. One PNG per session under `collected_outputs` with `(x,y)` plane and time colorbar
2. Runs headless on GL under `--only figures_export_2d_position_over_time`
3. Does not require DirectionalLaps / decoder computations
4. Failures for one session do not abort the batch unless fail-on-exception is set

## Out of scope for v1

- Vector PDF export (PNG only unless trivially free)
- Adding the function to the default `figure_run` mega-set (opt-in via `--only` only)
- Pipeline `@display` registration (optional follow-up)

## Open decisions (defaults if unspecified)

| Decision | Default |
|----------|---------|
| Downsample | stride to ≤ 50k points |
| Colormap | `viridis` |
| Also PDF | no |
| Default `figure_run` inclusion | no (recipe-only) |
