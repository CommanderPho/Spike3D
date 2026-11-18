---
name: Momentum-Limited Column Propagation
overview: Update `compute_momentum_limited_change_df` so constrained angular changes are propagated into the full smoothed trajectory and derived kinematic columns, overwriting existing columns as requested.
todos:
  - id: inspect-existing-method
    content: Confirm the exact in-function data assumptions (index alignment, dt handling, first-row anchoring).
    status: completed
  - id: add-trajectory-propagation
    content: Implement post-limiter heading and x/y reconstruction using preserved original step lengths.
    status: completed
  - id: recompute-kinematic-columns
    content: Overwrite velocity_x_smooth, velocity_y_smooth, speed_xy, and approx_head_dir_degrees from reconstructed trajectory.
    status: completed
  - id: consistency-checks
    content: Add/retain internal checks so limiter diagnostics and recomputed columns are numerically consistent.
    status: completed
isProject: false
---

# Momentum-Limited Dependent Column Recompute

## Goal
Update [`h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) in `MomentumHelpers.compute_momentum_limited_change_df` so the turn-limited sequence (`dtheta_applied`) is used to recompute and overwrite:
- `x_smooth`, `y_smooth`
- `velocity_x_smooth`, `velocity_y_smooth`
- `speed_xy`
- `approx_head_dir_degrees`

while preserving original per-frame step magnitude from `diff(x_smooth, y_smooth)`.

## Planned Changes
- Keep the existing queue-based limiter logic intact (`request -> clip -> queue`) to preserve total turn input over time.
- Add a trajectory propagation stage after `dtheta_applied` is computed:
  - derive per-frame step magnitude from original `x_smooth/y_smooth` diffs,
  - reconstruct constrained heading from cumulative `dtheta_applied` with a stable initial heading,
  - synthesize constrained `dx/dy` from `step_mag * [cos(theta), sin(theta)]`,
  - re-integrate `x_smooth/y_smooth` in index order using the original first-row position as anchor.
- Recompute dependent kinematic columns from the updated trajectory:
  - `velocity_x_smooth`, `velocity_y_smooth` from finite differences divided by `dt` (or `dt_col`),
  - `speed_xy = hypot(velocity_x_smooth, velocity_y_smooth)`,
  - `approx_head_dir_degrees` from `atan2(velocity_y_smooth, velocity_x_smooth)` normalized to `[0, 360)`.
- Preserve existing diagnostic outputs (`requested_dtheta_with_queue`, `dtheta_applied`, `dTheta_dt_applied`, `turn_queue`, `respects_turn_limit`) and ensure they remain consistent with the recomputed trajectory.

## Validation
- Verify angular constraint still holds (`abs(dTheta_dt_applied) <= abs(dTheta_dt_max) + eps`).
- Verify per-frame step length is preserved (new `hypot(diff(x_smooth), diff(y_smooth))` matches original within tolerance).
- Sanity-check first/last rows for integration drift and NaN handling around `dt` edges.