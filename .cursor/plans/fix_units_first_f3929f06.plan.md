---
name: Fix Units First
overview: Make `compute_momentum_limited_change_df` unit-consistent in degrees so clipping and turn-limit checks are physically correct, while minimizing scope to the unit bug first.
todos:
  - id: inspect-current-unit-path
    content: Trace existing angular unit flow and mark deg/rad boundaries in the method.
    status: completed
  - id: normalize-limiter-to-degrees
    content: Convert raw turns to degrees before queue/clip and compute max_step in deg/frame.
    status: completed
  - id: write-unit-consistent-outputs
    content: Emit dtheta_applied in radians for integration, dTheta_dt_applied in deg/s for comparison.
    status: completed
  - id: validate-limit-check-path
    content: Verify respects_turn_limit compares like units and remains numerically stable with dt guards.
    status: completed
isProject: false
---

# Degrees-Canonical Unit Fix

## Goal
Patch [`h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) so `MomentumHelpers.compute_momentum_limited_change_df` uses degrees consistently for limiter math and limit validation.

## Why This First
The function currently clips `dtheta` (radians/frame) with a degree-based max-rate (`dTheta_dt_max`) and then compares a rad/s-derived `dTheta_dt_applied` against deg/s, producing misleadingly weak constraints.

## Planned Changes
- In `compute_momentum_limited_change_df`, normalize the angular pipeline to degrees:
  - Convert raw turn increments to `raw_deg` (deg/frame) before queue/clip logic.
  - Keep `max_rate` as deg/s and compute `max_step_deg = max_rate * dt_arr` (deg/frame).
  - Run queue limiter on degree values (`requested`, `applied`, `queue` all in deg/frame).
- Preserve public output compatibility while clarifying units:
  - Keep `dtheta_applied` as radians/frame for downstream heading integration (derive from `applied_deg`).
  - Set `dTheta_dt_applied` explicitly in deg/s from `applied_deg / dt_arr`.
  - Ensure `respects_turn_limit` compares deg/s to deg/s only.
- Add minimal internal unit-clarity fields/comments:
  - transient vars named with `_deg` / `_rad` suffixes in function scope,
  - optional debug column(s) only if already part of current diagnostics pattern.
- Keep scope limited to units (no behavioral refactor beyond unit-correct clipping/checking).

## Validation
- Confirm `abs(dTheta_dt_applied) <= abs(dTheta_dt_max) + eps` is true when clipping saturates.
- Confirm queue dynamics still conserve requested turn (delayed, not lost).
- Spot-check with a few rows that `dtheta_applied` (rad/frame) equals `deg2rad(applied_deg)` and trajectory code still consumes radians as expected.