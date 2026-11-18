---
name: Harden Vispy Data Guards
overview: Apply a focused hardening pass to trajectory/central rendering so only finite numeric data reaches Vispy line/arrow/image draws. Standardize guard logic and add small defensive checks for degenerate normalization and timeline values.
todos:
  - id: harden-trajectory-guards
    content: Update predicitive_decoding_vispy.py finite checks for trajectory, extensions, timeline ticks, and background lines
    status: completed
  - id: harden-central-guards
    content: Update predictive_decoding_central_view.py finite checks for centroid and current-position rendering paths
    status: completed
  - id: verify-no-regression
    content: Run diagnostics and perform quick epoch-sweep validation to confirm rendering stability
    status: completed
isProject: false
---

# Harden Vispy Guard Checks

## Scope

- Update guard logic in [H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py).
- Update parallel guard logic in [H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predictive_decoding_central_view.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predictive_decoding_central_view.py).

## Planned Changes

- Replace remaining `np.isnan(...)`-only filters with `np.isfinite(...)` filters for coordinates/timestamps/angles where values are later used in `vz.Line`, `vz.Arrow`, or timeline plotting.
- In trajectory-side rendering, enforce finite `t` at the same time as `x/y` when time is used for:
  - `mean_time` (timeline ticks)
  - `traj_t_min` / `traj_t_max` (extension window)
  - opacity computation
- Harden `_extend_trajectory_xy_opacity(...)` to only concatenate finite extension arrays (`x/y/t`) so extensions cannot inject infinities.
- Guard degenerate color normalization path:
  - before `t_rel_valid / np.ptp(t_rel_valid)`, branch when `np.ptp(t_rel_valid) <= 0` and fall back to zeros (or equivalent stable default) to avoid NaN vertex colors.
- Guard timeline tick creation so ticks are emitted only for finite `mean_time`.
- In background/full-position trace path, require finite `x/y` and at least 2 valid points before constructing `vz.Line`.
- Mirror equivalent finite checks in central renderer for:
  - centroid positions
  - centroid angles
  - current position trajectory (`x/y/t`)

## Verification

- Run lint/diagnostic check for edited files and fix only newly introduced issues.
- Quick functional validation path:
  - instantiate viewer
  - sweep several epochs (including previously failing epoch)
  - confirm no GL access-violation during line draw
- If crash persists, add temporary debug logging around per-trajectory skip reasons and first invalid value source (without broad refactor).

## Constraints

- Keep edits minimal and localized to guard conditions and small defensive branches.
- Preserve existing behavior for valid numeric inputs and existing UI interactions.

