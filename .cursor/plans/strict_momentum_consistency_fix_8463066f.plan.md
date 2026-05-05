---
name: Strict Momentum Consistency Fix
overview: Repair remaining momentum-limiter inconsistencies by enforcing per-second angular semantics end-to-end, fixing trajectory reconstruction alignment, and correcting constrained/original overlay plotting behavior.
todos:
  - id: normalize-per-second-semantics
    content: Enforce per-second angular-rate consistency from dTheta_dt source through limiter outputs.
    status: completed
  - id: fix-heading-index-alignment
    content: Correct dtheta application alignment in trajectory propagation and preserve queue conservation.
    status: completed
  - id: repair-overlay-plot-call
    content: Fix plot helper call/update wiring so constrained overlay styling and legend are correct.
    status: completed
  - id: add-and-run-consistency-checks
    content: Apply strict invariants and validate unit-matched limits, conservation, and edge-case dt behavior.
    status: completed
isProject: false
---

# Strict Momentum Consistency Fix

## Target File
- [`h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py)

## Goal
Make the momentum-limited pipeline logically consistent under a **per-second** convention, then fix trajectory/plotting bugs that currently distort behavior and interpretation.

## Planned Changes
- Normalize angular semantics to per-second where required for consistency:
  - ensure upstream turn-rate source (`dTheta_dt`) used for fitting/limits is computed as true deg/s,
  - ensure `dTheta_dt_max` and `dTheta_dt_applied` are both deg/s,
  - keep integration deltas (`dtheta`, `dtheta_applied`) as radians per frame derived via `dt`.
- Fix limiter/integration alignment in `compute_momentum_limited_change_df`:
  - correct index alignment between per-step turn deltas and heading propagation (`i-1 -> i` transition consistency),
  - keep queue conservation intact while preventing off-by-one heading drift,
  - keep `dt` guards consistent for near-zero/invalid intervals.
- Fix constrained overlay plotting correctness in `plot_momentum_pyqtgraph_line_plot`:
  - correct the malformed `_subfn_perform_plot` call/update pattern so kwargs are applied to the right function,
  - ensure constrained curves are actually plotted with intended style/legend separation from baseline.
- Add strict diagnostics/invariants (minimal but explicit):
  - invariant checks for unit-consistent comparisons,
  - optional helper columns/logical checks to verify queue conservation and limit compliance.

## Validation
- Numeric consistency checks:
  - `abs(dTheta_dt_applied) <= abs(dTheta_dt_max) + eps` using matching deg/s units,
  - turn conservation over window: raw request equals applied plus residual queue (in consistent units).
- Behavioral sanity checks:
  - constrained trajectory updates lag/smooth relative to baseline without pathological divergence,
  - no new plotting artifacts; constrained overlay is visually and legend-wise unambiguous.
- Spot-check boundaries:
  - first/last row heading integration behavior,
  - `dt` edge cases do not produce spurious infinities or invalid turn-rate values.