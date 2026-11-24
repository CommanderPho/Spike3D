# GL batch 2D (x,y) position-over-time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Headless Agg PNG of maze-plane `(x, y)` colored by time, exported via a GL `--only` batch completion.

**Architecture:** Thin pure plot helper reusing `DecodedTrajectoryMatplotlibPlotter._helper_add_gradient_line`; new `figures_export_2d_position_over_time_completion_function` mirrors the laps side-by-side figure completion; wire into `MAIN_get_template_string` and `ProcessBatchOutputs_qclus1246789_Only.ipy` `--only` recipes.

**Tech Stack:** matplotlib Agg, pyPhoPlaceCellAnalysis batch completion helpers, Spike3D ProcessBatchOutputs.

**Spec:** `docs/superpowers/specs/2026-11-23-gl-batch-2d-xy-position-over-time-design.md`

## Global Constraints

- Layout C only: spatial `(x, y)` with time as color (not x(t)/y(t) stacks)
- Headless Agg; no Qt/OpenGL
- Opt-in via `--only` (not default figure_run)
- No decoder computation prereqs; freeze pipeline updates when possible
- PNG only to `collected_outputs`; stride downsample ≤ 50k points
- Prefer single-line Python `def` signatures when ≤ 400 chars; two blank lines between class methods; `## END for ...` on for-loops

---

### Task 1: Plot helper

**Files:**
- Modify: `pyPhoPlaceCellAnalysis/.../PhoPositionalData/plotting/laps.py`

**Interfaces:**
- Produces: `plot_2d_xy_position_over_time(position_or_df, *, ax=None, figsize=(10, 8), cmap='viridis', max_points: Optional[int] = 50000, add_colorbar: bool = True, title: Optional[str] = None) -> Tuple[Figure, Axes, dict]`

- [ ] Add helper extracting `t,x,y`, optional stride downsample, gradient line + colorbar, equal aspect
- [ ] Smoke: call with synthetic arrays via a tiny dataframe

### Task 2: Batch completion + catalog

**Files:**
- Modify: `.../batch_user_completion_helpers.py` (near `figures_export_laps_correct_decoder_side_by_side_completion_function`; register in `MAIN_get_template_string`)

**Interfaces:**
- Consumes: `plot_2d_xy_position_over_time`
- Produces: `figures_export_2d_position_over_time_completion_function(...)`

- [ ] Agg export to `{BATCH_DATE}-{session}_2d_xy_position_over_time.png`
- [ ] Register in `MAIN_get_template_string` dict

### Task 3: GL `--only` recipe

**Files:**
- Modify: `Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy`

- [ ] Import, recipe + alias, CALL EXAMPLES + `--only` help text
- [ ] Empty compute list + `should_freeze_pipeline_updates=True`
