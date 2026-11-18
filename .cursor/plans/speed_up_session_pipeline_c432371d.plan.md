---
name: Speed Up Session Pipeline
overview: Reduce end-to-end runtime of `run_session_pipeline` using low-risk orchestration improvements plus bounded parallel execution, while preserving current outputs and failure semantics.
todos:
  - id: add-timing
    content: Add per-step timing instrumentation and summary output in session_pipeline.py
    status: completed
  - id: parallelize-independent-steps
    content: Execute independent post-reformat heavy steps with bounded parallel workers
    status: completed
  - id: batch-momentum-gridsearch
    content: Implement one-load momentum gridsearch batching in scripts/o2/o2_lib.py and wire into session_pipeline.py
    status: completed
  - id: verify-regression-and-speed
    content: Run before/after timing and confirm artifact/output compatibility
    status: completed
isProject: false
---

# Speed Up `run_session_pipeline`

## Goals
- Cut wall-clock time for a single-session run without changing output file formats or step dependencies.
- Keep failure handling (`strict`, `critical`, dependency skips) behavior-compatible.

## Target Files
- [h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\pipelines\session_pipeline.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\pipelines\session_pipeline.py)
- [h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\scripts\o2\o2_lib.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\scripts\o2\o2_lib.py)

## Why It Is Slow Today
- `run_session_pipeline` executes several heavy phases serially even when they only depend on `reformat_data_for_structure_analysis_ripples`.
- Momentum gridsearch loops over all SWRs and currently re-loads structure data for each `spikemat_ind` via `submit_momentum_gridsearch(...)`, creating significant repeated I/O and deserialization overhead.

## Implementation Plan
1. Add lightweight step timing instrumentation in `session_pipeline.py`:
   - Record start/end duration for each `_run_step` execution.
   - Include elapsed seconds in `StepResult.detail` and summary prints.
   - Keep logs concise and compatible with current CLI output.

2. Introduce bounded parallel execution for independent heavy steps after reformat:
   - Parallelize `run_model_stationary`, `run_model_random`, `gridsearch_ripples`, and `get_trajectories_ripples` once reformat succeeds.
   - Keep `run_model_comparison_ripples` waiting for stationary/random/gridsearch completion.
   - Keep optional phases (`marginals`, `diffusion_constant`) dependency-safe.
   - Use a conservative worker cap to avoid overloading memory/CPU.

3. Remove repeated structure-data loads inside momentum gridsearch path:
   - Add a batch helper in `o2_lib.py` that loads structure data once and processes all `spikemat_ind` values in one pass, saving each result exactly as before.
   - Update `_run_gridsearch_phase(...)` to use this batch path.
   - Preserve `continue_on_gridsearch_error` behavior and partial-success semantics.

4. Keep artifact-cache checks but minimize extra filesystem work:
   - Consolidate repeated existence checks where practical.
   - Avoid duplicate path construction for the same step.

5. Validate behavior and speed:
   - Run a controlled before/after timing on one representative session.
   - Verify generated artifact filenames/locations are unchanged.
   - Confirm failure/skip behavior still matches existing dependency logic.

## Acceptance Criteria
- Pipeline outputs are identical for the same inputs (same artifact paths and successful step set).
- Wall-clock runtime is measurably reduced on representative sessions.
- `strict`, `continue_on_gridsearch_error`, and skip/critical behavior remain intact.