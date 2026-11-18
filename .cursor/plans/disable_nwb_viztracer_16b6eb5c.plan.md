---
name: Disable NWB viztracer
overview: Add `should_use_viztracer_logging=False` to the NWB W-maze batch driver so generated run/figure scripts omit VizTracer wrapping (same pattern as `ProcessBatchOutputs_qclus1246789_Only.ipy` and the existing Neptune disable in this notebook).
todos:
  - id: add-viztracer-false
    content: Add `should_use_viztracer_logging=False` to `active_phase_dict.update()` in ProcessBatchOutputs_NWB_WMaze_Batch.ipy (~line 468)
    status: completed
isProject: false
---

# Disable viztracer for NWB_WMaze batch scripts

## Context

VizTracer is wired through batch script generation, not the notebook itself:

- [`generate_batch_single_session_scripts`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py) defaults `should_use_viztracer_logging=True` (line 358).
- When `True`, [`python_template.py.j2`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Resources/Templates/python_template.py.j2) imports `VizTracer` and wraps the main computation in a trace context manager, writing large `viztracer_*.json` files per session/script.

[`ProcessBatchOutputs_NWB_WMaze_Batch.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_NWB_WMaze_Batch.ipy) already disables Neptune in its phase config but does **not** pass `should_use_viztracer_logging`, so NWB W-maze scripts currently get VizTracer enabled by default.

Existing precedent: [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy) passes `should_use_viztracer_logging=False` to `generate_batch_single_session_scripts` (lines 403–404).

```mermaid
flowchart LR
  ipy["ProcessBatchOutputs_NWB_WMaze_Batch.ipy"]
  gen["generate_batch_single_session_scripts"]
  tmpl["python_template.py.j2"]
  scripts["run_*.py / figures_*.py"]
  ipy -->|"active_phase_dict kwargs"| gen
  gen -->|"should_use_viztracer_logging=False"| tmpl
  tmpl --> scripts
```

## Change (single file, one line)

In [`ProcessBatchOutputs_NWB_WMaze_Batch.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_NWB_WMaze_Batch.ipy), inside `process_all_phases`, add `should_use_viztracer_logging=False` to the existing `active_phase_dict.update({...})` block (~lines 463–475), next to the already-present Neptune disable:

```python
active_phase_dict.update({
    ...
    'should_use_neptune_logging': False,
    'should_use_viztracer_logging': False,  # NEW
    'create_slurm_scripts': True,
    ...
})
```

Because `generate_batch_single_session_scripts(..., **active_phase_dict)` is called at line 486, this applies to **all phases** (`continued_run`, `final_run`, `figure_run`) for every `dandi_nwb` session in this driver.

## Scope / non-goals

- **No library change**: keep `generate_batch_single_session_scripts` default as `True` so other batch drivers (Bapun, KDIBA, etc.) are unaffected.
- **No new function parameter** unless you want runtime toggling; a hardcoded default in this NWB-specific driver matches how Neptune is already handled.
- **Regeneration required**: re-run the notebook (or `process_all_phases`) to regenerate scripts; already-generated `run_*.py` files on disk will still contain VizTracer until replaced.

## Verification

After re-running script generation for one phase:

1. Open a generated `run_*.py` under the batch output dir — it should **not** contain `from viztracer import VizTracer` or a `with VizTracer(...)` block.
2. Confirm stdout no longer prints `VizTracer output will be saved to: ...` at script start.
