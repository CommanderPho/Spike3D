---
name: Bapun preflight batch helpers
overview: Wire `BapunBatchHelpers.run_all` immediately after pipeline load inside `run_specific_batch`, **before** the batch completion handler runs, scoped to **`run_*.py` scripts only** and **only when `IdentifyingContext.format_name == 'bapun'`** (per your choice). Script generation merges the needed flag into each rendered script’s `run_specific_batch_kwargs`.
todos:
  - id: runBatch-preflight
    content: Pop `preflight_bapun_batch_helpers_run_all` in `run_specific_batch`, call lazy-imported `BapunBatchHelpers.run_all` after `batch_load_session` and before `post_run_callback_fn`.
    status: completed
  - id: templating-kwargs
    content: "In `generate_batch_single_session_scripts`, merge `run_specific_batch_kwargs`: True for `run_*.py` + notebook temps when format is `bapun`; False for `figures_*.py`."
    status: completed
  - id: bapun-batch-notebook
    content: Optionally document behavior in `ProcessBatchOutputs_Bapun_Batch.ipy` comment only (only if user wants notebook touched).
    status: completed
isProject: false
---

# Bapun `run_all` before batch completion hooks

## Why not only `python_template.py.j2`

The snippet needs `curr_active_pipeline`, which is only created inside [`batch_load_session`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\NonInteractiveProcessing.py) (called from [`run_specific_batch`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\runBatch.py#L1098-L1192)).

```mermaid
flowchart LR
  genScript[Generated run_*.py]
  rsb[run_specific_batch]
  load[batch_load_session]
  pre[BapunBatchHelpers.run_all]
  cb[post_run_callback_fn completion handlers]
  genScript --> rsb --> load --> pre --> cb
```

So ordering is enforced in **`run_specific_batch`** (between load and callback), not as top-of-template code.

The **figures** scripts use the same Jinja [`python_template.py.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\python_template.py.j2) macro but with **`should_freeze_pipeline_updates=True`** / figure mode; skipping preflight there matches your preference to avoid duplicate QT/video work.

## Code changes

### 1. [`runBatch.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\runBatch.py)

- At the **start** of `run_specific_batch`, **pop**:

  `preflight_bapun_batch_helpers_run_all = kwargs.pop('preflight_bapun_batch_helpers_run_all', False)`  

  so the flag is never forwarded to [`batch_load_session(..., **kwargs)`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\runBatch.py#L1169-L1173).

- **After** successful `curr_active_pipeline = batch_load_session(...)` and **before** `post_run_callback_fn(...)`:

  ```python
  if preflight_bapun_batch_helpers_run_all:
      from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import BapunBatchHelpers
      _preflight_out_dict = BapunBatchHelpers.run_all(curr_active_pipeline=curr_active_pipeline)
  ```

  (Use `new_print` for a short breadcrumb mentioning preflight ran; optionally log/summarize `_preflight_out_dict` keys—not the full dict.)

- **No try/except** unless you prefer batch jobs to tolerate Qt/headless failures; with `fail_on_exception=True`, a failed `run_all` behaves like failures today (propagate).

### 2. [`pythonScriptTemplating.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py)

Inside `generate_batch_single_session_scripts`’ loop over [`included_session_contexts`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py#L424):

- Base dict: **`rs_kwargs = dict(renderer_script_generation_kwargs.get('run_specific_batch_kwargs') or {})`**.

- **`run_*.py` render**: merge `_run_kwargs = rs_kwargs | {'preflight_bapun_batch_helpers_run_all': (curr_session_context.format_name == 'bapun')}` into the variables passed to `python_template.render` as **`run_specific_batch_kwargs=_run_kwargs`** (build a shallow-copied merged render kwargs dict so you do not mutate the shared `renderer_script_generation_kwargs`).

- **`figures_*.py` render**: always merge **`preflight_bapun_batch_helpers_run_all=False`** (explicitly disable for Bapun figure scripts).

- **Notebook temp `.py`** path (still “compute”, not figures): reuse the **`run_*` kwargs** merge so notebooks match computation scripts ([same render block ~503](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py#L499-L511)).

The template already interpolates **`run_specific_batch_kwargs`** into the script ([`python_template.py.j2` L187] and **`**run_specific_batch_kwargs`** on `run_specific_batch` calls in the macro)—no `.j2` edit required unless you want a literal comment documenting the knob.

### 3. [`ProcessBatchOutputs_Bapun_Batch.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_Bapun_Batch.ipy)

- **Optional-only**: add a short comment near [`generate_batch_single_session_scripts`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_Bapun_Batch.ipy#L303-L311) that Bapun `run_*.py` scripts preflight **`BapunBatchHelpers.run_all`** before completion helpers (wired in library code). Avoid duplicating Python into the notebook per your `.ipy`/`run_all` duplication concern.

*(Per your notebook policy, any `.ipy` edit is optional and only commentary if you want it documented.)*

## Behavioral summary

| Script type | `format_name == 'bapun'` | Preflight |
|-------------|-------------------------|-----------|
| `run_*.py` / notebook-equivalent | yes | Runs `run_all` before completion |
| `run_*.py` | no | Skip |
| `figures_*.py` | yes/no | Skip (always `False`) |

[`ProcessBatchOutputs_Bapun_Batch.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_Bapun_Batch.ipy) continues to drive batch generation unchanged; regenerated `run_*.py` files gain the merged kwargs automatically next time phases run.
