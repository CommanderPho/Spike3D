---
name: Figure export exception logs
overview: Make posterior/figure-export failures dump a full traceback into the batch `*.log` (via flushed stdout/print), without aborting the rest of the session’s completion functions by default. Also flush progress markers so hard kills leave a last checkpoint.
todos:
  - id: add-traceback-import
    content: Add traceback import to batch_user_completion_helpers.py
    status: completed
  - id: update-figure-except-blocks
    content: In figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function, print full traceback (stdout+flush) on all display except blocks; gate raise on fail_on_exception_for_debugging / self.fail_on_exception
    status: completed
  - id: flush-progress-prints
    content: Add flush=True to per-display 'trying ...' progress prints so last checkpoint survives buffered/hard-kill exits
    status: completed
isProject: false
---

# Figure-export exceptions visible in batch `*.log`

## Full log analysis (`debug_2026-09-15_12-09-41...Final.log`)

File ends at **line 432** (~83KB). Timeline of relevant events:

| Time / line | Event |
|-------------|--------|
| L101 | Instantaneous FR: `CapturedException` / `KeyError: 0` (warn only; continues) |
| L355–373 | Sweep completion: **ExceptionPrintingContext one-liner** `AssertionError` shape 715 vs 680 → continues |
| L374–413 | `compute_and_export_decoders...` starts; enters `directional_decoders_evaluate_epochs` |
| 13:19:42 L414 | `generalized_decode...` starts (EpochComputations missing → recompute) |
| 13:37:28 L419 | generalized_decode recompute done; CSVs written |
| 13:37:35 L429–431 | `figures_plot...` starts; **directional stacked** `trying` |
| 13:41:12 L432 | **MultiColor** `trying` — **LOG ENDS HERE** |

### Conclusions from this log

1. **Directional stacked succeeded** (next display started ~3.5 min later; no `failed with error`).
2. **MultiColor did not leave a Python exception in this log.** No `failed with error`, no ExceptionPrintingContext one-liner, no `done with`, no next completion function (`export_session_h5...`), no `COMPLETED` banner.
3. Most consistent with a **hard stop mid-MultiColor** (OOM / walltime / SIGKILL / node kill) or a **truncated/unflushed copy**, not a swallowed `except` in the figures function.
4. ExceptionPrintingContext **is** confirmed as a one-liner swallow for the earlier sweep `AssertionError` (L373) — that pattern still warrants fixing for real Python failures.
5. Missing `greyscale_shared_norm` / `combined/multi` for this session is consistent with MultiColor dying **before** stage-2 split export finished (or before any further prints flushed). Presence of `psuedo2D_ignore/raw_rgba` on disk (if from this run) would mean stage 1 wrote files while later log lines never flushed — check Slurm `.out`/`.err` / `sacct` for OOM/TIMEOUT.

### What better exception logging will and will not fix

- **Will help:** Python exceptions in directional / MultiColor / `post_export_build_combined_images` that today become one-liners or `{e}` only.
- **Will not help alone:** hard kills with no Python traceback. Need **flushed progress prints** so the last `trying ...` (and any mid-display markers) survive.

## Problem (code)

In [`figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py):

1. Local `except` blocks only print `failed with error: {e}` (no stack).
2. Directional / MultiColor then **`raise`**, which hits outer [`ExceptionPrintingContext`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py) (`suppress=True`) → one-liner via [`format_sys_exc_info`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/exception_helpers.py).
3. Unconditional `raise` aborts later figure subsections.
4. Progress `print(...)` calls lack `flush=True`, so buffered stdout can vanish on hard kill.

## Approach (safe, minimal)

Change **only** [`batch_user_completion_helpers.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) inside `figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function`.

Do **not** change global `ExceptionPrintingContext` or `format_sys_exc_info`.

Reuse existing kwarg `fail_on_exception_for_debugging: bool = False`.

### 1. Exception handlers

For each per-display `try/except` (directional, yellow/blue, hairy, MultiColor, placefield, occupancy, reliability, all-time tracks):

```python
except Exception as e:
    print(f'\tfigures_plot_...(...): "<display_name>" failed with error: {e}\n skipping.', flush=True)
    print(traceback.format_exc(), flush=True)  # full stack on stdout → *.log
    if fail_on_exception_for_debugging or getattr(self, 'fail_on_exception', False):
        raise
```

Remove unconditional `raise` on directional (~L4633–4635) and MultiColor (~L4742–4744).

### 2. Flushed progress markers

Add `flush=True` to each `\t trying "<display>"` print (and the function entry banner prints in this function) so a mid-MultiColor kill still leaves a clear last checkpoint in `*.log`.

Add `import traceback` at module top (or local import in the function).

## Out of scope

- Fixing missing `greyscale_shared_norm` / `combined/multi` root cause.
- Changing `BatchCompletionHandler`’s `ExceptionPrintingContext(suppress=True)`.
- Editing `ProcessBatchOutputs_qclus1246789_Only.ipy`.
- Diagnosing Slurm OOM/TIMEOUT (recommend checking `sacct` / `.err` separately for this job).

## Verification

1. On a **Python** failure: `*.log` shows a multi-frame traceback under the display failure banner.
2. On a **hard kill** mid-display: `*.log` still ends on a flushed `trying "..."` (or a later flushed mid-step print), matching this run’s abrupt end pattern but with less buffering risk.
