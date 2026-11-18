---
name: Bapun-aligned batch defaults
overview: Replace lingering KDIBA-first defaults with format-neutral batch behavior—especially KDIBA-specific completion hooks and hardcoded(epoch/epoch-comment) assumptions—while keeping KDIBA pipelines working via explicit overrides where needed (import side-effects + optional completion dict).
todos:
  - id: remove-phase0-kdiba-fixup
    content: Strip `kdiba_session_post_fixup_completion_function` from `phase0_any_run_custom_user_completion_functions_dict` in `pythonScriptTemplating.py`; trim import if unused.
    status: completed
  - id: clean-main-defaults
    content: Remove KDIBA fixup from `MAIN_get_template_string(..., override_custom_user_completion_functions_dict=None)` default dict in `batch_user_completion_helpers.py`.
    status: completed
  - id: migrate-kdiba-drivers
    content: Grep usages of `MAIN_get_template_string(` without override and add explicit fixup entries for KDIBA batches if any.
    status: completed
  - id: template-bapun-import
    content: Add `BapunDataSessionFormatRegisteredClass` side-effect import to `python_template.py.j2` next to KDIBA import.
    status: completed
  - id: batch-load-epoch-none
    content: Make `epoch_name_includelist` default `None` in `batch_load_session` (`NonInteractiveProcessing.py`) with short comment.
    status: completed
isProject: false
---

# Align batch defaults with Bapun-centric usage

## Problems found (KDIBA bleed)

1. **`ProcessingScriptPhases.clean_run` still merges KDIBA fixup**

   Despite the comment `# REMOVED`, [`phase0_any_run_custom_user_completion_functions_dict`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) still registers **`kdiba_session_post_fixup_completion_function`** ([L159–163](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py)). Any flow that merges phase completions into scripts without overriding (unlike [`ProcessBatchOutputs_Bapun_Batch.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_Bapun_Batch.ipy), which overrides via `MAIN_get_template_string`) can still enqueue KDIBA post-fix.

2. **`MAIN_get_template_string(None)` mega-default is KDIBA-heavy**

   When **`override_custom_user_completion_functions_dict` is `None`**, [`batch_user_completion_helpers.py` L3911–3933](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) still includes **`kdiba_session_post_fixup_completion_function`** (with a contradictory commented line beside it).

3. **Generated standalone script imports are KDIBA-only for format registration**

   [`python_template.py.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\python_template.py.j2) imports **`KDibaOldDataSessionFormatRegisteredClass`** ([L21–23](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\python_template.py.j2)) but **not Bapun** (your Bapun driver notebook imports `BapunDataSessionFormatRegisteredClass` explicitly). KDIBA-centric comments in the Neptune block are benign; missing Bapun import can matter if a script is invoked in a minimal environment before other modules register formats.

4. **`batch_load_session` default epochs are KDIBA-shaped**

   [`NonInteractiveProcessing.py` L88](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\NonInteractiveProcessing.py) uses `kwargs.get('epoch_name_includelist', ['maze1','maze2','maze'])`. Callers passing **`epoch_name_includelist=None`** get **`None`** (correct for Bapun: [`BapunDataSessionFormat.build_filters_any_maze_epochs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\Specific\BapunDataSessionFormat.py) resolves `maze*` vs `['roam','sprinkle']`). Callers omitting the key still get KDIBA defaults; format-neutral **`None`** is better.

## Recommended implementation

### A. Completion defaults (KDIBA cleanup)

- **`pythonScriptTemplating.py`**: Remove **`kdiba_session_post_fixup_completion_function`** from **`phase0_any_run_custom_user_completion_functions_dict`** (leave it commented under “KDIBA-only; add via `extra_run_functions`”). Optionally drop unused import if unused after removal.

- **`batch_user_completion_helpers.py`** (`MAIN_get_template_string`): Remove **`kdiba_session_post_fixup_completion_function`** from the **`override_custom_user_completion_functions_dict is None`** branch (or rename branch docstring to “legacy full kdiba”; prefer removal + document that KDIBA batch drivers must pass **`override_custom_user_completion_functions_dict`** that explicitly includes fixup).

**KDIBA batch migration:** Any KDIBA driver that relied on **`MAIN_get_template_string()`** without override must gain an explicit **`override_custom_user_completion_functions_dict`** containing `kdiba_session_post_fixup_completion_function`.

### B. Generated script imports (Bapun registration)

- **`python_template.py.j2`**: After the KDIBA import block, add a small side-effect import, matching your driver notebook:

```python
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
```

(Optional: Rachel/Hiro/KDIBA parity can stay minimal—Bapun is the gap for your batch path.)

### C. Epoch includelist default (format-neutral)

- **`batch_load_session`**: Change

`epoch_name_includelist = kwargs.get('epoch_name_includelist', ['maze1','maze2','maze'])`

→ **`kwargs.get('epoch_name_includelist', None)`** (keep explicit `None` behavior). Document in a one-line comment that KDIBA-era scripts should pass `['maze1','maze2','maze']` explicitly when needed.

## Out of scope / follow-up

- **[`slurm_template.sh.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\slurm_template.sh.j2)** `#SBATCH --account=kdiba1` — remains unless you define a parameterized `slurm_account` rendered from **`generate_batch_single_session_scripts`**.

- **Computation phase lists** in `get_run_configuration` are anatomically KDIBA/long-short oriented but still run for Bapun; revisit only if you want a **`format_name`**-scoped allowlist later.
