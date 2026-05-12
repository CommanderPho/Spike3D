---
name: Bapun-only batch script
overview: Deliver a brand-new IPython batch driver dedicated to Bapun sessions (no KDIBA sessions or KDIBA-specific hooks), reusing the existing format-agnostic batch machinery in pyPhoplacecellanalysis unless a gap is found.
todos:
  - id: add-bapun-ipy
    content: Add new Spike3D root script (e.g. ProcessBatchOutputs_Bapun_Batch.ipy) — Bapun contexts only, no kdiba post-fixup, distinct BATCH_DATE_TO_USE
    status: completed
  - id: pipeline-gap-check
    content: Confirm generate_batch_single_session_scripts / ProcessingScriptPhases work for bapun-only batches; only add new batch pipeline code if a Bapun-specific template or hook gap appears
    status: completed
  - id: verify-contexts
    content: Align included_session_contexts with BapunDataSessionFormatRegisteredClass keys; optional RatK commented
    status: completed
isProject: false
---

# Brand-new Bapun batch script (not KDIBA)

## Goal

Provide a **new, dedicated batch entry script** that processes **only** Bapun sessions (`format_name='bapun'`, `animal`, `session_name`). It must **not** include KDIBA session contexts or KDIBA-specific completion functions. This is separate from [`ProcessBatchOutputs_test_all_script_phases.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_test_all_script_phases.ipy), which remains the KDIBA-oriented driver.

## Batch / pipeline assessment (new vs existing)

- **Existing batch pipeline (reuse, no duplicate framework):** [`runBatch.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/runBatch.py), `generate_batch_single_session_scripts`, `ProcessingScriptPhases`, and `ConcreteSessionFolder.build_concrete_session_folders` already operate per registered session format. They require **homogeneous** `format_name` within one batch (assert ~254–256 in `build_concrete_session_folders`).
- **Deliverable “pipeline”:** For typical Bapun batch runs, **a new `.ipy` driver file is sufficient** — it selects Bapun `IdentifyingContext` list, wires `process_all_phases`-style orchestration (or equivalent calls from [`ProcessBatchOutputs_test_all_script_phases.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_test_all_script_phases.ipy)), and omits KDIBA-only hooks.
- **Only if needed:** Add or adjust **library** batch code (e.g. templating, completion helpers) if exercising a full Bapun phase run reveals a **Bapun-specific** failure; that would be a follow-up change outside the new script.

## Bapun session identities

Contexts must match the registry in [`BapunDataSessionFormat.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/core/session/Formats/Specific/BapunDataSessionFormat.py) (see `_get_session_specific_parameters`), for example:

- `RatN` / `Day4OpenField`
- `RatU` / `Day5OpenfieldSD` (and/or legacy `RatUDay5OpenfieldSD` if still used on disk)
- `RatS` / `Day5TwoNovel`
- `RatK` / `Day4Openfield` — optional / commented if unreliable per notebooks

## Implementation (after plan approval)

1. **New file** in Spike3D repo root with a **Bapun-specific name** (e.g. **`ProcessBatchOutputs_Bapun_Batch.ipy`**). Base structure on `process_all_phases` + imports from the test-all-phases script, but treat it as a **standalone Bapun batch driver** (clear file header comment).

2. **`included_session_contexts`:** Only `IdentifyingContext(format_name='bapun', ...)`. No `kdiba` entries or `exper_name`.

3. **`phase_any_run_custom_user_completion_functions_dict`:** **Exclude** `kdiba_session_post_fixup_completion_function`. Keep non–format-specific helpers (exports, decoding, sweep, backup, etc.) as appropriate; trim further if any helper assumes KDIBA paths.

4. **`BATCH_DAY_DATE` / `BATCH_DATE_TO_USE`:** Distinct tag for Bapun batches so artifacts and collected outputs never collide with KDIBA runs.

5. **Optional:** Bapun-specific `job_suffix` / qclu defaults if you want batch filenames to read as Bapun runs; otherwise mirror proven parameters from the reference script.

## Verification

- Run on a machine where `global_data_root_parent_path` resolves and `BapunDataSessionFormatRegisteredClass.build_session_basedirs_dict` includes the chosen contexts.
- Assert one format only: all `included_session_contexts` have `format_name='bapun'`.
- **Pipeline gap check:** Complete at least one phase of generated scripts for one Bapun session; if generation or Slurm/PowerShell scripts fail only for Bapun, file a targeted fix in templating or handlers (separate from the `.ipy`).
