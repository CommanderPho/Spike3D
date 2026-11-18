---
name: Mirror qclus12 batch script
overview: Update ProcessBatchOutputs_qclus12_Only.ipy to match all recent infrastructure and pipeline changes from ProcessBatchOutputs_qclus1246789_Only.ipy, while keeping qclu [1,2] as the active variant configuration.
todos:
  - id: copy-template
    content: Use ProcessBatchOutputs_qclus1246789_Only.ipy as base template for qclus12 file
    status: completed
  - id: swap-qclus12-active
    content: Activate qclus12 suffix block; comment out 1246789 active block; fix track_body base_suffix default
    status: completed
  - id: verify-run
    content: Run ipython ProcessBatchOutputs_qclus12_Only.ipy and confirm timing summary + qclu_12 job_suffix
    status: completed
isProject: false
---

# Mirror qclus1246789 changes into ProcessBatchOutputs_qclus12_Only.ipy

## Goal

Bring [`ProcessBatchOutputs_qclus12_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus12_Only.ipy) up to date with [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy), **except** the active qclu/suffix block stays qclus12-specific (`included_qclu_values = [1, 2]`, `job_suffix ...qclu_12...`).

## Approach

Use the 1246789 file as the template and apply one targeted swap for the active suffix block. This avoids missing any of the many scattered changes (timing, imports, track-body variant, completion-function dict, etc.).

```mermaid
flowchart LR
    src[ProcessBatchOutputs_qclus1246789_Only.ipy]
    tgt[ProcessBatchOutputs_qclus12_Only.ipy]
    src -->|"copy structure + shared config"| tgt
    swap["swap active suffix block to qclus12"]
    tgt --> swap
```

---

## Changes to port (shared with 1246789)

### 1. Import optimization + timing instrumentation

Replace the heavy import block (~50 lines importing `runBatch`, `AcrossSessionResults`, `Loading`, etc.) with the slim block from 1246789:

- Add `import time`, `_time_checkpoint()` / `_print_timing_summary()` helpers
- Set headless env vars before library imports: `MPLBACKEND=Agg`, `MPLCONFIGDIR`, `QT_QPA_PLATFORM=offscreen`
- Minimal imports only:
  - `find_first_extant_path`, `copy_to_clipboard`
  - `IdentifyingContext`, `KDibaOldDataSessionFormatRegisteredClass`
  - `ConcreteSessionFolder` from [`concrete_session_folder.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/concrete_session_folder.py)
  - `ProcessingScriptPhases`, `BatchScriptsCollection`, `generate_batch_single_session_scripts`, `build_windows_powershell_run_script`
  - `batch_user_completion_helpers` imports (unchanged set)
- Add `_time_checkpoint(...)` calls at: stdlib, path_helpers, KDiba, concrete_session_folder, pythonScriptTemplating, batch_user_completion_helpers, paths, build_concrete_session_folders, MAIN_get_template_string, active_phase.get_run_configuration, generate_batch_single_session_scripts, build_windows_powershell_run_script, copy_to_clipboard, and `_print_timing_summary()` at end

### 2. Remove obsolete helpers

Delete from qclus12 (not present in 1246789):

- `execute_code_block()` function
- `enable_auto_code_block_execution` flag and its `subprocess` auto-run calls in the Linux sbatch section

### 3. IPython magic lines

Comment out active magics to match 1246789:

```python
# %config IPCompleter.use_jedi = False
# %pdb off
```

### 4. Batch date and paths

- `BATCH_DAY_DATE = '2026-08-29'`
- Add `'K:/scratch/gen_scripts'` to `known_scripts_output_paths` list

### 5. Completion functions dict

Replace the minimal qclus12 dict with the full active dict from 1246789 (backup, kdiba fixup, t_delta, sweep, decoders, instantaneous rates, peak info, trial-by-trial, generalized decode + figures, save_custom, export h5). Keep the commented alternate dict block for reference.

### 6. Session list updates

Match 1246789 session selection:

- Comment out `vvp01/one/2006-4-10_12-25-50` (discarded)
- Comment out duplicate bad `vvp01/two/2006-4-10_12-58-3` entry (keep one active)
- Add `pin01/one/fet11-01_12-58-54` with MATLAB fix comment
- Update `pin01/one/11-02_17-46-44` comment (MATLAB fix)
- Add `## Known Bad Sessions` commented reference block

### 7. Run phase and pipeline flags

- `active_phase = ProcessingScriptPhases.clean_run` (not `continued_run`)
- `active_phase_dict['should_freeze_pipeline_updates'] = False`

### 8. Custom suffix section (structure from 1246789, qclus12 active)

Copy the multi-variant commented suffix section from 1246789, but **invert which block is active**:

| Block | qclus12 file (target) |
|---|---|
| qclu `[1,2,4,6,7,8,9]` @ frateThresh 5.0 | commented |
| **qclu `[1,2]` @ frateThresh 2.0** | **ACTIVE** |
| qclu `[1,2,4,6,7,8,9]` @ frateThresh 2.0 | commented |
| qclu `[1,2,4,6,7,8,9]` @ frateThresh 0.2 | commented |

Active lines for qclus12:

```python
active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0"
included_qclu_values = [1, 2]
minimum_inclusion_fr_Hz = 2.0
job_suffix = f"_withNormalComputedReplays-qclu_12-frateThresh_2.0"
```

### 9. Track-body variant block (Phase 2)

Port the full active track-body variant section from 1246789 (~lines 256–285), adjusting the default fallback in `base_suffix = active_phase_dict.get(...)` to the qclus12 suffix:

```python
"..._withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0"
```

Includes: `override_custom_pickle_output_suffix`, `batch_session_completion_handler_kwargs` with `apply_track_body_aclu_filter`, narrowed `force_recompute_override_computations_includelist`, and `export_filename_extra_suffix_parts`.

### 10. Common computation parameters

Match 1246789 values:

- `num_shuffles: int = 1024`
- `drop_previous_result_and_compute_fresh: bool = True`
- `laps_decoding_time_bin_size = 0.075`
- `ripple_decoding_time_bin_size = 0.075`
- `job_suffix = f"{job_suffix}_tbin_75ms"`
- `perform_wcorr_shuffle_analysis` uses `drop_previous_result_and_compute_fresh` variable (not hardcoded `True`)

### 11. Force-recompute conditional

Replace unconditional assignment with track-body guard from 1246789:

```python
if not active_phase_dict.get('batch_session_completion_handler_kwargs', {}).get('apply_track_body_aclu_filter', False):
    active_phase_dict['force_recompute_override_computations_includelist'] = list(...)
```

### 12. custom_user_completion_function_override_kwargs_dict

Port all updated kwargs from 1246789:

- instantaneous spike rates: `[0.001, 1000.0]`
- wcorr shuffles: `should_skip_previous_saved_shuffles=True`
- trial-by-trial: `active_laps_decoding_time_bin_size=ripple_decoding_time_bin_size`
- generalized decode: `force_recompute=True, export_pkl=False`
- figures: explicit `included_figures_names` list (placefield formation, yellow/blue marginal, occupancy, trial-to-trial reliability)

### 13. Export filename suffix propagation

Add the Option 2 block that propagates `export_filename_extra_suffix_parts` into generalized decode, save_custom, and figures completion functions (with `## END for ...` closing comment per workspace rules).

### 14. Script generation flags

In `generate_batch_single_session_scripts(...)`:

- Add `should_use_viztracer_logging=False`
- Change `fail_on_exception = False` (was `True` in qclus12)
- Add `_time_checkpoint('generate_batch_single_session_scripts')`

### 15. Cleanup

- Remove stale commented `force_recompute_override_computations_includelist` and `custom_phase_extended_computations_include_includelist` blocks (1246789 already cleaned these)
- Remove extra `# %%` cell divider before the Windows/Linux runner section

---

## qclus12-specific values preserved (not copied from 1246789 active block)

- `included_qclu_values = [1, 2]`
- `override_custom_pickle_suffix = "...-qclu_[1, 2]-frateThresh_2.0"`
- `job_suffix = "...qclu_12-frateThresh_2.0"` (+ trackBodyPeakOnly + tbin_75ms suffixes when variant block active)
- Track-body `base_suffix` default fallback uses qclus12 suffix string

---

## Verification (after edit)

Run on the target machine (Linux or Windows):

```bash
ipython ProcessBatchOutputs_qclus12_Only.ipy
```

Confirm:

1. `[TIMING SUMMARY]` prints at end (imports should be much faster than old ~520s cold start)
2. `job_suffix` printed value contains `qclu_12` (not `qclu_1246789`)
3. `included_qclu_values` in generated config is `[1, 2]`
4. sbatch / powershell script paths print without error
5. No references to removed `execute_code_block` / `enable_auto_code_block_execution`

---

## File touched

Single file edit: [`ProcessBatchOutputs_qclus12_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus12_Only.ipy) only. No library or notebook changes required.
