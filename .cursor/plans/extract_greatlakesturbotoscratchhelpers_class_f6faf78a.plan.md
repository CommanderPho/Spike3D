---
name: Extract GreatlakesTurboToScratchHelpers class
overview: Move all Greatlakes Turbo-to-Scratch staging logic out of the batch-output script into a new self-contained classmethod-only class `GreatlakesTurboToScratchHelpers` in `pythonScriptTemplating.py`, leaving the script with a single `FORCE_SCRATCH_WRITES` switch and a few compact gated call sites; the general time-bin pickling improvements stay untouched.
todos:
  - id: add-class
    content: Add GreatlakesTurboToScratchHelpers (class attrs + 7 moved classmethods + 4 new wrapper classmethods) to pythonScriptTemplating.py with shutil/ConcreteSessionFolder imports
    status: completed
  - id: script-remove-helpers
    content: Delete the 7 top-level scratch functions and 3 sidecar constants from ProcessBatchOutputs_qclus1246789_Only.ipy; add class to import line
    status: completed
  - id: script-gated-callsites
    content: Replace detection, output-path, assert, and staging blocks in the script with compact FORCE_SCRATCH_WRITES/use_scratch_writes-gated calls into the class
    status: completed
  - id: verify
    content: uv run import check, compile check of de-magic'd script copy in /tmp, grep for stale symbol references, lint
    status: completed
isProject: false
---

# Extract GreatlakesTurboToScratchHelpers class

## Where the class goes

Add `GreatlakesTurboToScratchHelpers` to [pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) (the batch-script-generation module the script already imports `generate_batch_single_session_scripts` from), appended after `build_windows_powershell_run_script`. Add `import shutil` and `from pyphoplacecellanalysis.General.Batch.concrete_session_folder import ConcreteSessionFolder` to that module's imports (no cycle: `concrete_session_folder.py` does not import `pythonScriptTemplating`).

## Class design (all `@classmethod`, no new top-level functions)

Class attributes (defaults, overridable per call via kwargs):
- `TURBO_DATA_ROOT = Path('/nfs/turbo/umms-kdiba/Data')`, `SCRATCH_DATA_ROOT = Path('/scratch/kdiba_root/kdiba99/halechr/Data')`, `SCRATCH_ANCHOR_PATH = Path('/scratch')`, `TURBO_PATH_MARKERS = ('/nfs/turbo/', '/cloud/turbo/')`
- `READONLY_SESSION_SIDECAR_SUFFIXES`, `READONLY_SESSION_SIDECAR_STANDALONE_NAMES`, `REQUIRED_SESSION_SIDECAR_SUFFIXES` (moved verbatim with their comments)

Methods moved verbatim from the script (bodies unchanged except `cls.`-qualified references):
- `kdiba_session_relpath(cls, context)`
- `build_scratch_stage_relpaths(cls, pickle_suffix, tbin_suffix='')`
- `find_turbo_session_basedir(cls, a_folder, source_data_root)`
- `ensure_scratch_kdiba_layout_symlink(cls, scratch_data_root)`
- `cleanup_leftover_scratch_symlinks(cls, scratch_basedir)`
- `copy_or_skip_mutable_file(cls, src_file, dest_file, skip_if_dest_newer_or_equal)`
- `stage_mutable_session_outputs_to_scratch(cls, turbo_session_folders, source_data_root, scratch_data_root, load_only_relpath, writable_relpath, skip_if_dest_newer_or_equal=True)`

New methods wrapping the inline script blocks:
- `detect_scratch_write_mode(cls, force_scratch_writes: bool, scratch_data_root=None, scratch_anchor_path=None) -> Tuple[Optional[Path], Optional[Path], bool]` returns `(turbo_data_root, scratch_data_root, use_scratch_writes)`; same Windows/anchor/mkdir/OSError logic and prints as script lines 259-275.
- `build_scratch_output_paths(cls, scratch_data_root=None) -> Tuple[Path, Path]` returns `(scripts_output_path, collected_outputs_path)` under `<scratch>/Output/{gen_scripts,collected_outputs}` with `mkdir(parents=True, exist_ok=True)` (script lines 281-285).
- `assert_output_paths_not_on_turbo(cls, scripts_output_path, collected_outputs_path)` performs the two asserts + `[SCRATCH WRITES]` prints (script lines 299-306).
- `stage_sessions_for_scratch_run(cls, turbo_session_folders, source_data_root, pickle_suffix, tbin_suffix, scratch_data_root=None, skip_if_dest_newer_or_equal=True) -> Tuple[List[ConcreteSessionFolder], Dict[IdentifyingContext, Path], Path]` returns `(scratch_session_folders, session_global_data_root_parent_paths, scratch_write_root)`; encapsulates script lines 741-768: build relpaths + print, already-on-scratch short-circuit, else `stage_mutable_session_outputs_to_scratch`, the missing-required-pickle warning loop, and the final job-write-root print. `scratch_data_root=None` defaults to the unresolved `cls.SCRATCH_DATA_ROOT` to preserve the current behavior (the script currently uses the unresolved `SCRATCH_DATA_ROOT` for the write root and symlink parent).

Docstring on the class states purpose ("temporary workaround while Turbo is read-only on Greatlakes") and shows the three-call usage from the script.

## Script changes ([ProcessBatchOutputs_qclus1246789_Only.ipy](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy))

- Import: extend the existing `pythonScriptTemplating` import line with `GreatlakesTurboToScratchHelpers`.
- Delete lines 71-243 (the 7 functions + 3 constants) entirely.
- Replace lines 252-275 with a clearly delimited block:

```python
# ==== GL Turbo->Scratch workaround (set False / comment out when Turbo is R/W again) ====
FORCE_SCRATCH_WRITES: bool = True  # set False when Turbo is writable again
turbo_data_root, scratch_data_root, use_scratch_writes = GreatlakesTurboToScratchHelpers.detect_scratch_write_mode(force_scratch_writes=FORCE_SCRATCH_WRITES)
print(f'FORCE_SCRATCH_WRITES: {FORCE_SCRATCH_WRITES}') ... (keep existing 4 prints)
```
- Lines 281-285: `if use_scratch_writes: scripts_output_path, collected_outputs_path = GreatlakesTurboToScratchHelpers.build_scratch_output_paths()` ; `else:` branch unchanged.
- Lines 299-306: `if FORCE_SCRATCH_WRITES: GreatlakesTurboToScratchHelpers.assert_output_paths_not_on_turbo(scripts_output_path, collected_outputs_path)`.
- Line 414: remove `turbo_session_concrete_folders = list(...)` (fold into the staging call); keep `session_global_data_root_parent_paths = None` (general, feeds `generate_batch_single_session_scripts`).
- Lines 741-768: replace with

```python
if use_scratch_writes:
    good_session_concrete_folders, session_global_data_root_parent_paths, global_data_root_parent_path = GreatlakesTurboToScratchHelpers.stage_sessions_for_scratch_run(turbo_session_folders=good_session_concrete_folders, source_data_root=global_data_root_parent_path, pickle_suffix=active_phase_dict.get('override_custom_pickle_suffix', ''), tbin_suffix=active_phase_dict.get('decoding_time_bin_filename_suffix', ''), skip_if_dest_newer_or_equal=True)
    _time_checkpoint('paths: GreatlakesTurboToScratchHelpers.stage_sessions_for_scratch_run')
```
- Everything else stays: `session_global_data_root_parent_paths=` kwarg, `should_save_local_pipeline_pkl`/`skip_save_on_initial_load`, `decoding_time_bin_filename_suffix`, tbin-only completion dict, DESTINATION SUMMARY, `_try_copy_to_clipboard`. Existing imports are left in place per the preserve-imports rule.

## Verification
- `python -c "import pyphoplacecellanalysis.General.Batch.pythonScriptTemplating as m; print(m.GreatlakesTurboToScratchHelpers.build_scratch_stage_relpaths('_x', '-tbin_75ms'))"` via `uv run` to confirm import + no cycle.
- `py_compile` the `.ipy` body (strip `%` magics to a temp copy under /tmp) to catch NameErrors from the removed symbols; grep the script for any remaining references to the deleted function names.
- Lint check on both files.