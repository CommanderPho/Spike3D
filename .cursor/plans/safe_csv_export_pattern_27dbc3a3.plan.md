---
name: Safe CSV export pattern
overview: Align the optional CSV export in `determine_decoded_context_uncertainty_as_fn_of_position` with existing patterns in the same file (directory creation, sanitized names, non-fatal errors) and with the pipeline’s output registration API used elsewhere in the package.
todos:
  - id: mkdir-once
    content: "When enable_export_path set: Path(), output/ mkdir(parents=True, exist_ok=True) before partition loop"
    status: completed
  - id: safe-write-register
    content: Sanitized basename, resolve(), to_csv(index=False), try/except WARN, register_output_file on success
    status: completed
  - id: doc-optional
    content: "Optional: document export behavior in docstring and/or output_provides"
    status: completed
isProject: false
---

# Safe, registered CSV export for decoded marginal posterior

## Current behavior and gaps

The export block in [`determine_decoded_context_uncertainty_as_fn_of_position`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) (lines 4186–4189) writes under `enable_export_path / "output" / ...` but never creates `output/`, so `to_csv` raises **FileNotFoundError** on a fresh base path. There is no **try/except** (unlike the sibling [`determine_percent_correctly_decoded_contexts`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) loop around 1933–1946, which catches failures per iteration and warns). There is no **`register_output_file`**, so exports are invisible to [`RegisteredOutputsMixin.register_output_file`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Loading.py) (same pattern as figure writers in [`ExportHelpers.build_and_write_to_file` / `write_to_file`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Mixins\ExportHelpers.py)).

The module already imports **`sanitize_filename_for_Windows`** ([line 89](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)), matching usage elsewhere in this file (e.g. figure basename around 7073).

## Recommended implementation (minimal, local change)

1. **Normalize base path**  
   When `enable_export_path is not None`, set `export_root = Path(enable_export_path)` and `out_dir = export_root / "output"`, then call **`out_dir.mkdir(parents=True, exist_ok=True)` once** before the `for a_pre_post_delta, ...` loop (same spirit as [`CellsFirstSpikeTimes.save_data_to_csvs`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) at 10956–10958).

2. **Stable, safe filenames**  
   Build the basename from:
   - `sanitize_filename_for_Windows(curr_active_pipeline.session_name)` (or a short fallback like `"unknown_session"` if missing),
   - `sanitize_filename_for_Windows(str(a_pre_post_delta))`,
   - `time_bin_size` (e.g. in the stem so reruns don’t blindly overwrite different bins),
   - optional: run date via existing `datetime` import, e.g. `datetime.now().strftime("%Y-%m-%d")`, instead of a hardcoded `2026-04-09` string.

3. **CSV write**  
   Use **`export_csv_path = out_dir / f"{...}.csv"`**, **`export_csv_path.resolve()`** for registration, and **`to_csv(..., index=False)`** to align with other CSV writers in this file (e.g. 10968). If you rely on the DataFrame index for analysis, switch to `index=True` only after confirming the index is meaningful (default RangeIndex → `index=False` is appropriate).

4. **Robustness (match `determine_percent_correctly_decoded_contexts`)**  
   Wrap each `to_csv` in **`try` / `except Exception`** (or a narrower set of OSError subclasses if you prefer): on failure, **`print` a WARN** with path and exception, **do not re-raise**, so joint matrices and Qt viewers still run.

5. **Register outputs**  
   After a successful write, call:
   - `curr_active_pipeline.register_output_file(output_path=resolved_path, output_metadata={...})`  
   with small metadata: e.g. `kind`, `pre_post_delta`, `time_bin_size`, and optionally function name. This matches pipeline conventions (see `Loading.py`).

6. **Docstring / metadata (optional, tiny)**  
   Extend the docstring for `enable_export_path` to mention: creates `output/`, registers paths, failures are non-fatal. Optionally add a brief note to `@function_attributes` `output_provides` if you want discoverability—many neighbors leave it empty, so this is optional.

## Out of scope (unless you want it in the same edit)

The function is annotated `-> pd.DataFrame` but currently has **no `return`** before the next section (lines 4207–4208 look like notebook residue). Fixing that would be a separate, one-line cleanup (`return a_decoded_marginal_posterior_df` or the appropriate frame); it is not required for the export behavior.

## Files to touch

- Only [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py): the `enable_export_path` branch and (optionally) docstring / `function_attributes`.
