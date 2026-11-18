---
name: Full batch error output
overview: Surface complete Python tracebacks for failed batch jobs by recording `CapturedException.get_full_traceback()` instead of the short repr, and optionally improving how PowerShell prints multi-line errors and Python process output.
todos:
  - id: runbatch-full-tb
    content: In runBatch.run_specific_batch, set _out_error and new_print to use CapturedException.get_full_traceback() (except + post_run_callback paths).
    status: completed
  - id: ps1-multiline-errors
    content: In powershell_template.ps1.j2, Write-Host Python errors as a multi-line block in Manage-JobQueue and WaitForAllJobs.
    status: completed
  - id: ps1-out-null-optional
    content: Optionally remove or gate `| Out-Null` on Python invocation so stdout is visible; document tradeoff for parallel jobs.
    status: completed
  - id: regen-scripts
    content: Re-run batch script generation so deployed gen_scripts/*.ps1 include template changes.
    status: completed
isProject: false
---

# Full Python stacktrace for generated batch runs

## Why you only see a one-line “traceback”

1. **`CapturedException` string form is intentionally short**  
   In [`pyPhoCoreHelpers/src/pyphocorehelpers/exception_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\exception_helpers.py), `CapturedException` defines `__repr__` as `CapturedException(..., traceback=<summary>)` where the summary comes from `get_traceback_summary()` (single file:line + function + last error line). The **full** traceback is already implemented as `get_full_traceback()` (uses `''.join(format_exception(*self.exc_info))`).

2. **Batch path stores the short form**  
   In [`run_specific_batch`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\runBatch.py), when `fail_on_exception` is false (your generated scripts default this way in [`python_template.py.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\python_template.py.j2)), failures set `_out_error = f"{an_error}"`, which uses that short repr. The same value is written to `status.json` as `"errors": str(run_errors)` (template ~line 351), and PowerShell only echoes that into one line in [`powershell_template.ps1.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\powershell_template.ps1.j2).

3. **Stdout from Python is discarded in the job**  
   The PS1 block uses `& $pythonExec $scriptPath | Out-Null`, so normal **stdout** (including template `new_print` lines like `finished run` / `errors: ...`) is dropped inside the job. Stderr can still show some warnings, which matches what you see.

```mermaid
flowchart LR
  Py[python_template.py.j2]
  RS[run_specific_batch]
  CE[CapturedException]
  SJ[status.json errors]
  PS[powershell Write-Host]
  Py --> RS
  RS --> CE
  CE -->|str repr today| SJ
  SJ --> PS
```

## Recommended code changes (minimal, targeted)

### 1. Record and log the full traceback in `run_specific_batch`

**File:** [`runBatch.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\runBatch.py)

- In the `except` block (~1175–1185): set `_out_error = an_error.get_full_traceback()` (not `f"{an_error}"`). Optionally keep a one-line prefix for grepability, e.g. `f"{an_error!r}\n{an_error.get_full_traceback()}"` if you still want the compact summary first.
- In the `post_run_callback_fn` failure path (~1197–1205): same change when assigning `_out_error`.
- Update `new_print(f'exception occured: {an_error}')` (and the callback error line) to print `an_error.get_full_traceback()` so file logging / any captured stream matches what is persisted.

No change required to **`python_template.py.j2`** for `status_data["errors"]` if you keep `run_errors` as a string: it will automatically contain the full traceback once `run_specific_batch` returns it.

### 2. PowerShell: print multi-line errors clearly

**File:** [`powershell_template.ps1.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\powershell_template.ps1.j2)

In both places that build `$statusInfo` (`Manage-JobQueue` ~74–86 and `WaitForAllJobs` ~143–158):

- Keep the short header line (`Python Status`, session id).
- If `$hashtable.PythonStatus.errors` is non-empty, **append a separate `Write-Host` block** (e.g. a banner line + the raw error text) instead of stuffing the entire traceback into parentheses on one line. That avoids awkward wrapping and makes stacks readable.

### 3. Optional: show Python stdout during the job

Still in **`powershell_template.ps1.j2`**, line ~17: removing `| Out-Null` from `& $pythonExec $scriptPath` will let `new_print`/normal prints appear in the job output (and can be echoed if you extend `Receive-Job` handling). **Tradeoff:** noisy output when many jobs run in parallel. A middle ground is to remove `Out-Null` only behind a parameter or comment it with a clear toggle—only include if you want live streaming logs.

## After implementation

- **Regenerate** the PowerShell and Python batch artifacts (whatever flow uses `pythonScriptTemplating` / Jinja templates) so `W:\Data\Output\gen_scripts\...` picks up the new PS1 template. Existing generated `.ps1` files on disk will not change until re-rendered.

## Out of scope (unless you want it)

- Changing `CapturedException.__str__` globally would affect every UI/log that expects a short message; prefer explicit `get_full_traceback()` at batch boundaries as above.
