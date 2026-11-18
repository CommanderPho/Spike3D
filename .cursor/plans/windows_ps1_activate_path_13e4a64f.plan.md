---
name: Windows PS1 activate path
overview: Fix Windows batch PowerShell runners failing at runtime by updating `build_windows_powershell_run_script` defaults to your Spike3D `.venv` `activate.ps1` and matching `python.exe`, then regenerating any already-emitted `.ps1` so embedded paths are not stale.
todos:
  - id: update-defaults
    content: Change `build_windows_powershell_run_script` defaults for both `activate_path` (activate.ps1) and `python_executable` (same .venv python.exe) in pythonScriptTemplating.py
    status: completed
  - id: regenerate-ps1
    content: Re-run script generation (or hand-edit) so existing files like `run_scripts_continued_run.ps1` no longer contain the old c:/Users/pho/... paths
    status: completed
  - id: template-doc
    content: "Optional: add one-line comment in powershell_template.ps1.j2 that activate_path should be activate.ps1"
    status: completed
isProject: false
---

# Update Windows `.ps1` output activate path (and prevent your runtime errors)

## Why `.\run_scripts_continued_run.ps1` failed

The **first** error is the real root cause:

- `The term 'c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv_UV/Scripts/activate.bat' is not recognized...`

That string is **not** coming from the template as a constant; it was **baked into the generated file** when Jinja rendered `{{activate_path}}` from the old default in [`build_windows_powershell_run_script`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) (`activate.bat` under `.venv_UV`).

- On your machine that path **does not exist**, so `& $activatePath` in [`powershell_template.ps1.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\powershell_template.ps1.j2) throws.
- Using **`activate.bat`** from PowerShell via **`&`** is also a brittle pattern compared to **`activate.ps1`** (same `&` invocation works for `.ps1`).

The **later** messages are **downstream**:

- **`InvalidOperation: You cannot call a method on a null-valued expression`** — typical when the script block’s `catch` or status handling assumes objects that never got populated after the activation failure.
- **`Completed Job 1 did not produce any output`** — `Start-Job` runs the script block in a worker process; if it errors before returning the expected hashtable, **`Receive-Job`** can look empty, which matches your line 212 `Manage-JobQueue` / job-pipeline behavior.

Fixing **`activate_path`** removes the primary failure so the block can proceed; aligning **`python_executable`** avoids a **second** hard failure (`& $pythonExec` pointing at a missing `.venv_UV\python.exe` even after a successful activate).

## What drives the emitted path

- Template only has **`{{activate_path}}`** and **`{{python_executable}}`** placeholders (no repo path literals).
- Defaults today (stale):

```761:764:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py
def build_windows_powershell_run_script(script_paths, max_concurrent_jobs: int = 3,
                                        activate_path='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv_UV/Scripts/activate.bat', 
                                        python_executable='c:/Users/pho/repos/Spike3DWorkEnv/Spike3D/.venv_UV/Scripts/python.exe',
                                        script_name: str = 'run_scripts'):
```

- Call sites (e.g. [`ProcessBatchOutputs.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs.ipy)) omit these kwargs, so **new** output picks up whatever defaults ship in the library.

## Implementation

1. **Edit** [`pythonScriptTemplating.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) defaults:
   - **`activate_path`**: `H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\Scripts\activate.ps1` (use a raw Python string or forward slashes—both fine for embedding into PS).
   - **`python_executable`** (required for full fix): **`...\Spike3D\.venv\Scripts\python.exe`** in the **same** venv as activation so both invocations resolve on disk.

2. **Regenerate or edit existing `.ps1` files**: Changing Python defaults does **not** rewrite `W:\Data\Output\gen_scripts\run_scripts_continued_run.ps1` on disk. You must either:
   - Re-run whatever step emits `run_scripts*.ps1` after upgrading the installed `pyPhoPlaceCellAnalysis` code, **or**
   - Search-replace the old `c:/Users/pho/.../activate.bat` and old `python.exe` strings inside that file.

   Until then, **`.\run_scripts_continued_run.ps1` will keep failing** because it still contains the old literals.

3. **[`powershell_template.ps1.j2`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Templates\powershell_template.ps1.j2)** — **No behavioral change required** for this fix; optional one-line comment that Windows runners should pass **`activate.ps1`**.

## Execution policy caveat (only if activation still blocked)

If `activate.ps1` exists but activation is blocked by policy, PowerShell may still error with an execution-policy message—not the same as “term not recognized.” That is orthogonal; your traceback matches **missing/wrong path + `.bat` default**, not policy.

## Out of scope (unless you ask)

- Other stale Windows literals in the same module (e.g. VS Code workspace helper ~546–551) — separate from `.ps1` runner failure.
- Hardening `WaitForAllJobs` / job output handling for clearer errors when the script block throws (nice-to-have, not required to fix your specific stack trace).
