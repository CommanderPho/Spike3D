---
name: Create Windows PowerShell Jupyter Launch Script
overview: Create a PowerShell (.ps1) version of the bash script that launches Jupyter Lab with remote access, extracts the URL from logs, copies it to clipboard, and maintains the process until termination.
todos:
  - id: create_ps1_script
    content: Create launch_jupyter_lab_with_remote_access.ps1 with all functionality from bash script
    status: completed
---

# Create Windows PowerShell Jupyter Launch Script

## Overview

Create `launch_jupyter_lab_with_remote_access.ps1` in `Spike3D/scripts/windows/` that replicates the functionality of the bash script `launch_jupyter_lab_with_remote_access.sh`.

## Key Functionality to Implement

1. **Script Setup**

- Use `$PSScriptRoot` to get script directory
- Calculate project root using `Split-Path -Parent` (two levels up from script)
- Set error handling with `$ErrorActionPreference = "Stop"`

2. **Variables**

- Jupyter port: 8889
- Log file: `$PROJECT_DIR/jupyter.log` (absolute path)
- Timeout: 60 seconds
- No clipboard command needed (use native `Set-Clipboard`)

3. **Clipboard Function**

- Create `Copy-ToClipboard` function using `Set-Clipboard` cmdlet

4. **UV Check**

- Use `Get-Command uv` to verify UV is available
- Exit with error message if not found

5. **Jupyter Launch**

- Use `Start-Process` with `-NoNewWindow` and redirect output to log file
- Store process ID for cleanup
- Alternative: Use `Start-Job` if background job is preferred

6. **Cleanup Handler**

- Use `try/finally` block to ensure Jupyter process is terminated on script exit
- Use `Stop-Process` with the stored PID

7. **URL Extraction**

- Poll log file using `Get-Content` and `Select-String` with regex pattern
- Pattern: `http://127\.0\.0\.1:8889/lab\?token=\w+`
- Wait up to timeout period with 2-second intervals

8. **Clipboard Copy**

- Copy extracted URL to clipboard using `Set-Clipboard`
- Display success message

9. **Process Maintenance**

- Use `Wait-Process` to keep script running until Jupyter terminates

## Implementation Details

- Follow existing PowerShell script patterns from `export_repo_requirements_txt.ps1` for directory navigation
- Use Windows-native clipboard (`Set-Clipboard`) instead of external tools
- Handle path separators correctly (Windows uses backslashes)
- Ensure proper error handling and cleanup on script termination

## Files to Create

- `Spike3D/scripts/windows/launch_jupyter_lab_with_remote_access.ps1`