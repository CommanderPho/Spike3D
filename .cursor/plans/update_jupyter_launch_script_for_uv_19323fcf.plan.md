---
name: Update Jupyter Launch Script for UV
overview: Rewrite the Jupyter Lab launch script to use the current Spike3D folder path dynamically and replace Poetry with UV package manager.
todos:
  - id: update-project-dir
    content: Replace hardcoded PROJECT_DIR with dynamic path calculation based on script's location
    status: completed
  - id: replace-poetry-with-uv
    content: Replace Poetry commands and checks with UV equivalents (uv run, uv --version check)
    status: completed
  - id: update-error-messages
    content: Update error messages to reference UV instead of Poetry
    status: completed
---

# Update Jupyter Lab Launch Script for UV

## Overview

Update [`Spike3D/scripts/unix/launch_jupyter_lab_with_remote_access.sh`](Spike3D/scripts/unix/launch_jupyter_lab_with_remote_access.sh) to:

1. Dynamically determine the Spike3D project directory based on the script's location instead of hardcoding `$HOME/repos/Spike3D`
2. Replace all Poetry commands with UV equivalents

## Changes Required

### 1. Dynamic Project Directory Detection

- Replace hardcoded `PROJECT_DIR="$HOME/repos/Spike3D"` with a dynamic path calculation
- Use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` to get the script's directory
- Calculate `PROJECT_DIR` as two levels up from `scripts/unix/` (i.e., `"$SCRIPT_DIR/../.."` and resolve to absolute path)
- Update `SCRIPT_PULL` to use the dynamic `PROJECT_DIR` instead of hardcoded path

### 2. Replace Poetry with UV

- Replace `poetry env info --path` check with `uv --version` check
- Replace `poetry run jupyter-lab` with `uv run jupyter-lab`
- Remove Poetry-specific environment path logic (lines 31-36) since UV handles this automatically with `uv run`

### 3. Update Error Messages

- Update error message from "Poetry could not be found" to "UV could not be found"

## Implementation Details

The script will:

- Calculate `PROJECT_DIR` as: `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" `then `PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"`
- Check for UV with: `if ! command -v uv &> /dev/null`
- Run Jupyter with: `uv run jupyter-lab --no-browser --port="$JUPYTER_PORT" --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*' > "$JUPYTER_LOG" 2>&1 &`

All other functionality (URL extraction, clipboard copying, cleanup, etc.) remains unchanged.