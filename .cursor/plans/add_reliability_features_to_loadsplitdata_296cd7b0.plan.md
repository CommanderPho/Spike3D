---
name: Add reliability features to loadSplitData
overview: "Enhance `loadSplitData` with reliability features from `loadData`: ProgressMessagePrinter wrapper, explicit move_modules_list handling with global_move_modules_list default, structured exception handling (especially EOFError), and cross-platform path handling."
todos:
  - id: add_progress_printer
    content: Wrap loadSplitData main operation in ProgressMessagePrinter context manager
    status: completed
  - id: extract_move_modules
    content: Extract move_modules_list from kwargs with global_move_modules_list default and pass explicitly to loadData calls
    status: completed
  - id: add_eoferror_handling
    content: Add specific EOFError exception handling that re-raises (matching loadData behavior)
    status: completed
  - id: improve_exception_structure
    content: Restructure exception handling to match loadData pattern with specific error types before generic handler
    status: completed
  - id: add_path_handling
    content: Add WindowsPath/PosixPath cross-platform handling for path resolution
    status: completed
---

# Add Reliability Features to loadSplitData

## Overview

Enhance `loadSplitData` in [`pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py) to include the same reliability features present in `loadData`:

1. **ProgressMessagePrinter wrapper** - Show progress when loading split files
2. **Explicit `move_modules_list` handling** - Extract from kwargs with `global_move_modules_list` default
3. **Structured exception handling** - Add specific EOFError handling and improve exception structure
4. **Cross-platform path handling** - Add WindowsPath/PosixPath workarounds for path resolution

## Implementation Details

### 1. Add ProgressMessagePrinter Wrapper

Wrap the main loading operation in `ProgressMessagePrinter` similar to `loadData` (line 510), showing progress for loading the split folder.

### 2. Extract move_modules_list from kwargs

Add explicit handling at the start of the function (mirroring line 508 in `loadData`):

```python
active_move_modules_list: Dict = kwargs.pop('move_modules_list', global_move_modules_list)
```

Then pass `move_modules_list=active_move_modules_list` to `loadData` calls instead of using `**kwargs`.

### 3. Improve Exception Handling

- Add specific `EOFError` handling that re-raises (like `loadData` lines 541-544) instead of catching it in the generic `BaseException` handler
- Structure exception handling to match `loadData`'s pattern: try/except for specific errors, then generic exception handling
- Keep the current behavior of continuing on errors for individual files, but add specific handling for EOFError

### 4. Cross-Platform Path Handling

Add WindowsPath/PosixPath workaround for path resolution (similar to `loadData` lines 516-539), wrapping the path resolution logic in a try/except block that handles `NotImplementedError` for cross-platform path issues.

## Files to Modify

- [`pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py) - Update `loadSplitData` function (lines 565-664)

## Notes

- The function should maintain backward compatibility - all existing behavior should be preserved
- Individual file loading errors should still be caught and logged (not break the entire operation)
- EOFError should be re-raised immediately (not caught and continued) as it indicates a corrupted file