---
name: Refactor safeSaveSplitData
overview: Convert `safeSaveSplitData` from a method (with `self`) to a standalone function consistent with `safeSaveData`, removing all `self` references and adding required parameters.
todos:
  - id: "1"
    content: "Update function signature: remove `self`, add `computed_data`, `global_computation_results_pickle_path`, `output_path`, and `override_global_pickle_filename` parameters"
    status: completed
  - id: "2"
    content: Replace `self.global_computation_results.computed_data` references with `computed_data` parameter (lines 203, 206)
    status: completed
  - id: "3"
    content: Replace `self.global_computation_results_pickle_path` references with `global_computation_results_pickle_path` parameter (lines 178, 185)
    status: completed
  - id: "4"
    content: Replace `self.get_output_path()` with `output_path` parameter (line 188)
    status: completed
  - id: "5"
    content: Update docstring to reflect new function signature (remove method-style usage example)
    status: completed
---

# Refactor safeSaveSplitData to match safeSaveData pattern

## Overview

The function `safeSaveSplitData` currently has `self` as the first parameter and references `self` attributes throughout. It needs to be converted to a standalone function matching the pattern of `safeSaveData`.

## Current Issues

1. Function signature has `self` parameter (line 149)
2. References `self.global_computation_results_pickle_path` (lines 178, 185)
3. References `self.get_output_path()` (line 188)
4. References `self.global_computation_results.computed_data` (lines 203, 206)
5. References undefined variable `override_global_pickle_filename` (lines 178, 183, 188)

## Changes Required

### Function Signature

Change from:

```python
def safeSaveSplitData(self, pkl_path: Union[str, Path], include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True):
```

To:

```python
def safeSaveSplitData(pkl_path: Union[str, Path], computed_data: Dict[str, Any], include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True, global_computation_results_pickle_path: Optional[Union[str, Path]]=None, output_path: Optional[Union[str, Path]]=None, override_global_pickle_filename: Optional[str]=None):
```

### Parameter Additions

- `computed_data: Dict[str, Any] `- replaces `self.global_computation_results.computed_data`
- `global_computation_results_pickle_path: Optional[Union[str, Path]]=None `- replaces `self.global_computation_results_pickle_path`
- `output_path: Optional[Union[str, Path]]=None `- replaces `self.get_output_path()`
- `override_global_pickle_filename: Optional[str]=None` - currently referenced but missing from signature

### Code Changes

1. **Line 149**: Remove `self` from function signature, add new parameters
2. **Line 178**: Replace `self.global_computation_results_pickle_path` with `global_computation_results_pickle_path` parameter
3. **Line 185**: Replace `self.global_computation_results_pickle_path` with `global_computation_results_pickle_path` parameter  
4. **Line 188**: Replace `self.get_output_path()` with `output_path` parameter
5. **Line 203**: Replace `self.global_computation_results.computed_data.keys()` with `computed_data.keys()`
6. **Line 206**: Replace `self.global_computation_results.computed_data` with `computed_data` parameter

### Path Handling Logic

The path resolution logic should handle cases where:

- `pkl_path` is provided (file or directory)
- `global_computation_results_pickle_path` is provided as fallback
- `output_path` is used when `override_global_pickle_filename` is provided but `pkl_path` is None

## Files to Modify

- [`pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py) - Update function signature and remove all `self` references