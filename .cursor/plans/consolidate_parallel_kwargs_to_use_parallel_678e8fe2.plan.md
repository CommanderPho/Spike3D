---
name: Consolidate parallel kwargs to use_parallel
overview: Remove the redundant `parallel` boolean kwarg and consolidate all parallel execution control to use only `use_parallel` throughout PredictiveDecodingComputations.py. This includes updating kwargs extraction, variable names, print statements, and function calls.
todos:
  - id: "1"
    content: Remove `parallel` kwarg extraction and replace with `use_parallel` in `_filter_single_epoch_result` method (lines 3567-3586)
    status: completed
  - id: "2"
    content: Update all variable references from `parallel` to `use_parallel` in `_filter_single_epoch_result` method
    status: completed
  - id: "3"
    content: Update function call at line 3713 to pass `parallel=use_parallel` to external PeakPromenence function
    status: completed
  - id: "4"
    content: "Remove `'parallel': False` from `parallel_kwargs` dictionary in `perform_predictive_decoding_analysis` (line 4262)"
    status: completed
isProject: false
---

## Consolidate `parallel` and `use_parallel` to `use_parallel` only

Remove the redundant `parallel` boolean parameter and standardize on `use_parallel` throughout the file.

### Changes Required

1. **In `_filter_single_epoch_result` method (lines 3567-3586)**:
  - Remove `parallel: bool = kwargs.pop('parallel', True)` 
  - Keep only `use_parallel: bool = kwargs.pop('use_parallel', True)`
  - Replace all references to `parallel` variable with `use_parallel`
  - Update `parallel_requested` to `use_parallel_requested`
  - Update warning/print messages to reference `use_parallel` instead of `parallel`
  - Update line 3583: `parallel = False` → `use_parallel = False`
  - Update line 3586: print statement to use `use_parallel` instead of `parallel`
2. **In `_filter_single_epoch_result` method (line 3713)**:
  - Change `parallel=parallel` to `parallel=use_parallel` when calling `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation`
  - Note: The external function still accepts `parallel` parameter, so we map `use_parallel` to `parallel` in the call
3. **In `perform_predictive_decoding_analysis` function (line 4262)**:
  - Remove `'parallel': False` from `parallel_kwargs` dictionary
  - Change from: `parallel_kwargs = {'max_workers': 1, 'use_parallel': False, 'parallel': False}`
  - To: `parallel_kwargs = {'max_workers': 1, 'use_parallel': False}`

### Files to Modify

- `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)`

### Notes

- The external function `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` still accepts `parallel` parameter, so we pass `parallel=use_parallel` when calling it. This maintains compatibility while using `use_parallel` internally.
- All internal logic, variable names, and print statements will use `use_parallel` consistently.
- The `parallel_kwargs` dictionary will only contain `use_parallel`, removing the redundant `parallel` key.

