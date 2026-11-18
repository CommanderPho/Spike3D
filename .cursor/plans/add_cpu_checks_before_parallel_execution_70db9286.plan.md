---
name: Add CPU checks and num_workers override
overview: Add n_cpus validation and warnings before any parallel code execution, and add num_workers parameter to perform_predictive_decoding_analysis that overrides all parallel execution when set to 1.
todos:
  - id: "1"
    content: Add num_workers parameter to perform_predictive_decoding_analysis that overrides all parallel execution when set to 1
    status: completed
  - id: "2"
    content: Add CPU check and warning in compute_specific_future_and_past_analysis before ThreadPoolExecutor creation (around line 2814-2825)
    status: completed
  - id: "3"
    content: Improve CPU check and warning in _filter_single_epoch_result to clearly indicate when parallel was requested but cannot run (around line 3550-3559)
    status: completed
  - id: "4"
    content: Verify warnings are printed correctly and ThreadPoolExecutor is only created when n_cpus > 1
    status: completed
isProject: false
---

# Add CPU Checks, Warnings, and num_workers Override

## Overview

1. Add `num_workers: int` parameter to `perform_predictive_decoding_analysis` that, when set to 1, overrides all `max_workers` parameters and disables parallel execution (`use_parallel=False`, `parallel=False`) throughout the call hierarchy.
2. Ensure that before any parallel code (ThreadPoolExecutor) is called, the code checks `n_cpus` to verify parallel execution is possible, and prints a warning if parallel execution was requested but cannot run due to insufficient CPUs.

## Current Issues

1. `**compute_specific_future_and_past_analysis**` (lines 2814-2825):
  - Checks `n_cpus` but silently disables parallel execution without warning if user requested it
  - ThreadPoolExecutor is created at line 2825, but the check happens at line 2817
2. `**_filter_single_epoch_result**` (lines 3550-3686):
  - Checks `n_cpus < 2` and overrides settings, but the warning message doesn't clearly indicate if parallel was requested
  - The `parallel` parameter is passed to `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` at line 3686

## Changes Required

### 1. Add `num_workers` parameter to `perform_predictive_decoding_analysis` (around line 4177-4180)

**Location**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

**Changes**:

- Add `num_workers: Optional[int] = None` parameter to function signature (after `max_workers` parameter)
- When `num_workers=1`, override all parallel execution parameters:
  - Set `max_workers=1` for all function calls
  - Set `use_parallel=False` for `compute_specific_future_and_past_analysis` calls
  - Set `parallel=False` for `_filter_single_epoch_result` calls (via `final_refine_single_epoch_result_masks`)
- Pass these overridden values through `**kwargs` to:
  - `build_masked_container` (lines 4466, 4531)
  - `compute_future_and_past_analysis` (line 4497)
  - `final_refine_single_epoch_result_masks` (line 4547)

**Code pattern**:

```python
def perform_predictive_decoding_analysis(..., max_workers: Optional[int]=1, num_workers: Optional[int] = None, ...):
    # If num_workers is set to 1, override all parallel execution
    if num_workers == 1:
        max_workers = 1
        parallel_kwargs = {'max_workers': 1, 'use_parallel': False, 'parallel': False}
        print(f'[{_fn_name}] num_workers=1 specified: disabling all parallel execution')
    else:
        parallel_kwargs = {'max_workers': max_workers}
    
    # Pass parallel_kwargs to all function calls
    a_masked_container = a_container.build_masked_container(..., **parallel_kwargs)
    a_container.compute_future_and_past_analysis(..., **parallel_kwargs)
    a_masked_container.final_refine_single_epoch_result_masks(..., **parallel_kwargs)
```

### 2. Fix `compute_specific_future_and_past_analysis` (around line 2814-2825)

**Location**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

**Changes**:

- Store the original `use_parallel` value before modification
- Check `n_cpus` BEFORE creating ThreadPoolExecutor
- Print a warning if `use_parallel=True` was requested but `n_cpus <= 1`
- Ensure the check happens before line 2825 where ThreadPoolExecutor is created

**Code pattern**:

```python
# Store original value
use_parallel_requested = use_parallel

# Decide whether to run in parallel or serial
n_tasks: int = n_total_epochs
n_cpus: int = os.cpu_count() or 1

# Check if parallel was requested but can't run
if use_parallel_requested and n_cpus <= 1:
    import warnings
    warnings.warn(f"Parallel execution was requested (use_parallel=True) but cannot run: only {n_cpus} CPU(s) available. Running sequentially instead.", UserWarning)
    print(f"WARNING: Parallel execution requested but only {n_cpus} CPU(s) available. Running sequentially.")

use_parallel: bool = use_parallel and (n_tasks > 1) and (n_cpus > 1)
```

### 3. Fix `_filter_single_epoch_result` (around line 3550-3559)

**Location**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

**Changes**:

- Store the original `parallel` value before modification
- Check `n_cpus` BEFORE passing `parallel` to PeakPromenence function
- Print a warning if `parallel=True` was requested but `n_cpus < 2`
- Improve the warning message to be clearer

**Code pattern**:

```python
parallel: bool = kwargs.pop('parallel', True)
max_workers: int = kwargs.pop('max_workers', 4)

# Store original value
parallel_requested = parallel

n_cpus: int = os.cpu_count() or 1
if n_cpus < 2:
    if parallel_requested:
        import warnings
        warnings.warn(f"Parallel execution was requested (parallel=True) but cannot run: only {n_cpus} CPU(s) available. Running sequentially instead.", UserWarning)
        print(f"WARNING: Parallel execution requested but only {n_cpus} CPU(s) available. Overriding: max_workers=1, parallel=False")
    else:
        print(f'Only {n_cpus} CPU detected. Using max_workers=1, parallel=False')
    max_workers = 1
    parallel = False
else:
    if parallel_requested:
        print(f'Running in parallel: max_workers={max_workers}, parallel={parallel}')
```

### 4. Verify PeakPromenence function handles parallel correctly

**Note**: The `parallel` parameter is passed to `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` at line 3686. We should verify this function also checks CPUs before using parallel execution, but that's outside the scope of this file. The fix in `_filter_single_epoch_result` ensures we don't pass `parallel=True` when it can't work.

## Implementation Details

- Use Python's `warnings` module for proper warning handling
- Print user-friendly messages to console
- Check `n_cpus` using `os.cpu_count() or 1` (already used in code)
- Ensure checks happen BEFORE any ThreadPoolExecutor creation or parallel function calls
- Maintain backward compatibility - behavior should be the same, just with better warnings

## Testing Considerations

- Test `num_workers=1` parameter in `perform_predictive_decoding_analysis` to verify it disables all parallel execution
- Test with `n_cpus = 1` and `use_parallel=True` / `parallel=True` to verify warnings appear
- Test with `n_cpus > 1` and `use_parallel=True` / `parallel=True` to verify parallel execution still works
- Test with `use_parallel=False` / `parallel=False` to verify no warnings appear
- Test that `num_workers=1` overrides `max_workers` values passed to nested function calls

