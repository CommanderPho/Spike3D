---
name: Simplify to max_workers only
overview: Remove the redundant `num_workers` parameter and use only `max_workers` to control parallel execution. When `max_workers=1`, disable all parallel execution throughout the call hierarchy.
todos:
  - id: "1"
    content: Remove num_workers parameter from perform_predictive_decoding_analysis function signature
    status: completed
  - id: "2"
    content: Simplify parallel_kwargs logic to check max_workers == 1 instead of num_workers == 1
    status: completed
isProject: false
---

# Simplify to max_workers Only

## Overview

Remove the `num_workers` parameter and use only `max_workers` to control parallel execution. When `max_workers=1`, disable all parallel execution (`use_parallel=False`, `parallel=False`).

## Changes Required

### 1. Update `perform_predictive_decoding_analysis` function signature (line 4200)

**Location**: [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

**Change**: Remove `num_workers: Optional[int]=None` from the function signature

```python
# Before:
max_workers: Optional[int]=1, num_workers: Optional[int]=None,

# After:
max_workers: Optional[int]=1,
```

### 2. Simplify parallel_kwargs logic (lines 4247-4256)

**Change**: Update the logic to check `max_workers == 1` instead of `num_workers == 1`

```python
# Before:
if num_workers == 1:
    max_workers = 1
    parallel_kwargs = {'max_workers': 1, 'use_parallel': False, 'parallel': False}
    print(f'[{_fn_name}] num_workers=1 specified: disabling all parallel execution')
else:
    parallel_kwargs = {'max_workers': max_workers}
    if num_workers is not None:
        parallel_kwargs['max_workers'] = num_workers
        print(f'[{_fn_name}] num_workers={num_workers} specified: using max_workers={num_workers}')

# After:
if max_workers == 1:
    parallel_kwargs = {'max_workers': 1, 'use_parallel': False, 'parallel': False}
    print(f'[{_fn_name}] max_workers=1: disabling all parallel execution')
else:
    parallel_kwargs = {'max_workers': max_workers}
    print(f'[{_fn_name}] Using max_workers={max_workers} for parallel execution')
```

## Summary

- Remove `num_workers` parameter entirely
- Use `max_workers=1` to disable parallel execution
- The `parallel_kwargs` dictionary will still pass `use_parallel=False` and `parallel=False` to nested functions when `max_workers=1`

