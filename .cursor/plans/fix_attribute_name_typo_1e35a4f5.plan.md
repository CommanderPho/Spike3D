---
name: Fix attribute name typo
overview: Fix the typo where `.a_masked_container` is used instead of `.masked_container` in lines 4184-4186, which would cause an AttributeError at runtime.
todos:
  - id: fix-attribute-name
    content: Replace .a_masked_container with .masked_container on lines 4184 and 4186
    status: completed
---

# Fix Attribute Name Typo in PredictiveDecodingComputations.py

## Issue Found

There is a typo at lines 4184-4186 in [`PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py):

The code uses `.a_masked_container` but the actual class attribute is `masked_container` (as defined at line 3761).

**Current (incorrect):**

```python
if (a_masked_container is not None) and (global_computation_results.computed_data['PredictiveDecoding'].a_masked_container != a_masked_container):
    global_computation_results.computed_data['PredictiveDecoding'].a_masked_container = a_masked_container
```

**Should be:**

```python
if (a_masked_container is not None) and (global_computation_results.computed_data['PredictiveDecoding'].masked_container != a_masked_container):
    global_computation_results.computed_data['PredictiveDecoding'].masked_container = a_masked_container
```

## Fix

Change `.a_masked_container` to `.masked_container` on both lines 4184 and 4186.

This is consistent with:

- Line 3761 (class definition): `masked_container: Optional[PredictiveDecodingComputationsContainer]`
- Line 4113 (earlier assignment): `global_computation_results.computed_data['PredictiveDecoding'].masked_container = a_masked_container`