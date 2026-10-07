---
name: Deepcopy handling test
overview: Add one unit test that deepcopy of a DynamicParameters copies nested mutables into a separate object, so removing the AttributeError on a missing `__deepcopy__` fails the test.
todos:
  - id: add-deepcopy-test
    content: Add test_deepcopy_copies_nested_values to tests/test_dynamic_parameters.py and run the file
    status: completed
isProject: false
---

# Deepcopy handling test

Add `test_deepcopy_copies_nested_values` to [`tests/test_dynamic_parameters.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\tests\test_dynamic_parameters.py). Do not change [`dynamic_parameters.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\dynamic_parameters.py).

`copy.deepcopy` calls `getattr(obj, "__deepcopy__", None)`. The current `__getattr__` raises `AttributeError` only when that name is missing, then `deepcopy` uses `__getstate__` / `__setstate__`. The test should fail with `KeyError: '__deepcopy__'` if that conversion is removed.

Build an object with a mutable list and a nested `DynamicParameters`:

```python
original = DynamicParameters(prop0=9, prop9=['a'], nested=DynamicParameters(inner=1))
cloned = copy.deepcopy(original)
```

Assert:

- `cloned` is a different `DynamicParameters` than `original`
- `cloned.prop0 == 9`, `cloned.prop9 == ['a']`, and `cloned.nested.inner == 1`
- `cloned.prop9` is not `original.prop9`, and `cloned.nested` is not `original.nested`
- appending to `cloned.prop9` and assigning `cloned.nested.inner = 2` leaves the original list and nested value unchanged
- `list(cloned.original_attributes)` matches the original init keys

Run `uv run python tests/test_dynamic_parameters.py` from `pyPhoCoreHelpers`.