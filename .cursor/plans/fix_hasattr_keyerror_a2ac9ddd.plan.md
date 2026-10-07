---
name: Fix hasattr KeyError
overview: Make DynamicParameters.__getattr__ raise AttributeError for a missing name, matching DynamicContainer, so hasattr and getattr-with-default work. Subscript access, has_attr, and the other bags stay unchanged.
todos:
  - id: fix-getattr
    content: Raise AttributeError from DynamicParameters.__getattr__ on KeyError and update the now-wrong hasattr comments
    status: completed
  - id: add-tests
    content: Cover hasattr, getattr default, has_attr, and subscript KeyError in test_dynamic_parameters.py
    status: completed
isProject: false
---

# Fix DynamicParameters hasattr

`hasattr` and `getattr(obj, name, default)` only treat a name as missing when they see `AttributeError`. In [`dynamic_parameters.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\dynamic_parameters.py), `__getattr__` re-raises the `KeyError` from `_mapping` except for `__deepcopy__`. [`DynamicContainer.__getattr__`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\dynamic_container.py) already does `raise AttributeError(item)` for every miss. `IdentifyingContext` stores real attributes and needs no change.

## Change

In `DynamicParameters.__getattr__`, replace the `__deepcopy__` special case with the same conversion for every missing name:

```python
except KeyError as err:
    if DynamicParameters.debug_enabled:
        print(...)
    raise AttributeError(item) from None
```

Keep the existing debug print, the commented `outcome_on_item_not_found` lines, and the `except Exception` handler. Update the class docstring item 3 and the `__getattr__` docstring so they no longer say `hasattr` is broken. `has_attr` stays; existing `plots.has_attr(...)` call sites keep working, and `hasattr` then agrees with them for mapping keys.

## Left unchanged

- `__getitem__` still raises `KeyError`, so `obj['missing']`, `MutableMapping.get`, and `key in obj` stay the same.
- Do not edit `DynamicContainer`, `IdentifyingContext`, `__getstate__` / `__setstate__`, or add a `.copy()` method. A new `copy` method would hide a stored key named `copy`.
- Subclasses (`RenderPlots`, `PhoUIContainer`, `RenderedEpochsItemsContainer`, and the others) inherit the method.

## Test

Add cases to [`tests/test_dynamic_parameters.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\tests\test_dynamic_parameters.py): missing name makes `hasattr` false and `getattr(..., default)` return the default; a present key is found; `obj['missing']` still raises `KeyError`; `has_attr` still matches mapping membership; `hasattr(obj, 'to_dict')` stays true.