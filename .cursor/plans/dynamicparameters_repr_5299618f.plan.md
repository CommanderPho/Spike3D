---
name: DynamicParameters __repr__
overview: Update `DynamicParameters.__repr__` in pyPhoCoreHelpers to list each mapping entry explicitly and to build the result defensively so the return value is always a valid `str`, even when nested `repr()` calls misbehave.
todos:
  - id: impl-repr
    content: Replace `__repr__` in dynamic_parameters.py with member-wise formatting and safe repr coercion/fallbacks.
    status: completed
  - id: verify-smoke
    content: "Quick manual or pytest smoke: empty instance, nested values, repr-throwing value."
    status: completed
isProject: false
---

# DynamicParameters `__repr__` improvements

## Context

Current implementation in [`pyPhoCoreHelpers/src/pyphocorehelpers/DataStructure/dynamic_parameters.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\dynamic_parameters.py):

```51:52:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\dynamic_parameters.py
    def __repr__(self):
        return f"{type(self).__name__}({self._mapping})"
```

This delegates value rendering to `dict.__repr__`, which calls each value’s `repr()`. If any value’s `__repr__` raises or (pathologically) returns a non-`str`, the whole `__repr__` fails or violates the `str` contract.

## Intended behavior

1. **Members visible in the repr string** — Build the inner part by iterating `self._mapping.items()` (insertion order, same as today’s dict) and formatting each entry as `key_repr: value_repr`, joined with `", "`, so each member is explicit rather than relying on a single nested dict repr.

2. **Always return a `str`** — For each key and value:
   - Wrap `repr(x)` in `try/except`; on failure use a short fallback like `<repr failed: TypeName>`.
   - If `repr(x)` succeeds but returns a non-`str`, coerce with `str(...)` (defensive against broken user types).
   - Optionally wrap the whole body in an outer `try/except` and return `f"{type(self).__name__}(<repr error>)` if iteration itself fails (e.g. concurrent mutation), so `__repr__` never raises.

## Implementation notes

- **Single-line signatures** per workspace rules: keep `def __repr__(self):` on one line; if a small private helper is needed (e.g. `_safe_repr(obj)`), use a one-line `def` or inline the try/except in the method to avoid unnecessary multi-line defs unless length forces it.
- **Minimal diff**: Only change `__repr__` (and add a tiny helper in the same file only if it keeps the method readable without expanding scope).
- **No `print()` in `__repr__`**: Standard practice is to return a string only; “print the members” is interpreted as “show members in the repr output.” If you instead want logging-style `print` side effects, say so and we can adjust.

## Verification

- Smoke test in REPL or a tiny test: empty `DynamicParameters()`, nested `DynamicParameters`, and a dummy class whose `__repr__` raises — repr should remain a non-empty string and not raise.
