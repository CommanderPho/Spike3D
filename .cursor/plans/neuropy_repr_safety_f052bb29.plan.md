---
name: Neuropy repr safety
overview: "Harden the small cluster of unsafe `__repr__` implementations in NeuroPy so broken nested reprs cannot cause `TypeError: __repr__ returned non-string`, while keeping the output style close to the current format."
todos:
  - id: repr-dynamic-container
    content: Harden `DynamicContainer.__repr__` with safe member-wise formatting.
    status: completed
  - id: repr-simple-printable
    content: Harden `SimplePrintable.__repr__` while preserving its current display style.
    status: completed
  - id: repr-sessionfolderspec
    content: Apply the same safe repr handling to `SessionFolderSpec.__repr__`.
    status: completed
  - id: verify-repr-smoke
    content: Run a small smoke check with broken nested reprs and review lints for edited files.
    status: completed
isProject: false
---

# NeuroPy `__repr__` safety cleanup

## Scope

Update the following files:

- [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\dynamic_container.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\dynamic_container.py)
- [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\print_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\print_helpers.py)
- [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\SessionSpecifications.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\SessionSpecifications.py)

## Problem

These implementations currently delegate directly to `dict.__repr__` or `self.__dict__` formatting, which can fail if any nested object has a broken `__repr__` that raises or returns a non-string:

```39:40:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\dynamic_container.py
    def __repr__(self):
        return f"{type(self).__name__}({self._mapping})"
```

```13:15:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\print_helpers.py
    def __repr__(self) -> str:
        ## TODO: this default printout is actually horrible for classes with any real content (like Pandas.DataFrame members, which spam the notebook)
        return f"<{self.__class__.__name__}: {self.__dict__};>"
```

## Implementation

1. Add defensive member-wise formatting in `DynamicContainer.__repr__`.
   - Iterate `self._mapping.items()`.
   - Use a tiny local safe-repr helper or equivalent inline logic.
   - If `repr(x)` raises, substitute a placeholder like `<repr failed: TypeName>`.
   - If `repr(x)` returns a non-`str`, coerce defensively and fall back if needed.
   - Keep the outer format close to current behavior: `ClassName({key: value, ...})`.

2. Apply the same safety principle to `SimplePrintable.__repr__`.
   - Render `self.__dict__` members explicitly instead of embedding `self.__dict__` wholesale.
   - Preserve its angle-bracket style as much as practical: `<ClassName: {...};>`.

3. Harden `SessionFolderSpec.__repr__` the same way for consistency.
   - Keep its current display shape, but ensure it can never bubble up the broken nested repr failure.

4. Add a quick smoke verification.
   - Reproduce with a dummy object whose `__repr__` returns a list or raises.
   - Verify `repr(...)` returns a real string for all three target types.
   - Run lints on the touched files and fix any introduced issues.

## Notes

- Keep changes minimal and local; no broader refactor or shared utility extraction unless it becomes clearly necessary while implementing.
- Keep function signatures on one line where possible, per workspace rules.
- Preserve current repr styles so notebook/debug output changes as little as possible beyond becoming safe.