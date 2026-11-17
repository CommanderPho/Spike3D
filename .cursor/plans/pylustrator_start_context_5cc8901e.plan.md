---
name: pylustrator start context
overview: Make `pylustrator.start(...)` return a context manager so `with pylustrator.start(...):` installs hooks on entry (via the call) and always restores them on exit via `stop()`, while bare `pylustrator.start()` keeps working unchanged.
todos:
  - id: start-context
    content: Add _StartContext; make initialize return it on both install paths; update docstring
    status: completed
  - id: docs-context
    content: Document with pylustrator.start() in styling/index/CLAUDE
    status: completed
  - id: test-context
    content: Add context-manager coverage to test_hooks.py
    status: completed
isProject: false
---

# Add `with pylustrator.start(...):` context manager

## Approach

Keep the existing call-site API (`pylustrator.start()` installs hooks immediately). Change [`initialize`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\QtGuiDrag.py) to return a small context object whose `__exit__` calls [`stop()`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\QtGuiDrag.py). Because `start` is already `initialize as start` in [`__init__.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\__init__.py), no export rename is needed.

```python
with pylustrator.start():
    # hooks active
    ...
# __exit__ → stop(); default matplotlib restored
```

Bare usage stays valid:

```python
pylustrator.start()  # installs; returned context ignored
plt.show()           # already auto-stops today
```

## Implementation

In [`QtGuiDrag.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\QtGuiDrag.py):

1. Add a tiny class (module-private is fine):

```python
class _StartContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        stop()
        return False
```

2. At the end of `initialize` (both the early-return path when `_hooks_installed` and the fresh-install path), `return _StartContext()`.

3. Update the `initialize` docstring to document `with pylustrator.start(...):` and that exiting the block calls `stop()` (safe no-op if `plt.show()` already restored hooks).

**Exit semantics:** always call `stop()` on `__exit__` (not “only if this call installed”). That matches Jupyter scoping: leaving the `with` block guarantees defaults are back. Double-`stop` after `pyl_show` is already a no-op.

## Docs / tests

- One short example in [`docs/styling.rst`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\docs\styling.rst) and a line in [`docs/index.rst`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\docs\index.rst) / [`CLAUDE.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\CLAUDE.md) core flow.
- Extend [`tests/test_hooks.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\tests\test_hooks.py): `with pylustrator.start():` asserts hooks inside; after the block, originals restored; exception inside the block still restores.
