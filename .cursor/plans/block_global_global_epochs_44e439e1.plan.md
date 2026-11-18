---
name: Block GLOBAL_GLOBAL epochs
overview: Stop duplicate synthetic global epochs by making `EpochsAccessor.adding_global_epoch_row` return the dataframe unchanged when the requested `global_epoch_name` is already present, instead of suffixing `_GLOBAL` and appending a new row.
todos:
  - id: patch-epoch-accessor
    content: In neuropy/core/epoch.py, replace adding_global_epoch_row collision logic with early return when global_epoch_name already exists
    status: completed
isProject: false
---

# Prevent `*_GLOBAL_GLOBAL` epoch rows

## Cause

In [NeuroPy/neuropy/core/epoch.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py), `EpochsAccessor.adding_global_epoch_row` (starts ~line 1783) handles a label clash by setting `global_epoch_name = f"{global_epoch_name}_GLOBAL"`. When the caller uses the default `maze_GLOBAL` and that label already exists (e.g. from the paradigm or an earlier fixup), the new name becomes `maze_GLOBAL_GLOBAL` and a **second** row is added. The existing "DOUBLE-WARNING" branch only skips if the suffixed name is *also* already in the table, which is not true on the first bad call.

```mermaid
flowchart LR
  call[adding_global_epoch_row maze_GLOBAL]
  exists{maze_GLOBAL in labels?}
  suffix[Append _GLOBAL to name]
  add[Append new row]
  ok[Return unchanged]
  call --> exists
  exists -->|no| add
  exists -->|yes current bug| suffix --> add
  exists -->|yes fixed| ok
```

## Minimal code change (single place)

**File:** [NeuroPy/neuropy/core/epoch.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py)

**Edit:** Replace the collision block at lines 1802–1807 so that if `global_epoch_name` is already in `all_epoch_names`, the function **returns `self._obj` immediately** (idempotent no-op). Optionally log once at reduced noise (e.g. a single `print` noting skip, or no print if you prefer silence).

Concretely, remove:

- The `f"{global_epoch_name}_GLOBAL"` rename
- The misleading WARNING that prints the name *after* mangling
- The nested `DOUBLE-WARNING` check (no longer needed for this case)

**Behavior after change:**

- First call: adds the global row as today.
- Second call with the same `global_epoch_name`: returns the same dataframe; **no** `maze_GLOBAL_GLOBAL` row.

**`inplace`:** Early return should return `self._obj` without assigning, matching current early-exit behavior in the DOUBLE-WARNING branch.

**`Epoch.adding_global_epoch_row`** (~line 2096): No change required; it delegates to the accessor and already returns a copy when `inplace=False`.

## Out of scope (unless you want follow-up)

- [BaseDataSessionFormats.build_global_epoch_filter_config_dict](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\BaseDataSessionFormats.py) still renames `global_epoch_name` with a single `_GLOBAL` for **filter keys** only (does not append dataframe rows). That does not create `*_GLOBAL_GLOBAL` labels; leave as-is unless you want the same idempotent semantics there.

## Quick verification

After the edit, in a REPL or test: build a small `DataFrame` with a `maze_GLOBAL` row, register epochs accessor as usual, call `.epochs.adding_global_epoch_row(global_epoch_name='maze_GLOBAL')` twice; assert `'maze_GLOBAL_GLOBAL'` never appears in `label` and row count is unchanged on the second call.
