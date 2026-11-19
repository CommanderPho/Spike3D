---
name: Epoch getitem column error
overview: Harden `Epoch.__getitem__` so DataFrame-style column access on an Epoch (empty label match + key(s) are real `_df` columns) raises a clear `IndexError` pointing callers at `ensure_dataframe(...)`, while leaving every successful named-epoch label lookup bit-identical.
todos:
  - id: update-getitem
    content: "Update Epoch.__getitem__ in neuropy/core/epoch.py: empty label match + key(s) in _df.columns → IndexError advising ensure_dataframe(...); leave non-empty label matches on the exact existing return path"
    status: completed
  - id: verify-named-lookups
    content: "Verify named-epoch lookup is untouched: gate raise only after empty match; run NeuroPy tests/test_shapely_valid_epochs.py; sanity-check maze1/maze2/get_named_timerange paths still use non-empty label matches"
    status: completed
isProject: false
---

# Harden Epoch.__getitem__ empty column-name footgun

## Problem

In [`neuropy/core/epoch.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py), `Epoch.__getitem__` treats a string as a **named epoch label** lookup (`epochs['maze1']`), not a DataFrame column. `epochs['label']` therefore searches for an epoch named `"label"`, finds none, and returns an empty array — silently wrong for callers that meant the `label` column.

List-of-strings already raises, but the message still recommends `.epochs.to_dataframe()` rather than `ensure_dataframe(...)`.

## Non-negotiable: preserve named-epoch lookup

The new error must run **only after** a label match fails. Any non-empty `self.labels == key` match must take the **exact existing return path** (same `np.vstack` / squeeze logic as today). Do not reorder, wrap, or reinterpret successful named lookups.

Existing named-label call sites that must keep working (non-empty label matches today):

- `x.epochs['maze1']` / `x.epochs['maze2']` in [`KDibaOldDataSessionFormat.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\Specific\KDibaOldDataSessionFormat.py)
- `Epoch.get_named_timerange(epoch_name)` → `self[epoch_name]` in [`epoch.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py)
- `valid_epochs['maze1']` / `['maze2']` in [`tests/test_shapely_valid_epochs.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\tests\test_shapely_valid_epochs.py)
- `sess.paradigm['maze']` / `['maze1']` style paradigm lookups

Also safe / unrelated: `self.epochs["start"]` / `["label"]` goes through the `.epochs` DataFrame accessor, **not** `Epoch.__getitem__`.

Edge case already preserved by the empty-match gate: if an epoch is literally labeled `"label"` / `"start"` / `"stop"`, named lookup wins and no error is raised.

## Change (single function)

Update only `Epoch.__getitem__` (~lines 2067–2095) in [`neuropy/core/epoch.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py):

```mermaid
flowchart TD
  key[key: str or list of str]
  labelMatch[match against self.labels]
  empty{empty match?}
  isCol{key(s) in self._df.columns?}
  raiseErr["raise IndexError: likely DataFrame column access; wrap with ensure_dataframe(...)"]
  namedOk[return start/stop for matched labels UNCHANGED]
  keepEmpty[keep current empty return for unknown non-column str]
  listOther[keep existing list-of-str IndexError if not columns]

  key --> labelMatch --> empty
  empty -->|no| namedOk
  empty -->|yes| isCol
  isCol -->|yes| raiseErr
  isCol -->|no, str| keepEmpty
  isCol -->|no, list| listOther
```

Concrete behavior:

1. **`str` key**
   - Compute `indices = np.where(self.labels == slice_)[0]` as today.
   - If `len(indices) >= 1`: return start/stop **exactly as today** (named-epoch lookup unchanged).
   - If `len(indices) == 0` **and** `slice_ in self._df.columns`: raise `IndexError` explaining likely DataFrame column access on an `Epoch`, advise `ensure_dataframe(epochs)['{slice_}']`.
   - If empty match but **not** a column name: keep returning the empty squeezed array.

2. **list/tuple of strings** (existing second branch)
   - Treat “empty match” as: none of the requested strings appear in `self.labels`.
   - If empty match **and** all requested strings are in `self._df.columns`: raise the same style of `IndexError` with `ensure_dataframe(epochs)[...]` advice (covers `epochs[['label']]` / `epochs[['start','stop']]`).
   - If any requested string matches a label: do not take the new column-error path (preserve any label-list semantics / existing raise only when still empty-and-not-columns / existing list raise).
   - Otherwise: keep raising `IndexError` (update advice text to prefer `ensure_dataframe(...)`).

Do **not** auto-return columns; only fail loudly. Do not change `.labels`, `ensure_dataframe`, or call sites in `data_exporting.py`.

## Verification

- Run `uv run pytest tests/test_shapely_valid_epochs.py` from the NeuroPy package root (covers `valid_epochs['maze1']` / `['maze2']`).
- Mentally/code-path check that `get_named_timerange('maze1')` and `sess.epochs['maze1']` still hit the non-empty branch before any column check.

## Out of scope

- No new dedicated unit tests unless you ask later.
- No call-site sweeps.