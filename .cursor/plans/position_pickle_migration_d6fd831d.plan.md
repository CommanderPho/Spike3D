---
name: Position pickle migration
overview: Add `__getstate__`/`__setstate__` to Position for backward-compatible unpickling of old `_data` attribute, and fix the existing broken migration code in Laps.
todos:
  - id: position-getstate-setstate
    content: Add __getstate__ / __setstate__ to Position class with _data -> _df migration
    status: completed
  - id: laps-fix-from-dict
    content: "Fix Laps.from_dict line 860: restore _data key in d.get()"
    status: completed
  - id: laps-fix-setstate
    content: "Fix Laps.__setstate__ lines 871-877: restore _data -> _df migration logic"
    status: completed
isProject: false
---

# Position `_data` to `_df` Pickle Migration

## Context

The `Position` class in [position.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py) had its internal `self._data` renamed to `self._df`. Old pickled `Position` objects store `_data` in their `__dict`__, so unpickling them with the new code will fail because the class now reads `self._df`.

No external code directly accesses `position._data` (confirmed via search). The `to_dict`/`from_dict` path uses the key `"df"` (not `"_data"`), so that serialization path is unaffected.

## 1. Add `__getstate_`_ / `__setstate__` to Position

Insert after the `to_dataframe` method (~line 1269), before `speed_in_epochs`:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    return state


def __setstate__(self, state):
    if '_df' not in state and '_data' in state:
        state['_df'] = state.pop('_data')
    self.__dict__.update(state)
```

- `__getstate__`: standard - emits current `__dict__` (which now contains `_df`).
- `__setstate__`: if the pickled state has `_data` but not `_df`, migrates the key. New pickles pass through unchanged.

## 2. Fix broken Laps migration code

In [laps.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\laps.py):

**Line 860** - `from_dict`: currently `d.get('_df', None) or d.get('_df', None)` (duplicate). Fix to:

```python
return Laps((d.get('_data', None) or d.get('_df', None)), metadata = d.get('metadata', None))
```

**Lines 874-876** - `__setstate__`: currently checks `if '_df' not in state` then asserts `'_df' in state` (contradictory). Fix to:

```python
def __setstate__(self, state):
    if '_df' not in state and '_data' in state:
        state['_df'] = state.pop('_data')
    self.__dict__.update(state)
```

