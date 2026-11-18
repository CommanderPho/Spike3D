---
name: Fix KDiba matching fallback
overview: KDiba’s `_get_session_specific_parameters` now returns a value correctly, but strict `IdentifyingContext.matching` finds no entries for the format-only fallback when criteria includes animal/exper/session. Add the same `find_best_matching_context` fallback used by NWB/DANDI.
todos:
  - id: kdiba-matching-fallback
    content: Add find_best_matching_context fallback in KDiba `_get_session_specific_parameters` when strict matching is empty
    status: completed
isProject: false
---

# Fix KDiba empty-match IndexError

## Root cause

[`IdentifyingContext.matching`](file:///home/halechr/repos/NeuroPy/neuropy/utils/result_context.py) uses `ctxt.query(criteria)`, which requires the **dict key** to have every criteria attribute. KDiba’s only entry is:

```python
IdentifyingContext(format_name='kdiba')
```

Criteria is `{format_name, animal, exper_name, session_name}` from a real/filtered session context, so the fallback key fails `query` → empty dict → `list(best_match.values())[0]` raises `IndexError`.

(The earlier AttributeError was the same empty/`dict` mismatch; extracting `[0]` exposed the empty match.)

NWB and DANDI already handle this:

```python
best_match = IdentifyingContext.matching(...)
if len(list(best_match.values())) > 0:
    return list(best_match.values())[0]
best_match, _ = IdentifyingContext.find_best_matching_context(...)
return the_dict[best_match]
```

`find_best_matching_context` allows underconstrained keys (format-only fallback matches a fuller session context).

## Change

In [`KDibaOldDataSessionFormat.py`](file:///home/halechr/repos/NeuroPy/neuropy/core/session/Formats/Specific/KDibaOldDataSessionFormat.py) `_get_session_specific_parameters`, replace the bare `return list(best_match.values())[0]` with the NWB/DANDI fallback pattern above (same criteria subset: `cls._session_basepath_to_context_parsing_keys`).

No notebook edits. Re-run the post-fixup cell; expect `get_spatial_dimensionality` to resolve via the kdiba fallback (`spatial_dimensionality is None` → `sess.position.ndim`) and placefield recompute to continue.
