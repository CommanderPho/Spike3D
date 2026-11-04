---
name: Group text by size
overview: Finish the TextHelpers class refactor (already started for set_text_font_size_centered), add group_text_by_font_size as a classmethod, keep thin module-level aliases, and update exports/docs.
todos:
  - id: texthelpers-class
    content: "Polish TextHelpers in text.py: blank line after imports; add group_text_by_font_size classmethod; add module-level aliases matching Layers pattern"
    status: completed
  - id: export-docs
    content: Export TextHelpers + aliases from __init__.py; update README for TextHelpers.group_text_by_font_size / set_text_font_size_centered
    status: completed
isProject: false
---

# TextHelpers + `group_text_by_font_size`

## Current state

[`text.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\text.py) already has `class TextHelpers` with `@classmethod set_text_font_size_centered`. [`__init__.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\__init__.py) still imports the old top-level name and will break until updated.

## Pattern (match [`layers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\layers.py))

- Implement logic as `TextHelpers` classmethods (two blank lines between methods; blank line after imports before `class`).
- Keep thin module-level wrappers that delegate, so existing call sites / README examples keep working:

```python
def set_text_font_size_centered(live, ids, size_pt):
    return TextHelpers.set_text_font_size_centered(live, ids, size_pt)


def group_text_by_font_size(live, *, limit=10000, effective=True):
    return TextHelpers.group_text_by_font_size(live, limit=limit, effective=effective)
```

## New method

```python
@classmethod
def group_text_by_font_size(cls, live: LiveVectorCraft, *, limit: int = 10000, effective: bool = True) -> Dict[str, Any]:
```

Behavior (from the notebook cell):

1. `document.find {kind: Type, limit}` → matches
2. Per match: `text.getRange {id, effective}` → plain `text` + style runs
3. Unique sizes from `run['style']['size']`
4. Row: `{id, name, text, sizes, runs}`
5. Single-size → `by_size[size]`; multi-size → `mixed` **and** each size bucket

Return: `{'by_size': Dict[float, List[Dict]], 'mixed': List[Dict], 'total': int}`

No printing inside the helper.

## Wiring

- [`__init__.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\__init__.py): `from .text import TextHelpers, group_text_by_font_size, set_text_font_size_centered`; add all three to `__all__`
- [`README.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\README.md): document both helpers (class or module alias); example for group that prints `id` + `text` per size

## Out of scope

- Notebook edit (ask-before-modifying-jupyter)
