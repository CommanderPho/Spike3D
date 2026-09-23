---
name: ClassVar template_dict
overview: Make `PlotlyHelpers.template_dict` a true attrs-ignored class attribute via `ClassVar`, matching how `get_plotly_template` already reads it on `cls`.
todos:
  - id: add-classvar
    content: Add ClassVar import and annotate template_dict as ClassVar on PlotlyHelpers
    status: completed
  - id: cleanup-classmethod
    content: Remove redundant None-init in get_plotly_template; keep template registration loop
    status: completed
  - id: verify
    content: Spot-check attrs fields exclude template_dict; get_plotly_template still works
    status: in_progress
isProject: false
---

# Fix PlotlyHelpers.template_dict as ClassVar

## Context

In [`plotly_templates.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\plotly_templates.py), `PlotlyHelpers` is `@define(slots=False)`. Line 206 is currently typed like an attrs field:

```python
template_dict: Dict[str, Dict] = deepcopy(_template_dict) ## NOT A INSTANCE PROPERTY, a class property
```

Attrs treats annotated attributes as instance fields, so this is not a class property. Call sites already expect class-level access:

```317:321:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\plotly_templates.py
        if getattr(cls, 'template_dict', None) is None:
            cls.template_dict = deepcopy(_template_dict)

        for k, v in cls.template_dict.items():
            pio.templates[k] = go.layout.Template(**v)
```

Grep shows `_template_dict` / `template_dict` only appear in this file.

## Approach (chosen)

Use `typing.ClassVar` so attrs skips the attribute and it remains shared on the class. One module-time `deepcopy(_template_dict)` is enough for a shared template registry.

## Edits (single file)

File: [`pyPhoPlaceCellAnalysis/.../plotly_templates.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\plotly_templates.py)

1. Add `ClassVar` to the existing `typing` import.
2. Replace the field with:

```python
template_dict: ClassVar[Dict[str, Dict]] = deepcopy(_template_dict)  ## NOT AN INSTANCE PROPERTY, a class property
```

3. In `get_plotly_template`, remove the dead `getattr(..., None) is None` re-init (ClassVar is always present after class body runs). Keep the `for k, v in cls.template_dict.items():` loop unchanged.
4. Leave the commented `__attrs_post_init__` block as-is (user comments).

## Verification

- Confirm `PlotlyHelpers` constructor / attrs fields no longer include `template_dict` (e.g. inspect `attrs.fields(PlotlyHelpers)` or instantiate with existing required args only).
- Confirm `PlotlyHelpers.template_dict` and `PlotlyHelpers.get_plotly_template(is_dark_mode=False)` still register `pho_diba_publication` on `pio.templates`.

No notebook changes. No other packages.