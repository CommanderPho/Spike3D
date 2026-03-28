---
name: Fix vispy title patch
overview: "Fix `_run_vispy_example.py` Canvas/SceneCanvas monkeypatches so forced example titles never combine a leftover positional “title slot” with a keyword `title`, which causes `TypeError: __init__() got multiple values for argument 'title'` for examples that route through `SceneCanvas` → `Canvas` with all-positional super args."
todos:
  - id: patch-canvas-init
    content: "Update _patched_canvas_init in _run_vispy_example.py: pop title from kwargs; if args then (example_name,) + args[1:] else kwargs[\"title\"] = example_name"
    status: completed
  - id: patch-scene-init
    content: Apply the same positional-vs-keyword title replacement logic to _patched_scene_canvas_init
    status: completed
  - id: verify-examples
    content: Run line_draw, plot_grids, quiver_arrows via the browser/subprocess and confirm no TypeError
    status: completed
isProject: false
---

# Fix duplicate `title` in vispy example launcher

## Root cause

In `[Spike3D/LibrariesExamples/vispy/_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)`, `_patched_canvas_init` does:

1. `kwargs["title"] = example_name`
2. If `args` is non-empty, `args = args[1:]` (strip old title)
3. `_orig_canvas_init(self, *args, **kwargs)`

`[SceneCanvas.__init](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\vispy\scene\canvas.py)__` calls `super(...).__init__(title, size, position, show, ...)` with **many positionals** (see lines 135–138). After step 2, `args[0]`, is `size`, not `title`, but `kwargs` still contains `title=example_name`. Python then binds the first positional to `title` **and** applies `title=` from kwargs → **multiple values for argument `title`**.

The same pattern in `_patched_scene_canvas_init` is inconsistent for call sites that pass `title` as the first positional to `SceneCanvas`.

Failing examples you saw (`line_draw`, `plot_grids` via `Fig`, `quiver_arrows`) all hit this `Canvas` path after `SceneCanvas` uses the all-positional `super()` form.

## Fix (minimal)

In **both** `_patched_canvas_init` and `_patched_scene_canvas_init`:

1. `kwargs = dict(kwargs)` then `kwargs.pop("title", None)` so no stale keyword title remains.
2. If `args` is non-empty, replace the first positional with the example name: `args = (example_name,) + args[1:]`.
3. Else (keyword-only construction), set `kwargs["title"] = example_name`.
4. Call the original `__init__` with the adjusted `*args` and `**kwargs`.

No changes to `[vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)` or example scripts are required unless you want regression tests.

## Verification

After the edit, re-run the previously failing names from the console:

- `basics/visuals/line_draw`
- `pho_custom/plot_grids`
- `pho_custom/quiver_arrows`

Confirm exit code 0 and window title still reflects the example name (behavior preserved).