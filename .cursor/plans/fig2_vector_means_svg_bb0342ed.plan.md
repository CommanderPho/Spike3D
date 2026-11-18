---
name: Fig2 vector means SVG
overview: Make Figure 2’s publication export path write `.svg` instead of matplotlib `.pdf` when callers (including the notebook) still pass `write_vector_format=True`, by remapping that flag inside `PaperFigureTwo.display()` before `perform_save`.
todos:
  - id: remap-vector-to-svg
    content: In PaperFigureTwo.display, remap write_vector_format=True to write_svg=True / write_vector_format=False before perform_save; update docstring
    status: completed
  - id: verify-notebook-kwargs
    content: Verify notebook-style fig2_kwargs (write_vector_format=True) produce SVG-only save kwargs, no PDF
    status: completed
isProject: false
---

# Figure 2: treat `write_vector_format` as SVG

## Why the previous change is not enough

The live notebook path is:

`fig2_kwargs` (`write_vector_format=True`) → [`AcrossSessionsVisualizations.across_sessions_bar_graphs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py) → [`PaperFigureTwo.display(**kwargs)`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) → `perform_save` → `AcrossSessionsVisualizations.output_figure` → `build_and_write_to_file`

`PaperFigureTwo.display` already defaults to SVG-only (`write_svg=True`, `write_png=False`, `write_vector_format=False`), but the notebook **overrides** with:

```python
fig2_kwargs = dict(..., write_vector_format=True, ...)
```

For matplotlib, `write_vector_format=True` still means **PDF** in [`write_to_file`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Mixins\ExportHelpers.py). So the publication cell still exports `.pdf` (and may also write `.svg` from the new default).

Do **not** change global `AcrossSessionsVisualizations.output_figure` defaults (`write_vector_format=True`) — that helper is shared by other across-session figures that still want PDF.

Do **not** edit the notebook (ask-before-notebooks rule); fix library behavior so the existing `fig2_kwargs` produces SVG.

## Change (Figure-2 local remapping)

**File:** [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) — `PaperFigureTwo.display`

Where `save_figure_kwargs` is built (~6997–6998), remap legacy vector intent to SVG before calling `perform_save`:

```python
# Notebook/publication callers pass write_vector_format=True for "vector export".
# For Figure 2 that now means SVG, not matplotlib PDF.
if write_vector_format:
    write_svg = True
    write_vector_format = False
save_figure_kwargs = dict(write_vector_format=write_vector_format, write_png=write_png, write_svg=write_svg) | pop_dict_subset(kwargs, ['bbox_inches', 'pad_inches'])
```

Update the `display` docstring to state that `write_vector_format=True` is accepted for back-compat and remapped to SVG-only for this figure.

Result for the notebook publication kwargs:

| Flag | Before | After remapping |
|------|--------|-----------------|
| `write_vector_format` | True → PDF | False (no PDF) |
| `write_svg` | True (display default) | True → `.svg` |
| `write_png` | False (display default; notebook omits it) | False |

`across_sessions_bar_graphs` needs no signature change; it already forwards `**fig2_kwargs` into `display`.

## Verification

- Static: with `write_vector_format=True` (notebook-style), `save_figure_kwargs` becomes `{write_vector_format=False, write_png=False, write_svg=True}`.
- Static: defaults with no override remain SVG-only.
- Optional: mock `perform_save` / `_pipeline_file_callback_fn` and call the remapping logic / display save path to assert callback kwargs match.