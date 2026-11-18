---
name: Vector SVG PDF export
overview: Add Illustrator-friendly SVG (and keep PDF) export with editable text via matplotlib font settings. The posterior heatmap stays an embedded image; overlays and labels remain true vectors.
todos:
  - id: generalize-save-helper
    content: Refactor _save_publication_pdf into format-aware _save_publication_figure with svg.fonttype=none / pdf.fonttype=42
    status: completed
  - id: export-formats-api
    content: Add formats kwarg to export_for_publication; add Export SVG toolbar action
    status: completed
isProject: false
---

# Vector export for Illustrator (editable text)

## Constraint (important)

The posterior is silx `ImageData` → matplotlib `imshow`. That layer is **always an embedded raster** in PDF/SVG. Curves, polygons, tick labels, and `radon=` markers can be true vectors. Making every heatmap cell a vector rectangle is possible (`pcolormesh`) but produces huge files and still is not “editable text.”

So the practical Illustrator workflow is: **vector text + lines + shapes, raster heatmap**.

## What you already have

[`_save_publication_pdf`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py) calls `saveGraph(..., fileFormat='pdf')` inside:

```python
PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=True)
# includes: 'pdf.fonttype': 42, 'ps.fonttype': 42
```

`pdf.fonttype=42` embeds TrueType fonts so text stays editable in Illustrator (not Type-3). SVG is not exported yet; matplotlib’s default `svg.fonttype='path'` converts text to outlines (not editable as text).

## Implementation — [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py)

1. **Generalize save helper** → `_save_publication_figure(save_path, file_format: str)` supporting `'pdf'` and `'svg'`.
   - Reuse existing resize / replot / restore logic.
   - Prefer `backend.fig.savefig(...)` directly (same as silx `saveGraph`) so we can pass Illustrator-friendly kwargs:
     - PDF: `format='pdf'`, rely on `pdf.fonttype=42`
     - SVG: `format='svg'`, rc `svg.fonttype='none'` (keep text as `<text>`, editable in Illustrator)
   - Do **not** pass a high raster `dpi` as the driver of page size (already fixed via layout dpi). For vector formats, omit `dpi` or keep it only as a hint for the embedded heatmap.

2. **`export_for_publication(..., formats=('pdf', 'svg'))`**
   - Write both (or whichever requested) under the same suffix.
   - Return `{'pdf': Path, 'svg': Path}` as applicable.
   - Default: export **both** so Illustrator users open the `.svg`.

3. **Toolbar**
   - Keep “Export PDF”; add “Export SVG” (or one “Export vector…” that writes both).

4. **rc_context for vector**
   ```python
   PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=True) | {
       'svg.fonttype': 'none',   # editable text in SVG
       'pdf.fonttype': 42,       # already in helper; keep explicit
   }
   ```

## Usage after change

```python
_out = dbgr.export_for_publication(figures_parent_folder=..., export_suffix=..., formats=('svg', 'pdf'))
# Open the .svg in Illustrator: text/lines editable; heatmap is one linked/embedded image
```

## Out of scope

No attempt to vectorize the posterior heatmap cells. No notebook edits required beyond choosing the SVG path.