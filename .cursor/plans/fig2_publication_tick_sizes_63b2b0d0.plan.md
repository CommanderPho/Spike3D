---
name: Fig2 publication tick sizes
overview: When `prepare_for_publication=True`, apply `PhoPublicationFigureHelper`'s tick label sizes (`xtick`/`ytick.labelsize` = 5) on Figure 2 axes in `PaperFigureTwo.create_plot`, including the publication secondary x-axis.
todos:
  - id: fig2-tick-params
    content: In PaperFigureTwo.create_plot, when prepare_for_publication, apply PhoPublicationFigureHelper ytick.labelsize via ax.tick_params
    status: completed
  - id: fig2-secondary-xtick
    content: Pass labelsize into _apply_publication_grouped_xticklabels for primary and secondary x ticks
    status: completed
isProject: false
---

# Figure 2: use publication tick sizes

## Context

[`PhoPublicationFigureHelper.rc_context_kwargs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) already defines publication ticks:

```python
'xtick.labelsize': 5, 'ytick.labelsize': 5
```

Fig4-style helpers enter that `rc_context` (and often call `ax.tick_params(..., labelsize=...)`). [`PaperFigureTwo.create_plot`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) does neither, so Fig 2 keeps session/global default tick sizes even with `prepare_for_publication=True`.

SVG/PDF export does not set tick size; it must be set on the axes at plot time.

## Change

**File:** [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) — `PaperFigureTwo` only.

1. **`create_plot`** — after spines / tick labels are finalized (right after the `prepare_for_publication` call to `_apply_publication_grouped_xticklabels`, ~6804), when `prepare_for_publication` is True:

```python
tick_fs = float(PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=True)['ytick.labelsize'])
ax.tick_params(axis='both', labelsize=tick_fs)
```

Use that helper as the single source of truth (currently `5`), not a hard-coded magic number elsewhere.

2. **`_apply_publication_grouped_xticklabels`** — add optional `labelsize: Optional[float] = None`. When set, pass it through:

- `ax.tick_params(axis='x', length=0, pad=2, labelsize=labelsize)`
- `sec.tick_params('x', length=0, labelsize=labelsize)`

From `create_plot`, pass `labelsize=tick_fs` into this helper so pre-/post-Δ and LdCs/SdCs group labels match the y ticks.

3. Do **not** wrap the whole `create_plot` in full publication `rc_context` (that would also force Arial / spine linewidth globally for this path). Scope is tick label sizes only, as requested.

Non-publication mode (`prepare_for_publication=False`) unchanged.

## Verification

Re-run the notebook Fig 2 cell with `prepare_for_publication=True` and confirm y-tick and publication x-tick labels render at size 5 (matching other publication figures), on screen and in SVG.