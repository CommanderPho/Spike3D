---
name: readable epoch labels
overview: Add an opt-in `epoch_label_mode` that draws Long/Short as readable annotations above the scatter (with halo), while keeping the current tiny vrect labels as the default.
todos:
  - id: epoch-shapes-mode
    content: Add epoch_label_mode to plotly_helper_add_epoch_shapes; implement annotation_above branch
    status: completed
  - id: wire-pre-post
    content: Thread epoch_label_mode through plotly_pre_post_delta_scatter and _helper_build_figure
    status: completed
isProject: false
---

# Readable Long/Short epoch labels (opt-in)

## Approach

Add `epoch_label_mode: str = 'vrect'` to [`plotly_helper_add_epoch_shapes`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) and thread it through callers. Default `'vrect'` preserves current behavior; `'annotation_above'` uses the improved style discussed earlier.

## Label behavior by mode

**`'vrect'` (default)** — unchanged:
- `add_vrect(..., label=dict(text=..., textposition="top center", font=...))` with `size≈6.67`

**`'annotation_above'`** — improved:
- `add_vrect` with **no** `label=` (shapes only, still `layer="below"`)
- `fig.add_annotation` per epoch, above the data:
  - `x` = midpoint of each epoch (`(t_start+t_split)/2`, `(t_split+t_end)/2`)
  - `y=1.0`, `yanchor="bottom"`, small `yshift` (~4–6 px), `cliponaxis=False` so text sits just above the y=1 edge without clipping
  - `showarrow=False`, `row`/`col` from existing `row_column_kwargs`
  - font default `size=12, family="Arial"` (overridable via existing `epoch_text_labels_kwargs`)
  - light halo: `bgcolor="rgba(255,255,255,0.75)"` (light) / `"rgba(0,0,0,0.55)"` (dark), small `borderpad`
  - named `long_epoch_label_{row}` / `short_epoch_label_{row}` for later clear/rebuild

## API / wiring

1. **`plotly_helper_add_epoch_shapes`** — add `epoch_label_mode: str = 'vrect'`. Assert mode in `{'vrect', 'annotation_above'}`. Branch only the label path; hlines/vlines/vrects fill stay shared.

2. **`plotly_pre_post_delta_scatter`** — add `epoch_label_mode: str = 'vrect'` to the signature; pass it (and any existing `epoch_text_labels_kwargs` from `kwargs.pop` if present) into the call at ~1087.

3. **`_helper_build_figure`** — `epoch_label_mode = build_fig_kwargs.pop('epoch_label_mode', 'vrect')` and pass into the call at ~1530.

No caller sites need to change for backward compatibility; paper/notebook code opts in with `epoch_label_mode='annotation_above'`.

## Out of scope

- No notebook edits
- No change to Pre-delta/Post-delta hist titles or shading colors
- No new legend
