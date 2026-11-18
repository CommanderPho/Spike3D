---
name: Revert overlay alpha
overview: Revert OverlayLabels inset layout from angled corner back to flush top-right stacking with lower alpha, axes-only anchoring, and clip_on=False so radon/wcorr/heuristic stay stacked and fully visible.
todos:
  - id: restore-inset-helper
    content: Switch inset path back to adjacent_right_lots_of_text; keep angled commented
    status: completed
  - id: keep-low-alpha
    content: Keep text_alpha/stroke_alpha/artist alpha at translucent values
    status: completed
  - id: simple-axes-stack
    content: Restore (1.0, y) axes stacking; drop angled offset bbox math
    status: completed
  - id: disable-clip
    content: set_clip_on(False) on overlay AnchoredText artists after upsert
    status: completed
isProject: false
---

# Revert angled overlays; keep low alpha, no cutoff

## Goal

Undo `_helper_build_text_kwargs_angled_upper_right_corner` (floating/missing radon, clipped `0.` text). Restore flush top-right stacking with **lower opacity** so heatmaps stay readable and nothing is clipped.

## Changes (locked) — [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py) OverlayLabels callback only

### 1. Restore inset helper

In the inset branch (~2479–2485):

- Use `_helper_build_text_kwargs_adjacent_right_lots_of_text` again.
- Leave angled helper commented out.

That helper already uses axes-only `bbox_to_anchor=(1.0, 1.0)` / `bbox_transform=ax.transAxes` (no `fig.transFigure`, no rotation).

### 2. Keep lower alpha

Keep (or set) after building `text_kwargs`:

- `text_alpha=0.75`, `stroke_alpha=0.55`
- `anchored_text_alpha_override_value=0.4`

Do **not** restore opaque `1.0` artist alpha.

### 3. Simple axes stack again (no angled offset math)

Replace the angled `anchor_x` / `anchor_top_y` / relative-drop bbox math (~2585–2619) with the previous flush stack:

- `wcorr` → `(1.0, 1.0)`
- `heuristic` → `(1.0, heuristic_y)` from `_overlay_stack_y_below`
- `radon` → `(1.0, radon_y)`
- `set_bbox_to_anchor(..., transform=curr_ax.transAxes)` for all three

### 4. Prevent cutoff

In `_subfn_upsert_overlay_label`, after create/update, call `extant_artist.set_clip_on(False)` (and `extant_artist.txt.set_clip_on(False)` when present) so multi-line corner text is not clipped by the axes patch.

## Verification

Refresh / re-`add_data_overlays(included_columns=['radon','wcorr','coverage','mseq_tcov'])`. Expect: three stacked labels top-right, semi-transparent, radon under heuristic on every row, full strings (no `0.` stubs).