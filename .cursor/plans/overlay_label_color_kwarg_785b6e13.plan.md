---
name: Overlay label color kwarg
overview: Add an `overlay_label_color` field on `RadonTransformDebugger` (default `'white'`) and use it for the score marker and rho/phi overlays that are currently hardcoded white.
todos:
  - id: add-overlay-label-color-field
    content: Add overlay_label_color field (default 'white') on RadonTransformDebugger
    status: completed
  - id: wire-overlay-colors
    content: Use overlay_label_color in add_score_label and add_rho_phi_overlay
    status: completed
isProject: false
---

# Add `overlay_label_color` to RadonTransformDebugger

## Scope

Only [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py). No notebook edits.

Apply the color to all white text/marker overlays used for contrast on the posterior (score + rho/phi). Leave the red trajectory line and orange scoring band unchanged.

## Changes

1. **Add field** next to the other optional display fields (near `posterior_heatmap_imshow_kwargs`):

```python
overlay_label_color: str = field(default='white')
```

2. **`add_score_label`** — replace hardcoded `color='white'` with `color=self.overlay_label_color`.

3. **`add_rho_phi_overlay`** — use `self.overlay_label_color` for:
   - the dashed rho curve (`color='#ffffff'` today)
   - both markers (`color='white'` today)

## Usage (notebook)

```python
RadonTransformDebugger(
    ...,
    posterior_heatmap_imshow_kwargs=dict(cmap='Greys', vmin=0.0),
    overlay_label_color='black',
)
```
