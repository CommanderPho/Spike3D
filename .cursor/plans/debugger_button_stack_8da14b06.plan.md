---
name: Debugger button stack
overview: Remove n=1 and n≈E from InteractiveBayesian2DEquationDebugger; keep Export PNG and n=0 in a right-aligned vertical action-button stack that clears the sliders.
todos:
  - id: remove-quickset
    content: Remove n=1 and n≈E buttons/handlers; keep n=0
    status: completed
  - id: vertical-stack
    content: Place Export PNG then n=0 in right-aligned vertical action stack; shorten slider width to avoid overlap; keep self.buttons wired for export hide
    status: completed
isProject: false
---

# Right-aligned vertical action button stack

## Target

[`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) — `InteractiveBayesian2DEquationDebugger.buildUI` (~2125–2142).

## Changes

### 1. Drop only n=1 and n≈E

Remove the `n=1` and `n≈E` axes, `Button`s, and `on_clicked` handlers. **Keep `n=0`** (calls `self.set_all(np.zeros(n_cells))`). Keep `set_all(...)` as the shared helper.

### 2. Right-aligned vertical action stack

Replace the horizontal button row with a vertical stack on the far right of the controls band:

- Column geometry: `btn_left=0.88`, `btn_w=0.10`, `btn_h=0.04`, `btn_gap=0.006`
- Top of stack: `y0 = controls_bottom + n_cells * slider_pitch - btn_h`
- Stack grows downward: `y_i = y0 - i * (btn_h + btn_gap)`, clamped `>= controls_bottom`
- Order (top → bottom): **Export PNG**, then **n=0**

Shorten slider tracks so valtext does not collide with the button column: use `[0.17, y_s, 0.55, slider_h]` (ends ~0.72; buttons start at 0.88).

### 3. Wire `self.buttons`

```python
action_specs = [
    ('Export PNG', lambda _e: self.export_to_png()),
    ('n=0', lambda _e: self.set_all(np.zeros(n_cells))),
]
# place each in the vertical stack...
self.buttons = tuple(created_buttons)
```

Loop over `action_specs` so later buttons are one more `(label, callback)` entry.

### 4. No other API changes

`fig._bayes_eqn_ui['export_to_png']` and `export_to_png(...)` stay as-is. `_get_export_control_axes` already iterates `self.buttons`, so both stack buttons are hidden during PNG export.

## Out of scope

- Restoring n=1 / n≈E
- Changing reliability radio / checkbox layout
- PNG export logic beyond control-axis listing
