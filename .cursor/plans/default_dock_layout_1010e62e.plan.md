---
name: Default dock layout
overview: "Make YellowBlue dock a peer of the DecodedEpochSlices row (not full-window height) and attach it before the rasters top dock, so startup matches your manual two-tier layout: 4 rasters on top, 5 equal-height docks below."
todos:
  - id: yb-relative-dock
    content: Change YellowBlue embed to dockAddLocationOpts=['right', find_display_dock('short_RL')]
    status: completed
  - id: reorder-attach
    content: In plot_full_paginated_decoded_epochs_window, attach YellowBlue before rasters; preserve return tuple order
    status: completed
  - id: verify-layout
    content: Relaunch window and confirm 4-on-top / 5-on-bottom matches screenshot 2
    status: completed
isProject: false
---

# Default two-tier dock layout on startup

## Problem

Startup currently looks like screenshot 1 because YellowBlue is added with absolute `['right']`, which claims the **full right edge** of the window (beside both the rasters strip and the decoded columns):

```
[==== RankOrderRastersDebugger (4 nested rasters) ====][YellowBlue]
[Decoded][Decoded][Decoded][Decoded]                   [YellowBlue]
[==================== Utility ========================]
```

Screenshot 2 is the two-tier layout you want: YellowBlue is a **5th peer** in the decoded row; rasters sit above that whole row:

```
[=========== 4 rasters (top strip) ===================]
[Decoded][Decoded][Decoded][Decoded][YellowBlue]
[==================== Utility ========================]
```

## Cause

In [`stacked_epoch_slices.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py):

1. `build_attached_yellow_blue_track_identity_marginal_window` (~L3289) uses `dockAddLocationOpts=['right']` with no relative dock → full-height right edge.
2. `plot_full_paginated_decoded_epochs_window` (~L3516–3532) attaches **rasters before YellowBlue**, so even a relative YellowBlue add can leave the top strip sized only over the original 4 columns.

Existing API already supports relative placement (`['right', some_dock]`), same pattern as [`RankOrderRastersDebugger`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py) / `add_display_dock` in `DynamicDockDisplayAreaContent.py`.

## Minimal changes (one file)

Only edit [`stacked_epoch_slices.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py).

### 1. Relative YellowBlue placement (~L3284–3290)

When embedding YellowBlue, place it to the right of the last decoded dock (`short_RL`):

```python
relative_dock = self.find_display_dock('short_RL')
assert relative_dock is not None
...
dockAddLocationOpts=['right', relative_dock]
```

Keep dock config / size / identifier as-is.

### 2. Attach YellowBlue before rasters (~L3515–3532)

In `plot_full_paginated_decoded_epochs_window`, swap the two attach blocks so that:

1. Build/attach YellowBlue (joins the decoded horizontal row → 5 peers)
2. Build/attach rasters with existing `['top']` (spans above that full 5-column row)
3. Keep `plot_attached_directional_templates_pf_debugger` after rasters exist

Return tuple order stays the same: `(main_tuple, ripple_rasters_plot_tuple, yellow_blue_trackID_marginals_plot_tuple)`.

## Out of scope

- No change to nested rasters-inside-one-dock vs transferring four raster docks (comment at L3148); screenshot 2 is achievable without that.
- No changes to Utility footer, decoder dock creation, or dock colors.

## Verification

Relaunch via the usual `plot_full_paginated_decoded_epochs_window` path and confirm:

- Top: one strip with the four raster docks
- Bottom: five equal-height docks (4× DecodedEpochSlices + YellowBlue)
- Utility still full-width at bottom
- Paging / `<controlled>` footers still work on the decoded + YellowBlue docks
