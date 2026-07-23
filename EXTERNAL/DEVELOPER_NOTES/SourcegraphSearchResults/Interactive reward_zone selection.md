---
title: Interactive reward_zone selection in pyPhoPlaceCellAnalysis
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/200ea8cf-b97c-4234-9e3f-26ef85a4386c
url: https://phohale.sourcegraph.app/deepsearch/365
created: 2026-05-06T10:34:38Z
updated: 2026-05-06T10:37:14Z
---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoPlaceCellAnalysis Find interactive reward_zone selection functions

The term `reward_zone` does not exist anywhere in either `github.com/CommanderPho/Spike3D` or `github.com/CommanderPho/pyPhoPlaceCellAnalysis`. Here is a summary of what was searched and what the closest related concepts are:

**Searched terms with no results:**
- `reward_zone`
- `reward_well`
- `reward_location`
- `reward_site`

**Closest related concepts that do exist:**

1. **Track endcap/platform analysis** - The codebase uses "endcaps" (i.e., the platform regions at the ends of the linear track, which correspond to reward locations) rather than "reward_zone":
   - [`LongShortTrackComputations.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py) - `_perform_long_short_endcap_analysis` computes which cells are localized to endcaps
   - [`track_shape_drawing.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py?L267-L270) - `platform_start_x` / `platform_stop_x` named positions define the platform (reward) regions geometrically
   - [`NotableTrackPositions`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py?L58-L65) holds the outer/inner x-positions of the platforms

2. **Interactive epoch/region selection** (the closest interactive selection mechanism):
   - [`stacked_epoch_slices.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py?L3480-L3498) - `interactive_good_epoch_selections()` lets users interactively select good epochs
   - [`EpochsEditorItem.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py?L589-L592) - `on_epoch_region_selection_toggled()` handles interactive region selection in PyQtPlot

Could you clarify what you mean by "reward_zone"? It's possible the feature is named differently in the codebase (e.g., "endcap", "platform"), or it may not have been implemented yet.

## Sources

- [repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ interactive reward_zone selection](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20interactive%20reward_zone%20selection)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) reward](/search?q=patterntype:keyword%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20reward)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) reward zone track region selection interactive](/search?q=patterntype:nls%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20reward%20zone%20track%20region%20selection%20interactive)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) reward well track position interactive select](/search?q=patterntype:nls%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20reward%20well%20track%20position%20interactive%20select)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) reward_well OR reward_location OR reward_site OR endcap interactive](/search?q=patterntype:keyword%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20reward_well%20OR%20reward_location%20OR%20reward_site%20OR%20endcap%20interactive)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) platform_start OR platform_stop OR track_endcap interactive](/search?q=patterntype:keyword%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20platform_start%20OR%20platform_stop%20OR%20track_endcap%20interactive)


---

# functions for interactively selecting the grid_bin_bounds in 2D

Here is a full breakdown of the interactive `grid_bin_bounds` selection functions. They all live in [`NeuroPy/neuropy/utils/matplotlib_helpers.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py) (accessed via the `Spike3DWorkEnv` submodule mirror of `NeuroPy`):

---

## Functions

### 1. `add_rectangular_selector` — low-level primitive
[Lines 1625–1695](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py?L1625-L1695)

Attaches a matplotlib `RectangleSelector` to any existing `fig`/`ax`. This is the building block used by everything else.

```python
from neuropy.utils.matplotlib_helpers import add_rectangular_selector

fig, ax = curr_active_pipeline.computation_results['maze'].computed_data.pf2D.plot_occupancy()
rect_selector, set_extents, reset_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds)
```

- `initial_selection` accepts either extents format `(xmin, xmax, ymin, ymax)` or pairs format `((xmin, xmax), (ymin, ymax))`
- Returns three objects:
  - `rect_selector` — the `RectangleSelector`; read the selection via `rect_selector.extents`
  - `set_extents(selection)` — programmatically update the selection
  - `reset_extents()` — restore to the initial selection
- Configured with `useblit=True`, `interactive=True`, `ignore_event_outside=True`, left+right mouse buttons

---

### 2. `interactive_select_grid_bin_bounds_2D` — high-level, pipeline-integrated
[Lines 1727–1799](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py?L1727-L1799)

Wraps `add_rectangular_selector` over a `pf2D.plot_occupancy()` figure, pre-seeded with the current `grid_bin_bounds` from the pipeline's computation config.

```python
from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D

# Non-blocking (returns handles for further use):
fig, ax, rect_selector, set_extents, reset_extents = interactive_select_grid_bin_bounds_2D(
    curr_active_pipeline, epoch_name='maze', should_block_for_input=False
)

# Blocking (waits for [Enter] keypress, then returns confirmed extents):
grid_bin_bounds = interactive_select_grid_bin_bounds_2D(
    curr_active_pipeline, epoch_name='maze',
    should_block_for_input=True,
    should_apply_updates_to_pipeline=False  # set True to write back to all filtered epochs
)
print(f'grid_bin_bounds: {grid_bin_bounds}')
```

**Key behaviors:**
- **`should_block_for_input=True`** — loops on `plt.waitforbuttonpress()` until any key is pressed. On keypress, confirms `rect_selector.extents` and closes the figure. Press **`r`** at any time to reset the rectangle to the initial bounds.
- **`should_apply_updates_to_pipeline=True`** — after confirmation, writes the new `grid_bin_bounds` back to `computation_result.computation_config['pf_params'].grid_bin_bounds` for **every** filtered epoch in the pipeline.
- Also prints a ready-to-paste `specific_session_override_dict` snippet with the confirmed bounds.

---

### 3. `interactive_select_grid_bin_bounds_1D` — 1D variant
[Lines 1700–1724](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py?L1700-L1724)

Same concept but for the 1D occupancy plot, using a horizontal `SpanSelector` (`add_range_selector`) instead of a rectangle.

```python
from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_1D

fig, ax, range_selector, set_extents = interactive_select_grid_bin_bounds_1D(
    curr_active_pipeline, epoch_name='maze'
)
```

---

## Where they're used / called from

- **[`PhoDiba2023Paper.py` L1896–1900](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py?L1896-L1900)** — commented-out usage showing both blocking and non-blocking call patterns
- **[`PendingNotebookCode.py` (Spike3DWorkEnv)](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/Spike3D/PendingNotebookCode.py?L377-L382)** — bare `add_rectangular_selector` usage directly on `pf2D.plot_occupancy()`
- **[`neuropy/analyses/placefields.py` L917–919](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L916-L919)** and **[`neuropy/core/user_annotations.py` L521–523](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/core/user_annotations.py?L521-L523)** — commented references showing where it was intended to hook into automated workflows

## Sources

- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) grid_bin_bounds interactive](/search?q=patterntype:keyword%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20grid_bin_bounds%20interactive)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ grid_bin_bounds select interactive widget](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20grid_bin_bounds%20select%20interactive%20widget)
- [(repo:^github.com/CommanderPho/Spike3D$ OR repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$) interactive_select_grid_bin_bounds_2D](/search?q=patterntype:keyword%20%28repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20OR%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%29%20interactive_select_grid_bin_bounds_2D)
- [interactive_select_grid_bin_bounds_2D](/search?q=patterntype:keyword%20interactive_select_grid_bin_bounds_2D)
- [github.com/CommanderPho/Spike3DWorkEnv/NeuroPy/neuropy/utils/matplotlib_helpers.py@eadd004?L1680-1820](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py?L1680-1820)
- [repo:^github.com/CommanderPho/Spike3DWorkEnv$ add_rectangular_selector matplotlib_helpers](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3DWorkEnv$%20add_rectangular_selector%20matplotlib_helpers)
- [github.com/CommanderPho/Spike3DWorkEnv/NeuroPy/neuropy/utils/matplotlib_helpers.py@eadd004?L1624-1696](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/NeuroPy/neuropy/utils/matplotlib_helpers.py?L1624-1696)

