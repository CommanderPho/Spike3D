---
name: Template debugger kwarg
overview: Add `enable_directional_template_debugger` (default True) to `plot_full_paginated_decoded_epochs_window` so the Pho Directional Template Debugger can be skipped at launch.
todos: []
isProject: false
---

# Add `enable_directional_template_debugger` kwarg

## Change

In [`stacked_epoch_slices.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py) `PhoPaginatedMultiDecoderDecodedEpochsWindow.plot_full_paginated_decoded_epochs_window` (~L3465):

- Pop kwarg: `enable_directional_template_debugger: bool = kwargs.pop('enable_directional_template_debugger', True)` near the top of the body (before `params_kwargs` merge so it is not forwarded into `init_from_track_templates`).
- Gate the existing attach call (~L3546):

```python
if enable_directional_template_debugger:
    _out_directional_template_pfs_debugger, debug_update_paired_directional_template_pfs_debugger = _out_ripple_rasters.plot_attached_directional_templates_pf_debugger(curr_active_pipeline=curr_active_pipeline)
```

- Document in the method docstring Usage note: pass `enable_directional_template_debugger=False` to skip creating that window.

Default stays `True` so existing callers are unchanged.

## Usage

```python
(..., _) = PhoPaginatedMultiDecoderDecodedEpochsWindow.plot_full_paginated_decoded_epochs_window(
    ...,
    enable_directional_template_debugger=False,
)
```