---
name: Fix keypath override
overview: The Compute New crash happens because the pickle-selector widget passes short parameter names (`minimum_inclusion_fr_Hz`) into `update_parameters`, which expects dotted `ComputationKWargParameters` keypaths (`rank_order_shuffle_analysis.minimum_inclusion_fr_Hz`).
todos:
  - id: map-widget-keypaths
    content: Add current_override_parameters_flat_keypaths_dict mapping widget short names to rank_order_shuffle_analysis.* keypaths
    status: completed
  - id: wire-compute-callbacks
    content: Use the mapped dict in _subfn_compute_new and _subfn_compute instead of current_parameter_values
    status: completed
isProject: false
---

# Fix widget override keypaths for Compute New

## Root cause

`PipelinePickleFileSelectorWidget.current_parameter_values` returns short names:

```620:623:/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/IPyWidgets/pipeline_ipywidgets.py
        return {
            'minimum_inclusion_fr_Hz': self.minimum_fr_widget.value,
            'included_qclu_values': qclu_values
        }
```

`_subfn_compute_new` and `_subfn_compute` pass that dict as `override_parameters_flat_keypaths_dict` into `batch_load_session` → `ComputedPipelineStage.update_parameters` → `ComputationKWargParameters.set_by_keypath(k, v)`.

`ComputationKWargParameters` has nested groups only (`rank_order_shuffle_analysis`, `wcorr_shuffle_analysis`, …). The firing-rate / qclu fields live on [`rank_order_shuffle_analysis_Parameters`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/SpecificComputationParameterTypes.py), so `hasattr(ComputationKWargParameters, 'minimum_inclusion_fr_Hz')` is false and [`set_by_keypath`](NeuroPy/neuropy/utils/mixins/gettable_mixin.py) raises.

Working callers already use dotted paths, e.g. in [`ProcessBatchOutputs_qclus1246789_Only.ipy`](Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy):

```python
override_parameters_flat_keypaths_dict = {
    'rank_order_shuffle_analysis.included_qclu_values': included_qclu_values,
    'rank_order_shuffle_analysis.minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
    ...
}
```

That matches the documented `update_parameters` example in [`Computation.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py).

```mermaid
flowchart LR
  widget["widget.current_parameter_values"]
  batch["batch_load_session"]
  update["update_parameters"]
  setkp["set_by_keypath"]
  widget -->|"short names"| batch --> update --> setkp
  setkp -->|"minimum_inclusion_fr_Hz missing on ComputationKWargParameters"| err["AttributeError"]
```

## Fix

Do **not** change `set_by_keypath` to auto-resolve leaf names (ambiguous if two groups share a field). Fix the widget, which is the mismatched caller.

In [`pipeline_ipywidgets.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/IPyWidgets/pipeline_ipywidgets.py):

- Keep `current_parameter_values` as short names (suffix building still uses `'minimum_inclusion_fr_Hz'` / `'included_qclu_values'`).
- Add a property such as `current_override_parameters_flat_keypaths_dict` that maps:
  - `minimum_inclusion_fr_Hz` → `rank_order_shuffle_analysis.minimum_inclusion_fr_Hz`
  - `included_qclu_values` → `rank_order_shuffle_analysis.included_qclu_values`
- Use that dict in `_subfn_compute_new` and `_subfn_compute` instead of `deepcopy(self.current_parameter_values)`.

No change needed in `Computation.py` or `gettable_mixin.py` for this crash.

## Verification

- Confirm `_subfn_compute_new` no longer calls `set_by_keypath('minimum_inclusion_fr_Hz', ...)`.
- Re-run Compute New in the notebook with the widget FR/qclu values; `update_parameters` should set `computation_config.rank_order_shuffle_analysis.*`.
