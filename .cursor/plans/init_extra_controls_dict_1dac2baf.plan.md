---
name: Init extra controls dict
overview: Add optional `extra_controls_dict` to `DataFrameFilter` so dropdown/selectMultiple extras can be declared at construction and built automatically in `__attrs_post_init__` via the existing builder methods.
todos:
  - id: add-field-helper
    content: Add extra_controls_dict field + _apply_extra_controls_dict helper
    status: completed
  - id: wire-post-init
    content: Call helper in __attrs_post_init__ after _setup_widgets, before first filter update; update class Usage docstring
    status: completed
isProject: false
---

# Init-time `extra_controls_dict` for DataFrameFilter

## Goal

Allow notebook code to pass extras at construction instead of calling builders after `DataFrameFilter(...)`:

```python
extra_controls_dict = {
    'dropdown': [
        dict(a_name='trained_compute_epochs', df_col_name='trained_compute_epochs', a_widget_label='TrainedComputeEpochs :', custom_initial_selection='laps'),
        dict(a_name='decoder_identifier', df_col_name='decoder_identifier', a_widget_label='decoder_identifier:', custom_initial_selection='pseudo2D'),
    ],
    'selectMultiple': [
        dict(a_name='masked_time_bin_fill_type', df_col_name='masked_time_bin_fill_type', a_widget_label='masked_time_bin_fill_type:', initial_selection_mode=InitialSelectionModeEnum.FIRST_SELECTED),
    ],
}

df_filter = DataFrameFilter(..., extra_controls_dict=extra_controls_dict)
```

## Change (single file)

**File:** [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) — `DataFrameFilter`

1. **Field** (near other non-serialized init knobs ~8662):

```python
extra_controls_dict: Optional[Dict[str, List[Dict]]] = non_serialized_field(default=None)
```

Shape: `{control_kind: [kwargs_dict, ...]}` with `control_kind` in `{'dropdown', 'selectMultiple'}`. Each kwargs dict is passed through to the matching builder (`**kwargs`).

2. **Helper** ` _apply_extra_controls_dict(self)`:
   - No-op if `extra_controls_dict` is None/empty
   - Map: `'dropdown' -> self.build_extra_dropdown_widget`, `'selectMultiple' -> self.build_extra_selectMultiple_widget`
   - Iterate dict items in order; for each kwargs dict call the builder
   - `ValueError` on unknown control kind

3. **Wire in `__attrs_post_init__`** after `_setup_widgets()` (which clears/creates `custom_dynamic_filter_widgets_list`) and **before** the first `update_filtered_dataframes(...)`, so widget predicates exist for the initial filter pass:

```python
self._setup_widgets()
self._apply_extra_controls_dict()
if getattr(self, 'time_bin_size_widget', None) is not None:
    self.update_filtered_dataframes(...)
```

4. **Docstring** Usage block on `DataFrameFilter`: show `extra_controls_dict=` example; keep post-init `build_extra_*` calls valid.

No notebook edits unless you ask later.
