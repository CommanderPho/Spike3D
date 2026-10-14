---
name: Laps col override mirror
overview: Mirror the existing PBE column/ylabel override for the Laps panel of Figure 2, so lap participation fraction (or any other pair of columns) can be selected the same way as for PBEs.
todos:
  - id: laps-col-shell
    content: Add Laps_col_variable_names to build_shell_object_for_plot; use for Fig2_Laps_FR and Fig2_Laps_AnyC_FR
    status: completed
  - id: laps-ylabel
    content: Add laps_y_axis_label kwargs.get in fig_2_Theta_FR_matplotlib; update docstring example
    status: completed
isProject: false
---

# Mirror Laps Column Override Like PBE

## Approach
Copy the existing PBE pattern exactly:
- `PBE_col_variable_names` → `Laps_col_variable_names` (default `['lap_delta_minus', 'lap_delta_plus']`)
- `pbe_y_axis_label` → `laps_y_axis_label` (default `'Lap-averaged Firing Rates (Hz)'`)

## 1. Shell builder — [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py)

In `build_shell_object_for_plot` (~280):
- Add param `Laps_col_variable_names: Optional[List[str]] = None`
- Resolve like PBE:

```python
if Laps_col_variable_names is None:
    Laps_col_variable_names = ['lap_delta_minus', 'lap_delta_plus']
assert len(Laps_col_variable_names) == 2
Laps_delta_minus_col_name, Laps_delta_plus_col_name = Laps_col_variable_names
```

- Build `Fig2_Laps_FR` from those columns (replace hard-coded `lap_delta_*` at ~412)
- When `include_non_XdC_cells_as_inset=True`, build `Fig2_Laps_AnyC_FR` from the same Laps column names (~444)
- Update Usage 2 docstring with the participation example:

```python
# Laps_col_variable_names = ['lap_delta_minus', 'lap_delta_plus']
Laps_col_variable_names = ['ThetaDeltaMinus_ratio_participating_epochs', 'ThetaDeltaPlus_ratio_participating_epochs']
laps_y_axis_label = 'Fraction of Participating Lap Epochs'
```

## 2. Plot ylabel — [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)

In `fig_2_Theta_FR_matplotlib` (~1227), mirror PBE:

```python
laps_y_axis_label: str = kwargs.get('laps_y_axis_label', 'Lap-averaged Firing Rates (Hz)')
# ...
return cls.create_plot(..., laps_y_axis_label, 'Lap ($\\theta$)', ...)
```

`across_sessions_bar_graphs` / `display` already forward `**kwargs`, so `laps_y_axis_label=...` reaches the plotter with no extra wiring.

## Example call (after change)

```python
Laps_col_variable_names = ['ThetaDeltaMinus_ratio_participating_epochs', 'ThetaDeltaPlus_ratio_participating_epochs']
PBE_col_variable_names = ['ReplayDeltaMinus_ratio_participating_epochs', 'ReplayDeltaPlus_ratio_participating_epochs']
shell_obj, viz = build_shell_object_for_plot(..., Laps_col_variable_names=Laps_col_variable_names, PBE_col_variable_names=PBE_col_variable_names)
AcrossSessionsVisualizations.across_sessions_bar_graphs(
    shell_obj, num_sessions=n,
    laps_y_axis_label='Fraction of Participating Lap Epochs',
    pbe_y_axis_label='Fraction of Participating PBE Epochs',
    **fig2_kwargs)
```
