---
name: Fig2 title metric labels
overview: Derive the Figure 2 flexitext title metric phrase from the plot's y-axis label so participation plots no longer hard-code "Firing Rates".
todos:
  - id: title-metric-from-ylabel
    content: Add metric_name to _build_formatted_title_string; derive it in create_plot by parsing ylabel
    status: completed
  - id: title-metric-remove-kwargs-docs
    content: No new title kwargs; docstring already has ylabel overrides only (confirm no laps_title_metric docs needed)
    status: completed
isProject: false
---

# Fig2 Title Metric Labels (from ylabel)

## Problem
[`_build_formatted_title_string`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) hardcodes:

```python
f"<size:...><weight:bold>{epochs_name}</> Firing Rates\n..."
```

When y-axis is participation, titles still say **"PBE Firing Rates"** / **"Lap (θ) Firing Rates"**.

## Locked approach
No new title kwargs. Parse the **ylabel already passed into `create_plot`** and use that to choose the title metric phrase.

### Parse rules (in a small helper, e.g. `_metric_name_from_ylabel`)
- if `'participat'` in `ylabel.lower()` → `'Participation Fractions'`
- elif `'firing rate'` in `ylabel.lower()` → `'Firing Rates'`
- else → `'Firing Rates'` (safe default)

Covers current labels:
- `'Lap-averaged Firing Rates (Hz)'` / `'PBE-averaged Firing Rates (Hz)'` → Firing Rates
- `'Fraction of Participating Lap Epochs'` / `'Fraction of Participating PBE Epochs'` → Participation Fractions

### 1. [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)

- Add `@classmethod def _metric_name_from_ylabel(cls, ylabel: str) -> str` with the rules above.
- Change `_build_formatted_title_string(..., metric_name: str = 'Firing Rates')` to interpolate `{metric_name}`.
- In `create_plot`, after ylabel is known:
  ```python
  metric_name = cls._metric_name_from_ylabel(ylabel)
  ...
  cls._build_formatted_title_string(epochs_name=title, prepare_for_publication=..., metric_name=metric_name)
  ```
- No changes required in `fig_2_Theta_FR_matplotlib` / `fig_2_Replay_FR_matplotlib` beyond existing ylabel kwargs (already wired).

### 2. Docs
No docstring updates for title kwargs. Existing `laps_y_axis_label` / `pbe_y_axis_label` usage is sufficient; titles follow automatically.

## Resulting titles
- FR ylabel → `Lap (θ) Firing Rates` / `PBE Firing Rates`
- Participation ylabel → `Lap (θ) Participation Fractions` / `PBE Participation Fractions`
  (plus existing `title_modifier` session suffix)
