---
name: Heuristic triangle style
overview: Make a one-line style tweak so green heuristic triangles on DecodedEpochSlices are more transparent and have much thinner black edges.
todos:
  - id: tweak-dots-kwargs
    content: Update main_sequence_position_dots_kwargs alpha=0.45 and linewidths=0.5 in DecoderPredictionError.py
    status: completed
isProject: false
---

# Heuristic triangle alpha and stroke

## Target

In the live DecodedEpochSlices / stacked-epoch path, triangles come from `main_sequence_position_dots_kwargs` in [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py) (`DecodedSequenceAndHeuristicsPlotDataProvider`).

Current live kwargs (~line 3378):

```python
main_sequence_position_dots_kwargs = dict(should_skip=False, linewidths=2, marker ="^", edgecolor="#141414F9", s=75, zorder=11, alpha=0.85),
```

## Change (minimal)

Edit only that dict:

- `alpha`: `0.85` → `0.45`
- `linewidths`: `2` → `0.5`

Leave marker, edge color, size, and zorder unchanged. Do not touch defaults in `heuristic_replay_scoring.py` (debug path) or shared inclusion-status `override_alpha` (that also drives hline alpha).

## Note

When `show_heuristic_criteria_filter_epoch_inclusion_status` is on, included epochs still force triangle `alpha=0.85` via `override_alpha`. That behavior is left as-is for a minimal visual fix of the default overlay.