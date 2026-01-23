---
name: clarify per-epoch terminology
overview: Update comments and docstrings in `MatchingPastFuturePositionsResult` to replace ambiguous "per-epoch" terminology with "per-position-trajectory-epoch" to distinguish from the parent decoding epoch.
todos:
  - id: update-line-1483
    content: Update comment at line 1483 to use 'per-position-trajectory-epoch'
    status: completed
  - id: update-line-1640
    content: Update docstring at line 1640 to use 'per-position-trajectory-epoch'
    status: completed
  - id: update-line-1723
    content: Update comment at line 1723 to use 'per-position-trajectory-epoch'
    status: completed
isProject: false
---

# Clarify Per-Epoch Terminology in MatchingPastFuturePositionsResult

## Context

Each `MatchingPastFuturePositionsResult` instance corresponds to a specific **decoding epoch**. Within that, positions are partitioned into **position trajectory epochs** (individual past/future trajectory segments). The recent "per-epoch" terminology is ambiguous.

## Changes

Update comments/docstrings in [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py):

### Line 1483

```python
## Propagate per-epoch segmentation columns back to relevant_positions_df
```

Change to:

```python
## Propagate per-position-trajectory-epoch segmentation columns back to relevant_positions_df
```

### Line 1640 (docstring)

```python
Tuple of (epochs DataFrame, dict of per-epoch position DataFrames with segmentation columns)
```

Change to:

```python
Tuple of (position trajectory epochs DataFrame, dict of per-position-trajectory-epoch DataFrames with segmentation columns)
```

### Line 1723

```python
## Segment trajectories per-epoch (so each trajectory gets its own representative direction angle)
```

Change to:

```python
## Segment trajectories per-position-trajectory-epoch (so each trajectory gets its own representative direction angle)
```