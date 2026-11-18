---
name: Port track body method
overview: Port `LinearTrackInstance.add_is_track_body_long_smart` from `release/pho-diba-2025-paper` onto current `develop` so the PhoDiba archive notebook (and any other callers) stop hitting AttributeError.
todos:
  - id: port-method
    content: Add LinearTrackInstance.add_is_track_body_long_smart to track_shape_drawing.py on develop (after build_x_position_classification_df)
    status: completed
  - id: verify-import
    content: Smoke-check that the classmethod is importable from the package
    status: completed
isProject: false
---

# Port `add_is_track_body_long_smart` to develop

## Context

The archive notebook calls `LinearTrackInstance.add_is_track_body_long_smart`, but that `@classmethod` exists only on `release/pho-diba-2025-paper` (added in `b3e531f3b`, 2025-11-05). Current branch is `develop`, where [`track_shape_drawing.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py) has no such method.

Only **one** missing method was found; no other `LinearTrackInstance` helpers from the paper branch are absent on develop for this failure.

## Change

Insert the paper-branch implementation into [`LinearTrackInstance`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py) immediately after `build_x_position_classification_df` (after line ~877), before `plot_rects`, with **two blank lines** between methods.

Source to copy (from `release/pho-diba-2025-paper`):

```python
@function_attributes(short_name=None, tags=['track_body', 'endcap', 'smart'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-11-04 14:03', related_items=[])
@classmethod
def add_is_track_body_long_smart(cls, a_df: pd.DataFrame) -> pd.DataFrame:
    """..."""  # keep paper docstring/usage
    assert 'P_Short' in a_df.columns
    assert 'is_decoded_pos_long_track_body' in a_df.columns
    assert 'is_decoded_pos_short_track_body' in a_df.columns

    a_df['is_most_likely_decoder_Long'] = a_df['P_Short'].apply(lambda x: (x < 0.5))
    a_df['is_track_body'] = np.where(
        a_df['is_most_likely_decoder_Long'],
        a_df['is_decoded_pos_long_track_body'],
        a_df['is_decoded_pos_short_track_body']
    )
    return a_df
```

`function_attributes`, `np`, and `pd` are already imported in that file.

## Out of scope

- Do **not** port the larger `MultiContextComparingDisplayFunctions._display_measured_vs_decoded_occupancy_distributions` paper-branch changes (`include_only_track_body_time_bins`, nested `_subfn_add_decoded_marginal_important_columns`). Those are optional display plumbing; develop has no call site for this method today, and the notebook already builds the long/short columns then calls the classmethod.
- Do **not** edit the archive `.ipynb` (call site is already correct).

## Verify

- Confirm `hasattr(LinearTrackInstance, 'add_is_track_body_long_smart')` / import from `pyphoplacecellanalysis.Pho2D.track_shape_drawing`.
- Re-run the failing notebook cell after module reload.