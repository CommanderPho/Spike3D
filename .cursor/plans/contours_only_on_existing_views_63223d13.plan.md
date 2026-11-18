---
name: Contours only on existing views
overview: Make the contour block explicitly draw only on views that exist (filter out None), so in multi-epoch mode we never iterate over past/future and updates remain correct in both single-epoch and multi-epoch modes.
todos: []
isProject: false
---

# Contours only on existing views in both modes

## Current behavior

- In **single-epoch** mode, `past_view`, `posterior_2d_view`, and `future_view` are all set; the contour loop draws on all three (and the later `_update_dict` update writes all three contour lists).
- In **multi-epoch** mode, the caller passes `past_view=None`, `future_view=None` in `_update_dict`. The loop at 7881–7885 runs over all three pairs but does nothing when `view is not None` is false, so contours are only drawn on the central view. The zip at 7902–7904 only sets `_update_dict[a_name]` when `a_view is not None`, so past/future keys are left as the initial empty lists.

So behavior is already correct, but the code does two no-op iterations per contour in multi-epoch and does not make the “no past/future views” case obvious.

## Change

**File:** [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py), contour block ~7854–7907.

1. **Draw only on existing views**
  After building `active_posterior_contours_dict_list` (line 7861), define a list that excludes missing views and use it only for drawing:
  - e.g. `views_and_lists_to_draw = [(v, c) for v, c in active_posterior_contours_dict_list if v is not None]`
  - In the inner contour loop (7881), iterate over `views_and_lists_to_draw` instead of `active_posterior_contours_dict_list`.
  - Remove the `if view is not None:` check inside that loop (redundant once the list is filtered).
2. **Keep _update_dict logic unchanged**
  Keep using `active_posterior_contours_dict_list` (and the existing zip with `list_names`) for the update at 7902–7904. That way single-epoch still gets all three keys written when views exist, and multi-epoch still only writes keys for non-None views (posterior only), leaving past/future as the initial empty lists.
3. **Comment**
  Short comment that in multi-epoch mode there are no past/future views, so only the central view (and time-bin views) receive contours.

## Summary


| Item            | Action                                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| After line 7861 | Add `views_and_lists_to_draw = [(v, c) for v, c in active_posterior_contours_dict_list if v is not None]`. |
| Line 7881       | Use `views_and_lists_to_draw` instead of `active_posterior_contours_dict_list`.                            |
| Lines 7882–7883 | Remove the `if view is not None:` guard (filtering makes it redundant).                                    |
| Comment         | Note that multi-epoch has no past/future views, so drawing is central (and time-bin) only.                 |


No change to the multi-epoch call site or to the `_update_dict` update loop; both modes continue to behave as today, with drawing explicitly limited to existing views.