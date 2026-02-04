---
name: Central view contours both modes
overview: Make `_render_central_view` rely only on `_update_dict` (and shared data) for view and contour list references so contours and other visuals render correctly in both single-epoch and multi-epoch overview modes, without depending on global instance views.
todos: []
isProject: false
---

# Make _render_central_view work in both single-epoch and multi-epoch modes

## Root cause

`_render_central_view` was refactored to take `_update_dict` and use it for some state, but it still **hardcodes** `self.past_view` and `self.future_view` when building the list of (view, contour_list) pairs for drawing contours (line 7840). It also does not receive explicit contour list or view overrides from the multi-epoch call site.

- **Single-epoch mode**: `self.past_view`, `self.posterior_2d_view`, `self.future_view` are set when building the grid (lines 7171–7174). The single-epoch call (8379) passes a full `_update_dict` including `past_mask_contours`, `posterior_mask_contours`, `future_mask_contours` from `self`, and does not pass `posterior_2d_view` (so it falls back to `self.posterior_2d_view`). Contours are drawn on all three views.
- **Multi-epoch mode**: Only a **per-epoch** central view (`a_posterior_2d_view`) and `time_bin_grid` exist; there are no per-epoch past/future views. The multi-epoch loop (7286–7304) passes `_an_update_dict` with only `posterior_2d_view` and `time_bin_grid`; contour list keys are commented out (7292). So:
  - `posterior_2d_view` is correct (per-epoch central view).
  - `past_view` and `future_view` are taken from `self` inside `_render_central_view`. In multi-epoch, the grid that sets `self.past_view` / `self.future_view` is never created (that code is in `if not self.enable_multi_epoch_overview_display_mode`), so they remain `None` (dataclass default). So the contour block effectively only draws on the central view. If the widget was ever used in single-epoch first, `self.past_view`/`self.future_view` could be non-None and contours would incorrectly be drawn on those global views for every epoch.
- **Contours “not showing”**: Even with `past_view`/`future_view` as `None`, contours should appear on the central view **if** contour data exists (`epoch_flat_mask_future_past_result`, `epoch_t_bins_high_prob_pos_mask`). Making view and contour list resolution fully explicit via `_update_dict` removes reliance on `self` for views and ensures only the intended views receive contours; it also makes multi-epoch behavior robust if instance state is ever mixed (e.g. single-epoch then multi-epoch).

## Approach

1. **Resolve all view and contour list references from `_update_dict**` inside `_render_central_view`, with fallback to `self` only for single-epoch compatibility.
2. **Multi-epoch call site**: Pass explicit `past_view=None`, `future_view=None`, and initial contour lists (e.g. empty lists) so that contour rendering only targets the per-epoch central view and returned state is complete.

No change to contour **data** source (`self.epoch_flat_mask_future_past_result`, etc.); only where views and contour lists come from.

---

## 1. In `_render_central_view`: resolve views and contour lists from `_update_dict`

**File:** [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

- **Posterior view** (already correct): Keep `posterior_2d_view = _update_dict.get('posterior_2d_view', self.posterior_2d_view)` (line 7512).
- **Contour block (lines 7835–7840):** Replace the hardcoded use of `self.past_view` and `self.future_view` when building `active_posterior_contours_dict_list`:
  - Define `past_view = _update_dict.get('past_view', self.past_view)` and `future_view = _update_dict.get('future_view', self.future_view)` (or equivalent inline).
  - Build `active_posterior_contours_dict_list` as `[(past_view, _update_dict.get('past_mask_contours', [])), (posterior_2d_view, _update_dict.get('posterior_mask_contours', [])), (future_view, _update_dict.get('future_mask_contours', []))]` so all three views and contour lists come from `_update_dict` with fallback to `self` where appropriate.

Result: In multi-epoch, when the caller passes `past_view=None`, `future_view=None`, contours will only be drawn where `view is not None` (the per-epoch central view). In single-epoch, the caller does not pass these keys, so `self.past_view` and `self.future_view` are used and behavior stays the same.

---

## 2. Multi-epoch call site: pass explicit view and contour list keys

**File:** Same, multi-epoch loop where `_an_update_dict` is built and `_render_central_view` is called (lines 7284–7304).

- When creating the new `_an_update_dict` (lines 7286–7293), add:
  - `past_view=None`, `future_view=None` (so contours are only drawn on the per-epoch central view).
  - Optional but recommended for clarity and so returned state is complete: `past_mask_contours=[]`, `posterior_mask_contours=[]`, `future_mask_contours=[]` (or pass the lists that will be updated so the same references are returned).

This makes multi-epoch mode self-contained and independent of `self.past_view` / `self.future_view`.

---

## 3. Single-epoch call site: no change required

**File:** Same, single-epoch call (lines 8379–8388).

- The existing `_update_dict` already passes `posterior_2d_view` implicitly (not in dict → uses `self.posterior_2d_view`), and `past_mask_contours`, `posterior_mask_contours`, `future_mask_contours` from `self`. Do **not** add `past_view`/`future_view` here so that fallback to `self.past_view` and `self.future_view` continues to apply and single-epoch behavior is unchanged.

---

## 4. Optional: docstring and contour list update consistency

- In the docstring of `_render_central_view` (lines 7474–7486), document that `_update_dict` may optionally include `past_view`, `future_view`; when omitted, `self.past_view` and `self.future_view` are used (single-epoch). When provided as `None` (multi-epoch), contours are drawn only on the central view.
- After the contour block, the code already updates `_update_dict` with the three contour list keys (7880, 7881–7884). No change needed there once views and lists are taken from `_update_dict` as above.

---

## Summary of edits


| Location                                 | Change                                                                                                                                                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contour block ~7838–7840                 | Introduce `past_view` / `future_view` from `_update_dict.get(..., self.past_view/future_view)` and use them (and contour lists from `_update_dict`) to build `active_posterior_contours_dict_list`. |
| Multi-epoch `_an_update_dict` ~7286–7293 | Add `past_view=None`, `future_view=None`, and optionally `past_mask_contours=[]`, `posterior_mask_contours=[]`, `future_mask_contours=[]`.                                                          |
| Single-epoch call ~8379                  | No change.                                                                                                                                                                                          |
| Docstring ~7474–7486                     | Optionally document optional `past_view` / `future_view` in `_update_dict`.                                                                                                                         |


---

## If contours still do not appear in multi-epoch

Then the issue is likely **data**, not views:

- Confirm `self.epoch_flat_mask_future_past_result` is set and has an entry for the epoch index used in the loop.
- Confirm each `MatchingPastFuturePositionsResult` has `epoch_t_bins_high_prob_pos_mask` not None and with non-empty masks so `measure.find_contours` produces contours.

No code changes are required for that in this plan; the above changes only make view and contour list resolution explicit and correct for both modes.