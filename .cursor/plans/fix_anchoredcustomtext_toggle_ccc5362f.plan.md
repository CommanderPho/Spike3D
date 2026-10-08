---
name: Fix AnchoredCustomText toggle
overview: Minimally fix the crash when toggling `use_AnchoredCustomText=False` by recreating overlay text artists whenever the extant type does not match the requested mode, instead of calling `.txt.set_text` on `AnchoredCustomText`.
todos: []
isProject: false
---

# Fix `use_AnchoredCustomText` toggle crash

## Cause

In [DecoderPredictionError.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py):

- WCorr callback (~L2521-2529): when `use_AnchoredCustomText` is False, it does `anchored_text.txt.set_text(...)`. Extant artists created with `AnchoredCustomText` have no `.txt`.
- Radon callback (~L1973-1981): same pattern.

The True branch already remove+recreates; the False branch assumes a plain `AnchoredText`.

## Fix

In both callbacks, when updating an extant label and `use_AnchoredCustomText` is False: if the extant artist is not a plain `AnchoredText` (or lacks `.txt`), remove it and recreate via `add_inner_title(...)` with the current flag — same as the True path. Only call `.txt.set_text` when `isinstance(anchored_text, AnchoredText)` and not `AnchoredCustomText`.

Concrete replacement for the False branch in both places:

```python
else:
    if (not isinstance(anchored_text, AnchoredText)) or isinstance(anchored_text, AnchoredCustomText) or (not hasattr(anchored_text, 'txt')):
        anchored_text.remove()
        anchored_text = add_inner_title(curr_ax, final_text, use_AnchoredCustomText=use_AnchoredCustomText, custom_value_formatter=custom_value_formatter, **text_kwargs)
        anchored_text.patch.set_ec("none")
        anchored_text.set_alpha(...)  # keep existing alpha for that provider
    else:
        anchored_text.txt.set_text(final_text)
```

Simpler equivalent (preferred): always remove+recreate in the False branch too (matches True branch; slightly more redraw, zero type checks). Use that for minimal risk.

## Files

- Only [DecoderPredictionError.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py): radon `_callback_update_curr_single_epoch_slice_plot` and wcorr `_callback_update_curr_single_epoch_slice_plot`.

## Verify

```python
paginated_multi_decoder_decoded_epochs_window.update_params(use_AnchoredCustomText=False)
paginated_multi_decoder_decoded_epochs_window.refresh_current_page()
```

No `AttributeError`; labels show as solid `text_foreground` color (no coolwarm value coloring). Toggle back to `True` and refresh still works.