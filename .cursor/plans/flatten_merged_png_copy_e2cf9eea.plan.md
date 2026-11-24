---
name: Flatten merged PNG copy
overview: After the batch export writes the merged 4-decoder PNG under the per-session subdirectory, copy that final image one level up into `laps_correct_decoder_side_by_side/` and report the flat path in callback outputs.
todos:
  - id: copy-merged-to-parent
    content: After successful display unwrap in figures_export_laps_correct_decoder_side_by_side_completion_function, copy merged PNG to parent dir and update combined_img_out_path in callback_outputs
    status: completed
isProject: false
---

# Flatten merged laps PNG to parent dir

## Context

`_display_laps_correct_decoder_side_by_side_posteriors` currently writes:

- pages under `.../laps_correct_decoder_side_by_side/<session_key>/pages/`
- merged PNG at `.../laps_correct_decoder_side_by_side/<session_key>/<basename>.png`

via `output_dir = parent_output_folder.joinpath('laps_correct_decoder_side_by_side', session_key)` in [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py).

Desired browsing layout for the merged image only:

- `.../laps_correct_decoder_side_by_side/<basename>.png`

Session subdirs (pages + nested merged original) stay as-is.

## Change (one place)

Edit [`figures_export_laps_correct_decoder_side_by_side_completion_function`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) immediately after unwrapping `graphics_output_dict` (~4487–4489):

1. Read nested `combined_img_out_path` from the display result.
2. If it exists, `shutil.copy2` to `Path(combined_img_out_path).parent.parent / Path(combined_img_out_path).name` (i.e. `laps_correct_decoder_side_by_side/<basename>.png`).
3. Set `callback_outputs['combined_img_out_path']` to that flat path (keep `output_dir` / `page_png_paths` pointing at the nested session folder).
4. Print the flat destination.

`shutil` and `Path` are already imported at module top — no new imports.

No changes to the display/export subfn; interactive `display(...)` behavior unchanged.
