---
name: Laps GT Decoder Export
overview: Add a headless-safe (matplotlib Agg) global display function that plots all four directional lap posteriors side-by-side with a green ground-truth decoder box per lap, exports every page, and stitches them into one combined PNG per session—plus a thin batch completion wrapper registered for SLURM.
todos:
  - id: core-helpers
    content: "Add Agg-safe helpers: groundtruth mapping, page plot (4 cols), multipage export + vertical combine"
    status: completed
  - id: display-fn
    content: Add global _display_laps_correct_decoder_side_by_side_posteriors on DirectionalPlacefieldGlobalDisplayFunctions
    status: completed
  - id: batch-completion
    content: Add figures_export_laps_correct_decoder_side_by_side_completion_function and register in MAIN_get_template_string
    status: completed
isProject: false
---

# Laps 4-Decoder GT-Boxed Batch Export (Headless Agg)