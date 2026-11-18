---
name: Pseudo2D Walkthru PDF
overview: Add a standalone matplotlib function that reproduces the figS2 Map-Based Context Decoder walkthrough PDF for one epoch’s Pseudo2D posterior (SingleEpochDecodedResult or NDArray), always drawing Long/Short after collapsing directional contexts.
todos:
  - id: add-function
    content: Add plot_map_based_context_decoder_walkthru (+ helpers) to PhoDiba2023Paper.py with input normalization + Long/Short collapse using DirectionalPseudo2DDecodersResult helpers
    status: completed
  - id: plot-layout
    content: Implement matplotlib GridSpec walkthrough (raw stacks, row-sum, +, track_identity marginal, trapezoid, labels) and PDF save
    status: completed
  - id: smoke-check
    content: Smoke-test with synthetic 4-context and 2-context arrays; compare structure to figS2 PNG
    status: completed
isProject: false
---

# Pseudo2D Context-Decoder Walkthrough PDF

## What figS2 depicts

Illustrator PDF [`figS2_Map-BasedContextDecoderWalkthru.pdf`](c:\Users\pho\.cursor\projects\h-TEMP-Spike3DEnv-ExploreUpgrade-Spike3DWorkEnv-Spike3D\attachments\ac0936b5-475e-471d-a7bb-58c9418aec82\figS2_Map-BasedContextDecoderWalkthru.pdf) (page ~475×576 pt) is a **methods schematic**, not existing library output. One epoch panel shows left→right:

```mermaid
flowchart LR
  raw["Raw Pseudo2D Long/Short x-heatmaps per t"] --> rowSum["row sum over x"]
  rowSum --> plus["+ across Long/Short"]
  plus --> trackMarg["track_identity marginal 2 x T"]
  trackMarg --> epochSum["optional sum_t epoch bar"]
```

Template elements to recreate (from PDF Form/Image sizes):
- Per-time stacked **Long** (blue label) / **Short** (red label) 1D posterior heatmaps (`viridis`-like)
- Per-row **row ∑** thin color bars (mass after ∑_x)
- **+** between Long/Short row-sums per t
- Grey **trapezoid/callout** from mid-column into right **track_identity marginal** (2 rows × T cols)
- Labels: `t=i`, `raw (non-marginalized) pseudo2D decoder for each time bin`, `track_identity marginal`, `row ∑`, optional `∑_t` / `epoch`
- Example PDF stacks two epochs; **our function draws one epoch panel** (caller can stack later if needed)

## Locked decisions

- **Input (1B):** `SingleEpochDecodedResult` primary; also accept `NDArray` `(n_x, n_contexts, n_t)` (and squeeze 4D by summing spatial dims if needed).
- **Display (2A):** Always draw **Long/Short only** on the left. If `n_contexts==4`, sum LR+RL into Long/Short in position space first (same order as `DirectionalPseudo2DDecodersResult._marginalize_p_x_given_n_over_context_groups_in_position_space`). If already 2 contexts, use as Long/Short.
- **Location:** add the standalone function (and small private helpers) to [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py), near the other paper-figure exporters. Minimal edit: append a focused section; do not refactor the rest of the file.
- **Renderer:** matplotlib `GridSpec` + `savefig(..., format='pdf')` (vector page matching schematic; Agg-safe). No Illustrator/reportlab rebuild of the original PDF objects.

## Computation (reuse existing math)

Use helpers on [`DirectionalPseudo2DDecodersResult`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py):

1. Normalize input → `p_x_given_n` `(n_x, n_contexts, n_t)`.
2. Resolve `context_names` (from arg, or default `['long_LR','long_RL','short_LR','short_RL']` / `['long','short']`).
3. Build **display Long/Short position posteriors** `(n_x, 2, n_t)` via track-identity grouping in position space.
4. **Row sums:** `row_sum[c,t] = sum_x display[x,c,t]`; renormalize per t across Long/Short → `track_identity_marginal` `(2, n_t)` (same spirit as `_normalize_per_timebin_context_marginal`).
5. Optional epoch vector: `sum_t` of that marginal, renormalized.

Colors/labels from [`LongShortDisplayConfig`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\Configs\LongShortDisplayConfig.py) (Long blue / Short red text).

## Public API

```python
def plot_map_based_context_decoder_walkthru(
    pseudo2D_posterior: Union[SingleEpochDecodedResult, NDArray],
    save_path: Union[str, Path],
    context_names: Optional[List[str]] = None,
    max_time_bins: Optional[int] = None,  # subsample if epoch is long; default None = all, or cap e.g. 12 for readability
    cmap: str = 'viridis',
    show_epoch_sum: bool = True,
    title: Optional[str] = None,
) -> Tuple[Figure, Dict[str, Any]]:
```

Returns `(fig, extras)` where `extras` includes `display_p_x_given_n`, `track_identity_marginal`, `row_sums`, and `save_path`. Writes PDF (and optionally PNG sibling) to `save_path`.

## Layout implementation sketch

- Dark figure background close to the example.
- Left column: `n_t` blocks × 2 rows of `imshow` (or `pcolormesh`) for Long/Short at each t; y-tick `t=i` on the left of each block.
- Mid column: matching-height color strips for row ∑; text `row ∑` / `+`.
- Right: one `imshow` of shape `(2, n_t)` for track_identity marginal; Long/Short row labels via existing label styling.
- `matplotlib.patches.Polygon` (or `ConnectionPatch` fan) for the grey flow from the mid stack into the right heatmap (schematic, not pixel-perfect Illustrator clone).
- Cap `n_t` when `max_time_bins` set: take first `max_time_bins` bins (document clearly) so long continuous epochs remain readable like the ~9–10 bin example.

## Out of scope

- Reproducing Adobe Illustrator Form XObjects / CMYK indexed images bit-identically
- Interactive PyQt / `BinByBinDecodingDebugger` (different tool)
- Notebook edits (unless you later ask to wire a call site)

## Verify

- Synthetic `(n_x, 4, n_t)` array → PDF opens, Long/Short left panel, `(2,n_t)` marginal on right
- Synthetic `(n_x, 2, n_t)` array → same layout without directional collapse
- `SingleEpochDecodedResult` path extracts `.p_x_given_n` correctly
- Visual check against the rendered figS2 PNG for structure (labels, flow, colors)