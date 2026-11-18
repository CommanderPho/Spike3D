---
name: peak-prominence-memory-refinement
overview: Refactor `compute_2d_dt_posterior_peak_promenences` to keep identical outputs while cutting peak memory and temporary allocations for large time-bin counts (up to ~100k). Add low-risk internal optimizations and validation/benchmark checks to ensure numerical equivalence.
todos:
  - id: refactor-prealloc-masks
    content: Preallocate per-alpha 3D bool masks and replace list-append/stack pipeline with direct indexed writes.
    status: completed
  - id: replace-label-component
    content: Switch dominant connected-component extraction from `ndimage.label` to seed-based `ndimage.binary_propagation` with matching connectivity semantics.
    status: completed
  - id: temp-buffer-reuse
    content: Reuse per-time-bin temporary buffers and move invariant structures out of inner loops.
    status: completed
  - id: add-memory-guardrail
    content: Add expected-memory estimation warning/strict guard option before full computation starts.
    status: completed
  - id: validate-caller-compat
    content: Verify output type/shape equivalence against existing call sites and add a quick equivalence test on representative input.
    status: completed
isProject: false
---

# Refine Peak Prominence Memory Use

## Goal

Prevent kernel crashes from excessive memory allocation in `compute_2d_dt_posterior_peak_promenences` while preserving the exact return contract and numerical behavior.

## Current Bottlenecks

- The function currently accumulates nested per-time-bin masks in Python lists and later stacks them:
  - `[H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py)`
  - `epoch_masks.append(dominant_peak_mask)` inside the time loop, then final `np.stack(...)` across all bins.
- This creates major overhead from:
  - 100k Python container entries of 2D arrays.
  - Temporary duplication when converting list-of-2D masks into final 3D arrays.
  - Per-alpha full `ndimage.label(...)` int arrays for every time bin.

## Proposed Refinements (API-preserving)

1. **Preallocate final output arrays and write directly by time index**
  - In `compute_2d_dt_posterior_peak_promenences`, allocate once:
    - `epoch_masks = [np.zeros((n_x_bins, n_y_bins, n_t_bins), dtype=bool) for _ in alpha]`
  - Replace per-time-bin `epoch_masks.append(...)` with direct assignment:
    - `epoch_masks[alpha_idx][:, :, t_idx] = computed_mask`
  - Remove the final `np.stack(...)` step entirely.
  - Keeps return type and shape exactly as callers expect (`List[NDArray]`, each `(x, y, t)`).
2. **Replace label-based component extraction with seed-propagation component growth**
  - Instead of:
    - `labeled, _ = ndimage.label(threshold_mask)` then `labeled == labeled[px, py]`
  - Use:
    - one-pixel seed at dominant peak and `ndimage.binary_propagation(..., mask=threshold_mask)`.
  - This avoids allocating an integer label map each alpha/time bin and returns the same connected component semantics when using the same connectivity.
3. **Reuse temporary buffers per time bin where possible**
  - Reuse `threshold_mask` / seed buffers in loop (or use `np.greater_equal(..., out=...)`) to reduce transient array churn.
  - Keep function behavior identical for no-peak bins (all-False masks).
4. **Low-risk micro-optimizations in hotspot path**
  - Move repeated structural constants (connectivity structure) out of inner loops.
  - Keep `epoch_promenence_tuples` as-is for compatibility, but pre-size the list to `n_t_bins` and assign by index to reduce Python overhead.
5. **Add explicit memory guardrails and diagnostics**
  - Estimate expected output bytes before heavy work using `n_x_bins * n_y_bins * n_t_bins * n_alpha`.
  - If estimate is above a configurable threshold, emit a clear warning (or optionally raise if strict mode is enabled).
  - This does not change outputs; it prevents surprise kernel crashes and clarifies memory floor for large runs.

## Compatibility Validation

- Verify compatibility against current consumers:
  - `[H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py)`
  - `[H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/SequenceBasedComputations.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/SequenceBasedComputations.py)`
- Ensure unchanged expectations:
  - `dict(zip(alpha_list, epoch_masks_list))` still works.
  - Each mask array shape remains equal to `a_p_x_given_n.shape`.
  - Numeric metrics derived via `np.nansum` remain unchanged.

## Notes on Practical Memory Floor

- Even with perfect internals, returning full boolean masks has an unavoidable memory floor of roughly:
  - `n_x_bins * n_y_bins * n_t_bins * n_alpha` bytes.
- Example at `41x63x100000`:
  - ~246 MiB per alpha for mask storage alone.
- The refinements above remove avoidable overhead (especially list-of-2D accumulation and label-map churn), which is the primary source of current crashes.

