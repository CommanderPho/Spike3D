---
name: Harden position-dependent reliability
overview: Hardening `BayesianPlacemapPositionDecoder` so `POSITION_DEPENDENT` reliability estimation is correct end-to-end (aligned rates, fail-loud missing inputs, one-shot refresh after confusion, slice-safe recompute), without changing Zhang `decode()`, debugger UI, or per-bin confusion aggregation.
todos:
  - id: align-rates-fail-loud
    content: Reindex confusion rates by neuron_IDs; POSITION_DEPENDENT raises if confusion/masks missing; PER_CELL keeps ones fallback
    status: completed
  - id: auto-refresh-slice-docs
    content: Auto-call _compute_reliability_metrics after confusion; slice t_bin_aclus_reliability_df in Bayesian+DST get_by_id; fix Skaggs docstring
    status: completed
isProject: false
---

# Harden position-dependent reliability on Bayesian decoder

## Constraints (from plan out-of-scope)

- Do **not** change Zhang [`neuropy_bayesian_prob`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) / `BasePositionDecoder.decode()` formulas.
- Do **not** touch debugger / [`PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) UI.
- Do **not** add per-spatial-bin confusion aggregation; keep existing per-cell rates × `in_field_masks`.

Estimation already exists on [`BayesianPlacemapPositionDecoder`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (~4023–4153). This pass only hardens that path so `POSITION_DEPENDENT` works fully and correctly.

## Gaps to fix

1. **Rate / neuron alignment** — `_compute_reliability_metrics` does `rel_df['true_pos'].to_numpy()` without reindexing by `self.neuron_IDs`. Misordered or partial dfs would mis-assign rates. Reindex: `rel_df.reindex(neuron_ids)` then extract columns; `nan_to_num` after.
2. **Silent fallback vs mode** — Missing confusion currently always writes 1D ones, even when `reliability_estimation_mode == POSITION_DEPENDENT`. For that mode, raise a clear error requiring `compute_unit_confusion_reliability_variables(...)` first. Keep ones fallback only for `PER_CELL`.
3. **One-shot refresh** — End of `compute_unit_confusion_reliability_variables` should call `_compute_reliability_metrics()` so setting mode then computing confusion fills `reliability_*` without a second manual call.
4. **Slice-safe recompute** — `get_by_id` currently clears `t_bin_aclus_reliability_df` while keeping sliced 2D reliability maps. Also neuron-slice the confusion df (`df.loc[kept_ids]` in `neuron_IDs` order) so a later `_compute_reliability_metrics()` on the slice can rebuild maps. Same for DST `get_by_id`.
5. **Docs** — Remove stale Skaggs wording in `compute_unit_confusion_reliability_variables` docstring; note modes / auto-refresh.

## Concrete edits (only these files)

### [`reconstruction.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) — `BayesianPlacemapPositionDecoder`

- `compute_unit_confusion_reliability_variables`: update docstring; after writing confusion products, call `self._compute_reliability_metrics()`.
- `_compute_reliability_metrics`:
  - `neuron_ids = np.asarray(self.neuron_IDs ...)`
  - if no confusion: `POSITION_DEPENDENT` → raise; `PER_CELL` → ones (unchanged)
  - else `rel_df = self.t_bin_aclus_reliability_df.reindex(neuron_ids)` then rates
  - assert `len(true_pos) == n_neurons`
- `get_by_id`: instead of setting `t_bin_aclus_reliability_df = None`, copy neuron-sliced rows when present: `self.t_bin_aclus_reliability_df.reindex(source_ids[keep])` (keep other time-bin tables `None` as today).

### [`reconstruction_dst.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction_dst.py)

- Mirror the `t_bin_aclus_reliability_df` neuron-slice copy in DST `get_by_id`.

Flatten order (`mask.ravel('C')` vs `np.reshape(tuning/occupancy, -1)`) already matches `build_concatenated_F`; no change there.

## Usage after hardening

```python
decoder.reliability_estimation_mode = ReliabilityEstimationMode.POSITION_DEPENDENT
decoder.compute_unit_confusion_reliability_variables(...)
# → reliability_active.shape == (n_flat_position_bins, n_neurons)
```
