---
name: Simplify confusion matrix
overview: Simplify `perform_compute_confusion_matrix` to a tighter Polars pipeline using the standardized position/spike column names, without changing the known-bin / sparse-spike semantics or the rate formulas.
todos:
  - id: rewrite-confusion-fn
    content: Rewrite perform_compute_confusion_matrix body with standardized names and fewer temps
    status: completed
  - id: update-docstring
    content: Update Returns/Notes docstring to match new column names
    status: completed
isProject: false
---

# Simplify `perform_compute_confusion_matrix`

## Scope

Rewrite only the body of [`perform_compute_confusion_matrix`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/reliability.py) (lines ~1151–1265). No callers need code changes: downstream code only requires `true_pos` (and aclu index); notebooks comment old column lists but are out of scope.

## Correctness invariants (unchanged)

- Restrict visits/spikes to **known** animal positions (`pos` ∩ unique LUT spatial bins).
- Per aclu: `n_outfield_tbins = n_total_tbins - n_infield_tbins` (no `aclus × tbins` cross-join).
- `per_tbin` is nonzero-only → spike rows are firing events (`len` on filtered rows is fine instead of `n_unique`).
- Rates: TP/FP = spike counts / `n_total_spikes`; TN/FN = nonspike tbins / opportunity tbins.

## Column naming (standardized)

Keep rates and spike totals; rename opportunity/silent counts to the comment names:

- Position: `n_infield_tbins`, `n_outfield_tbins`, `n_total_tbins` (was `n_in_field_tbins` / `n_out_of_field_tbins` / `n_known_tbins`)
- Spike: `n_infield_spike_tbins`, `n_outfield_spike_tbins`, `n_infield_nonspike_tbins`, `n_outfield_nonspike_tbins` (nonspike replaces `false_neg_n_tbins` / `true_neg_n_tbins`)
- Keep: `neuron_IDX`, `n_computed_bins`, `true_pos_n_spikes`, `false_pos_n_spikes`, `n_total_spikes`, `true_pos`/`false_pos`/`true_neg`/`false_neg`

Do **not** drop the spike-tbin columns at the end.

## Implementation shape

Collapse into: prepare → `known_pos` → two aggs (`n_in`, `spike_aggs`) → one join pipeline → pandas.

```python
known_pos = pos.join(lut.select(['binned_x', 'binned_y']).unique(), on=['binned_x', 'binned_y'], how='inner')
n_total_tbins = known_pos.height

n_in = known_pos.join(lut, on=['binned_x', 'binned_y'], how='inner').group_by('aclu').agg(pl.len().alias('n_infield_tbins'))

sp = (
    spikes
    .join(known_pos.select(['t_bin_idx', 'binned_x', 'binned_y']), on='t_bin_idx', how='inner')
    .join(lut.with_columns(pl.lit(True).alias('is_in_field')), on=['aclu', 'binned_x', 'binned_y'], how='left')
    .with_columns(pl.col('is_in_field').fill_null(False))
)
spike_aggs = sp.group_by('aclu').agg([
    pl.col('n_spikes').filter(pl.col('is_in_field')).sum().fill_null(0).alias('true_pos_n_spikes'),
    pl.col('n_spikes').filter(~pl.col('is_in_field')).sum().fill_null(0).alias('false_pos_n_spikes'),
    pl.len().filter(pl.col('is_in_field')).fill_null(0).alias('n_infield_spike_tbins'),
    pl.len().filter(~pl.col('is_in_field')).fill_null(0).alias('n_outfield_spike_tbins'),
])

out_pl = (
    pl.DataFrame({'aclu': neuron_ids_i64, 'neuron_IDX': np.arange(len(neuron_ids), dtype=np.int64)})
    .with_columns([pl.lit(n_total_tbins).alias('n_total_tbins'), pl.lit(n_computed_bins).alias('n_computed_bins')])
    .join(n_in, on='aclu', how='left')
    .join(spike_aggs, on='aclu', how='left')
    .with_columns([...fill_null 0...])
    .with_columns((pl.col('n_total_tbins') - pl.col('n_infield_tbins')).alias('n_outfield_tbins'))
    .with_columns([
        (pl.col('n_infield_tbins') - pl.col('n_infield_spike_tbins')).alias('n_infield_nonspike_tbins'),
        (pl.col('n_outfield_tbins') - pl.col('n_outfield_spike_tbins')).alias('n_outfield_nonspike_tbins'),
        (pl.col('true_pos_n_spikes') + pl.col('false_pos_n_spikes')).alias('n_total_spikes'),
    ])
    .with_columns([true_pos / false_pos / true_neg / false_neg rates using new names])
    .sort('neuron_IDX')
)
```

Temps removed vs current: separate `known_keys`, mutable `base`, multi-stage dual-named intermediates, and the final `drop(columns=...)`.

Update the docstring Notes/Returns to use the new names (`true_neg = n_outfield_nonspike_tbins / n_outfield_tbins`, etc.).

## Out of scope

- No notebook edits
- No changes to `_prepare_visit_polars_frames` or `perform_compute_position_dependent_reliability_maps`