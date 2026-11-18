---
name: Fix replay stats merge
overview: Restore the two-frame merge in `build_neuron_identities_df_for_CSV` so peak columns from `_neuron_replay_stats_df` are joined onto the identity table. Git history shows the self-merge was an accidental rename, not an intentional drop.
todos:
  - id: restore-merge
    content: Restore the two-frame first merge in build_neuron_identities_df_for_CSV and remove the overwrite TODO
    status: completed
isProject: false
---

# Fix replay-stats column merge

The self-merge in [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py) was not intentional.

- `799854fea` (2025-06-09, `build_neuron_identities_df_for_CSV`) introduced the correct two-frame merge: accumulator `all_neuron_stats_table = deepcopy(unique_neuron_identities_df)`, then `pd.merge(all_neuron_stats_table, _neuron_replay_stats_df[...])`.
- `5ac92f58a` (2025-07-08) added the decoder-spike merge in the same pattern.
- `23278acee` (2025-07-31, `cleanup, changes to stability_df`) renamed the accumulator from `all_neuron_stats_table` to `_neuron_replay_stats_df`. That clobbered the peak-enriched frame and made the first merge compare a dataframe to itself. The commit message is about `stability_df`, not about dropping peak columns.

Keep the later accumulator name `_neuron_replay_stats_df` (stability merge and `return` already use it). Only change the identity assignment and the first merge so the peak frame is still the right-hand side when the merge runs. Leave the inner join as in the last known-good version.

Replace lines 1073 and 1078:

```python
all_neuron_stats_table: pd.DataFrame = deepcopy(unique_neuron_identities_df)
_neuron_replay_stats_df = pd.merge(all_neuron_stats_table, _neuron_replay_stats_df[['neuron_uid'] + [col for col in _neuron_replay_stats_df.columns if col not in all_neuron_stats_table.columns and col != 'neuron_uid']], on='neuron_uid')
```

Drop the TODO on the current line 1073; it describes the bug being removed. Leave the following merges (firing-rate index, rate remapping, decoder spikes, stability) unchanged.