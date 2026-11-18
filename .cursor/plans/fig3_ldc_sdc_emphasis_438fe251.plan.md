---
name: Fig3 LdC SdC emphasis
overview: Reuse the large Figure-3 FRI scatter’s LxC/SxC edgecolor styling inside `across_sessions_long_and_short_firing_rate_replays_v_laps_figure` so the long and short rate plots highlight the same cells identically.
todos:
  - id: style-small-plots
    content: In across_sessions_long_and_short_firing_rate_replays_v_laps_figure, after each dropna build LxC/SxC edgecolors (matching large FRI) and pass per-track to _plot_single_track_firing_rate_compare
    status: completed
isProject: false
---

# Add LdC/SdC emphasis to Figure 3 small plots

## Context

Figure 3 is built in the notebook via two `AcrossSessionsVisualizations` calls:

- Large FRI scatter: `across_sessions_firing_rate_index_figure` (already highlights LdC/SdC)
- Small long/short rate scatters: `across_sessions_long_and_short_firing_rate_replays_v_laps_figure` (currently uniform gray)

Membership and style already live in [`AcrossSessionResults.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py):

- Column: `active_set_membership_from_user_annotations` (`'LxC'` = LdC, `'SxC'` = SdC)
- Face: `#33333333` (transparent gray)
- Edges: LxC `#0000B3`, SxC `#9C0000`, others `#33333300`

The notebook input (`long_short_fr_indicies_analysis_table` from `across_session_inst_fr_computation_df`) already has that column, so **no notebook edit is required**.

## Change

Edit only [`across_sessions_long_and_short_firing_rate_replays_v_laps_figure`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py) (~4057–4116).

After each track’s `dropna(subset=...)`, build per-row `point_colors` / `edgecolors` on the **filtered** frame (same order as the `{index: value}` dicts passed to `_plot_single_track_firing_rate_compare`), then pass them into that plot call:

```python
# same defaults/colors as across_sessions_firing_rate_index_figure
neuron_replay_stats_df = neuron_replay_stats_df.copy()
neuron_replay_stats_df['point_colors'] = '#33333333'
neuron_replay_stats_df['edgecolors'] = '#33333300'
if 'active_set_membership_from_user_annotations' in neuron_replay_stats_df:
    neuron_replay_stats_df.loc[neuron_replay_stats_df['active_set_membership_from_user_annotations'] == 'LxC', 'edgecolors'] = '#0000B3'
    neuron_replay_stats_df.loc[neuron_replay_stats_df['active_set_membership_from_user_annotations'] == 'SxC', 'edgecolors'] = '#9C0000'

track_scatter_kwargs = common_scatter_kwargs | dict(
    point_colors=neuron_replay_stats_df['point_colors'].to_numpy(),
    edgecolors=neuron_replay_stats_df['edgecolors'].to_numpy(),
    zorder=5,
)
```

Apply this independently for long and short (they drop different NaN rows, so arrays must not be shared).

`_plot_single_track_firing_rate_compare` already forwards `edgecolors` / `zorder` via `ax.scatter(..., **scatter_params)`, so no change is needed in [`LongShortTrackComparingDisplayFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/MultiContextComparingDisplayFunctions/LongShortTrackComparingDisplayFunctions.py).

## Verify

Re-run the Figure 3 notebook cell: long and short panels should show the same blue/red-edged LxC/SxC markers as the large FRI plot, with other cells remaining gray.