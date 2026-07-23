---
title: 2D laps computation
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/d217b9a5-f7df-4472-92fc-9760f6a9e228
url: https://phohale.sourcegraph.app/deepsearch/335
created: 2026-02-09T22:40:47Z
updated: 2026-07-22T17:43:51Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/CommanderPho/pyPhoCoreHelpers github.com/CommanderPho/Spike3D find where Bapun 2D laps are computed

Bapun 2D laps are computed through the following code locations:

## Main Entry Points

1. **[plot_laps_2d() in laps.py:219](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py?L219)**
   - This plotting function ensures laps are computed via `sess.compute_position_laps()`

2. **[LapsVisualizationMixin._compute_laps_specific_position_dfs() in LapsVisualizationMixin.py:29-38](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/LapsVisualizationMixin.py?L29-38)**
   - Computes lap-specific position DataFrames
   - Line 38: Calls `session.compute_position_laps()`
   - Line 39: Groups position data by lap ID to create per-lap DataFrames

## Core Computation

The actual lap computation happens in the **NeuroPy library** (external dependency), not in these three repositories. The key method is:

- `DataSession.compute_position_laps()` - This is imported from `neuropy.core.session.dataSession`

## Data Flow

After `compute_position_laps()` executes:
1. The position DataFrame gets a `'lap'` column added with lap IDs
2. Data is grouped by lap: `curr_position_df.groupby('lap').get_group(i)` for each lap ID
3. Specific columns extracted per lap: `['t','x','y','lin_pos']`
4. Results stored in `lap_specific_position_dfs` list

## Usage Examples

The lap computation is triggered in various places:
- [Computation.py:3608](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py?L3608) - Pipeline stage
- [LongShortTrackComputations.py:1636](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py?L1636) - Via `replace_session_laps_with_estimates()`

The actual lap detection algorithm (determining lap start/stop times from position data) is implemented in the NeuroPy library, which is not part of these three repositories.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D Bapun](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20Bapun)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D Bapun 2D laps compute](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20Bapun%202D%20laps%20compute)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/pyPhoCoreHelpers OR repo:github.com/CommanderPho/Spike3D Bapun 2D laps](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20Bapun%202D%20laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ plot_laps_2d](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20plot_laps_2d)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ compute position laps estimate](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20compute%20position%20laps%20estimate)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ compute_position_laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20compute_position_laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ def compute_position_laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20def%20compute_position_laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ estimation_session_laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20estimation_session_laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ replace_session_laps_with_estimates](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20replace_session_laps_with_estimates)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py@26f673d?L200-250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py?L200-250)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ class DataSession compute_position_laps laps](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20class%20DataSession%20compute_position_laps%20laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ BapunDataSessionFormat](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20BapunDataSessionFormat)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ _compute_laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20_compute_laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ def _compute_laps_specific_position](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20def%20_compute_laps_specific_position)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/LapsVisualizationMixin.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/LapsVisualizationMixin.py)

## Suggested Follow-ups

- Can you provide more details on the specific lap detection algorithm used in the NeuroPy library within the `compute_position_laps()` function?
- How are the lap-specific position DataFrames used in the `plot_laps_2d()` function for visualization?

