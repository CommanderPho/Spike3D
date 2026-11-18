---
title: Context decoder performance comparison for Spike3D and pyPhoPlaceCellAnalysis
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/aaeedef2-243d-4ab8-8e33-99ff67e564af
url: https://phohale.sourcegraph.app/deepsearch/377
created: 2026-06-10T05:51:19Z
updated: 2026-06-10T05:59:29Z
---

# context decoder performance comparison github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoPlaceCellAnalysis 

Here is a comprehensive overview of how **context decoder performance comparison** is implemented across your codebase.

---

## Context Decoder Performance Comparison

### Core Concepts

The system compares **four directional decoders** (long-LR, long-RL, short-LR, short-RL) against each other and against ground-truth lap data to evaluate how well a decoder discriminates track identity (long vs. short) and direction (LR vs. RL).

---

### Key Data Structures

**[`CompleteDecodedContextCorrectness`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4052-4055)** — the top-level result container:
```python
@define(slots=False, repr=False, eq=False)
class CompleteDecodedContextCorrectness(UnpackableMixin, object):
    correctness_arrays_tuple: Tuple   # per-lap boolean arrays
    percent_correct_tuple: Tuple      # (track_ID_correct, dir_correct, complete_correct)
```

**[`MeasuredDecodedPositionComparison`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4057-4062)** — compares actual vs. decoded positions:
```python
class MeasuredDecodedPositionComparison:
    measured_positions_dfs_list: List[pd.DataFrame]
    decoded_positions_df_list: List[pd.DataFrame]
    decoded_measured_diff_df: pd.DataFrame
```

**[`CustomDecodeEpochsResult`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4063)** — wraps a full decoding run with position comparison.

---

### Primary Performance Evaluation Functions

#### 1. `_check_result_laps_epochs_df_performance` — ground-truth lap scoring
[Lines 5820–5851](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5820-5851)

Takes a `result_laps_epochs_df` and computes three metrics:
- `percent_laps_track_identity_estimated_correctly` — was long vs. short decoded right?
- `percent_laps_direction_estimated_correctly` — was LR vs. RL decoded right?
- `percent_laps_estimated_correctly` — both properties simultaneously correct

Compares `maze_id` and `is_LR_dir` columns against `is_most_likely_track_identity_Long` / `is_most_likely_direction_LR` posterior-derived columns.

#### 2. `_perform_variable_time_bin_lap_groud_truth_performance_testing` — sweep over time bin sizes
[Lines 5855–5951](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5855-5951)

Makes a copy of `DirectionalMergedDecoders`, optionally filters to a subset of neurons (`included_neuron_ids`), re-decodes laps at a given `desired_laps_decoding_time_bin_size`, then calls `_check_result_laps_epochs_df_performance`. Returns `(alt_directional_merged_decoders_result, result_laps_epochs_df, CompleteDecodedContextCorrectness)`.

#### 3. `perform_sweep_lap_groud_truth_performance_testing` — sweep over neuron subsets
[Lines 5958–6008](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5958-6008)

Iterates over a list of `included_neuron_ids` (e.g. long-exclusive, short-exclusive, both, either, XOR, neither cell subsets) and runs the above test for each, producing a `percent_laps_correctness_df` with columns `("track_ID_correct", "dir_correct", "complete_correct")`. Useful for isolating which cell populations contribute most to context discrimination.

#### 4. `_perform_run_rigorous_decoder_performance_assessment` — train/test split evaluation
[Lines 10343–10407](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L10343-10407)

Uses `_do_train_test_split_decode_and_evaluate` to do a proper held-out evaluation. Ensures `TrainTestSplit` is computed, then decodes on the test set only. Returns the full suite of results including `test_decoder_results_dict`, `train_decoded_results_dict`, and `train_decoded_measured_diff_df_dict`.

Called from the batch pipeline in [`compute_and_export_session_trial_by_trial_performance_completion_function`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py?L1009-1012).

---

### Decoder Posterior Marginalization for Context Scores

[`DirectionalPlacefieldGlobalComputationFunctions`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3640-3647) computes the context-marginal probabilities used in all downstream comparisons:

```python
extracted_merged_scores_df['P_Long']  = P_decoder_long_LR  + P_decoder_long_RL
extracted_merged_scores_df['P_Short'] = P_decoder_short_LR + P_decoder_short_RL
extracted_merged_scores_df['P_LR']    = P_decoder_long_LR  + P_decoder_short_LR
extracted_merged_scores_df['P_RL']    = P_decoder_long_RL  + P_decoder_short_RL
```

---

### GenericDecoderDictDecodedEpochsDictResult — generalized multi-context decoding

[`context_dependent.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py) provides the fully generic result container used by the newer `EpochComputationFunctions` path. It supports arbitrary decoder dicts (not just the hardcoded long/short directional ones) and is the foundation for the Bapun-session contextual `pf2D` decoder ([`build_contextual_pf2D_decoder`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L544-580)).

---

### Summary of the Comparison Pipeline

```
TrackTemplates (4 decoders: long_LR, long_RL, short_LR, short_RL)
        ↓
DirectionalPseudo2DDecodersResult.perform_compute_marginals()
        ↓
P_Long, P_Short, P_LR, P_RL per time bin / epoch
        ↓
_check_result_laps_epochs_df_performance()
        → % track identity correct
        → % direction correct
        → % both correct  (CompleteDecodedContextCorrectness)
        ↓
perform_sweep_lap_groud_truth_performance_testing()   (over cell subsets)
_perform_run_rigorous_decoder_performance_assessment() (train/test split)
```

## Sources

- [repo:^github.com/CommanderPho/Spike3D context decoder performance comparison](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D%20context%20decoder%20performance%20comparison)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis context decoder performance comparison](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20context%20decoder%20performance%20comparison)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis decoder performance score accuracy long short](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20decoder%20performance%20score%20accuracy%20long%20short)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis context decoder performance](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20context%20decoder%20performance)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis _perform_run_rigorous_decoder_performance_assessment](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20_perform_run_rigorous_decoder_performance_assessment)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis _check_result_laps_epochs_df_performance CompleteDecodedContextCorrectness](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20_check_result_laps_epochs_df_performance%20CompleteDecodedContextCorrectness)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L4052-4070](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4052-4070)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L5820-5910](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5820-5910)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L10343-10450](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L10343-10450)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis perform_sweep_lap_groud_truth_performance_testing](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20perform_sweep_lap_groud_truth_performance_testing)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis context_dependent decoder comparison](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20context_dependent%20decoder%20comparison)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py@26f673d?L1-80](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py?L1-80)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L5958-6020](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5958-6020)


---

# Write a complete function that takes a Bapun Two-maze style `curr_active_pipeline`, uses the maze_epoch_names=['maze1', 'maze2'] to build a context decoder, builds performance comparison metrics similar to the above during the laps for each maze and returns the evaluation

Now I have everything needed. Here is the complete function:

```python
# ============================================================================================================================ #
# Bapun Two-Maze Context Decoder Performance Evaluation
# ============================================================================================================================ #

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from attrs import define, field

import numpy as np
import pandas as pd

from pyphocorehelpers.function_helpers import function_attributes
from neuropy.analyses.placefields import PfND
from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
from neuropy.core.laps import Laps
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from neuropy.utils.result_context import IdentifyingContext

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import (
    BasePositionDecoder,
    DecodedFilterEpochsResult,
)
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import (
    CompleteDecodedContextCorrectness,
    DecodedContextCorrectnessArraysTuple,
    PercentDecodedContextCorrectnessTuple,
    DirectionalPseudo2DDecodersResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@define(slots=False, repr=False, eq=False)
class BapunContextDecoderPerformanceResult:
    """Holds all artefacts produced by
    ``evaluate_bapun_context_decoder_performance``.

    Fields
    ------
    maze_epoch_names : list[str]
        The two epoch names that were used as contexts (e.g. ``['maze1', 'maze2']``).
    pf2D_Decoder_dict : dict[str, BasePositionDecoder]
        One per-context 2-D place-field decoder, keyed by maze epoch name.
    contextual_pf2D : PfND
        The merged pseudo-context PfND built from the two per-maze pf2Ds.
    contextual_pf2D_Decoder : BasePositionDecoder
        The merged decoder used to decode laps.
    per_maze_laps_decoder_result : dict[str, DecodedFilterEpochsResult]
        Raw ``decode_specific_epochs`` result for the lap epochs of each maze.
    per_maze_laps_marginals_df : dict[str, pd.DataFrame]
        Per-epoch context-marginal summary df for each maze
        (columns: ``lap_idx``, ``lap_start_t``, ``P_maze1``, ``P_maze2``,
        ``is_most_likely_context_maze1``).
    per_maze_context_correctness : dict[str, CompleteDecodedContextCorrectness]
        ``CompleteDecodedContextCorrectness`` object for each maze.
    combined_laps_df : pd.DataFrame
        All lap epochs concatenated with ground-truth and decoded columns,
        plus ``source_maze`` and ``is_context_correct`` convenience columns.
    overall_percent_correct : float
        Fraction of laps across both mazes where the most-likely decoded
        context matches the actual maze that generated the lap.
    """
    maze_epoch_names: List[str] = field()
    pf2D_Decoder_dict: Dict[str, BasePositionDecoder] = field()
    contextual_pf2D: PfND = field()
    contextual_pf2D_Decoder: BasePositionDecoder = field()
    per_maze_laps_decoder_result: Dict[str, DecodedFilterEpochsResult] = field()
    per_maze_laps_marginals_df: Dict[str, pd.DataFrame] = field()
    per_maze_context_correctness: Dict[str, "CompleteDecodedContextCorrectness"] = field()
    combined_laps_df: pd.DataFrame = field()
    overall_percent_correct: float = field()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_bapun_context_marginals_df(
    decoder_result: DecodedFilterEpochsResult,
    maze_epoch_names: List[str],
) -> pd.DataFrame:
    """Compute a per-epoch marginal posterior df over the two contexts.

    The pseudo-context decoder stacks the two per-maze pf2Ds along the
    *second* spatial axis (context axis), so marginalising over x and y
    gives a (n_contexts, n_time_bins) posterior.  We sum time bins inside
    each epoch to get one probability per epoch per context.

    Returns a DataFrame with columns:
        ``lap_idx``, ``lap_start_t``, ``<maze1_name>``, ``<maze2_name>``,
        ``is_most_likely_context_<maze1_name>``
    """
    ctx_name_0, ctx_name_1 = maze_epoch_names  # e.g. 'maze1', 'maze2'

    n_epochs = decoder_result.num_filter_epochs
    p_ctx0_per_epoch: List[float] = []
    p_ctx1_per_epoch: List[float] = []

    for epoch_idx in range(n_epochs):
        p_x_given_n = decoder_result.p_x_given_n_list[epoch_idx]
        # p_x_given_n.shape: (n_xbins, n_ybins, n_contexts=2, n_tbins)
        # Marginalise over spatial dimensions → (n_contexts, n_tbins)
        marginal_ctx = np.nansum(p_x_given_n, axis=(0, 1))          # (2, n_tbins)
        # Normalise each time bin then average across time bins per epoch
        col_sums = np.nansum(marginal_ctx, axis=0, keepdims=True)    # (1, n_tbins)
        col_sums[col_sums == 0] = 1.0                                # guard /0
        marginal_ctx_norm = marginal_ctx / col_sums                  # (2, n_tbins)
        epoch_mean = np.nanmean(marginal_ctx_norm, axis=1)           # (2,)
        p_ctx0_per_epoch.append(float(epoch_mean[0]))
        p_ctx1_per_epoch.append(float(epoch_mean[1]))

    filter_epochs_df = ensure_dataframe(decoder_result.filter_epochs)
    marginals_df = pd.DataFrame({
        'lap_idx':                             np.arange(n_epochs),
        'lap_start_t':                         filter_epochs_df['start'].to_numpy(),
        'start':                               filter_epochs_df['start'].to_numpy(),
        'stop':                                filter_epochs_df['stop'].to_numpy(),
        'duration':                            filter_epochs_df['duration'].to_numpy(),
        ctx_name_0:                            np.array(p_ctx0_per_epoch),
        ctx_name_1:                            np.array(p_ctx1_per_epoch),
    })
    marginals_df['is_most_likely_context_' + ctx_name_0] = (
        marginals_df[ctx_name_0] >= marginals_df[ctx_name_1]
    )
    return marginals_df


def _check_bapun_context_correctness(
    marginals_df: pd.DataFrame,
    true_maze_name: str,
    maze_epoch_names: List[str],
) -> "CompleteDecodedContextCorrectness":
    """Compute correctness arrays and percentages for one maze's laps.

    ``true_maze_name`` is the ground-truth context (e.g. ``'maze1'``).
    A lap is correct if the most-likely decoded context matches
    ``true_maze_name``.
    """
    ctx_name_0 = maze_epoch_names[0]
    is_true_context_maze0: bool = (true_maze_name == ctx_name_0)

    is_context_correct: np.ndarray = (
        marginals_df['is_most_likely_context_' + ctx_name_0].to_numpy()
        if is_true_context_maze0
        else ~marginals_df['is_most_likely_context_' + ctx_name_0].to_numpy()
    )
    n_laps = len(marginals_df)
    percent_correct = float(np.sum(is_context_correct)) / n_laps

    # Re-use the existing named tuple types so downstream code is compatible
    correctness_arrays = DecodedContextCorrectnessArraysTuple(
        is_decoded_track_correct=is_context_correct,
        is_decoded_dir_correct=np.ones(n_laps, dtype=bool),   # N/A for context-only
        are_both_decoded_properties_correct=is_context_correct,
    )
    percent_tuple = PercentDecodedContextCorrectnessTuple(
        percent_laps_track_identity_estimated_correctly=percent_correct,
        percent_laps_direction_estimated_correctly=1.0,        # N/A
        percent_laps_estimated_correctly=percent_correct,
    )
    return CompleteDecodedContextCorrectness(
        correctness_arrays_tuple=correctness_arrays,
        percent_correct_tuple=percent_tuple,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────────

@function_attributes(
    short_name=None,
    tags=['IMPORTANT', 'bapun', 'two-maze', 'context-decoding', 'performance', 'WORKING'],
    input_requires=[],
    output_provides=[],
    uses=[
        'build_contextual_pf2D_decoder',
        'PfND.build_merged_directional_placefields',
        'BasePositionDecoder.decode_specific_epochs',
        'DirectionalPseudo2DDecodersResult._check_result_laps_epochs_df_performance',
    ],
    used_by=[],
    creation_date='2026-06-10 00:00',
    related_items=[
        'build_contextual_pf2D_decoder',
        'decode_using_contextual_pf2D_decoder',
        'LapDecodingGroundTruth._perform_variable_time_bin_lap_groud_truth_performance_testing',
    ],
)
def evaluate_bapun_context_decoder_performance(
    curr_active_pipeline,
    maze_epoch_names: List[str] = None,
    laps_decoding_time_bin_size: float = 0.5,
    included_neuron_IDs: Optional[np.ndarray] = None,
    debug_print: bool = False,
) -> BapunContextDecoderPerformanceResult:
    """Build a two-context (Bapun-style) pseudo-2D decoder from the per-maze
    pf2Ds and evaluate how well it recovers maze identity during laps.

    The function mirrors the logic of
    :func:`LapDecodingGroundTruth._perform_variable_time_bin_lap_groud_truth_performance_testing`
    but is self-contained and works with the arbitrary ``maze_epoch_names``
    that a Bapun two-maze pipeline exposes instead of the hard-coded
    ``long / short`` directional nomenclature.

    Parameters
    ----------
    curr_active_pipeline:
        A fully-computed pipeline whose ``computation_results`` contain
        ``pf2D_Decoder`` entries for every name in ``maze_epoch_names``,
        and whose ``filtered_sessions`` contain ``.laps`` for each maze.
    maze_epoch_names:
        The two filtered-session / computation-result keys that correspond
        to the two mazes.  Defaults to ``['maze1', 'maze2']``.
    laps_decoding_time_bin_size:
        Desired time-bin size in seconds.  Will be clamped to the minimum
        lap duration found in each maze (same logic as
        ``_perform_variable_time_bin_lap_groud_truth_performance_testing``).
    included_neuron_IDs:
        Optional subset of ACLUs.  When provided the per-maze decoders are
        restricted via ``.get_by_id(...)`` before being merged.
    debug_print:
        Print intermediate progress information.

    Returns
    -------
    BapunContextDecoderPerformanceResult
        All intermediate and final artefacts (decoders, raw decoding results,
        per-maze correctness, combined summary df, and overall accuracy).

    Usage
    -----
    .. code-block:: python

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import (
            evaluate_bapun_context_decoder_performance,
        )

        result = evaluate_bapun_context_decoder_performance(
            curr_active_pipeline,
            maze_epoch_names=['maze1', 'maze2'],
            laps_decoding_time_bin_size=0.5,
        )

        print(f"Overall context-correct: {result.overall_percent_correct:.1%}")
        for maze_name, correctness in result.per_maze_context_correctness.items():
            pct = correctness.percent_correct_tuple.percent_laps_track_identity_estimated_correctly
            print(f"  {maze_name}: {pct:.1%}")

        result.combined_laps_df  # full per-lap summary DataFrame
    """
    if maze_epoch_names is None:
        maze_epoch_names = ['maze1', 'maze2']

    assert len(maze_epoch_names) == 2, (
        f"exactly two maze_epoch_names are required, got {maze_epoch_names}"
    )

    # ── 1. Build one pf2D_Decoder per maze ────────────────────────────────────
    pf2D_Decoder_dict: Dict[str, BasePositionDecoder] = {
        name: deepcopy(curr_active_pipeline.computation_results[name].computed_data.pf2D_Decoder)
        for name in maze_epoch_names
    }

    # Optional: restrict to the requested neuron subset
    if included_neuron_IDs is not None:
        pf2D_Decoder_dict = {
            name: dec.get_by_id(included_neuron_IDs)
            for name, dec in pf2D_Decoder_dict.items()
        }
        if debug_print:
            print(f"Restricted decoders to {len(included_neuron_IDs)} neurons.")

    # ── 2. Conform bin grids and merge into one pseudo-context pf2D ───────────
    # Replicate the bin-conforming loop from build_contextual_pf2D_decoder so
    # that both per-maze spatial grids are identical before merging.
    contextual_pf2D_dict: Dict[str, PfND] = {
        name: deepcopy(dec.pf)
        for name, dec in pf2D_Decoder_dict.items()
    }
    reference_pf: Optional[PfND] = None
    for name, a_pf in contextual_pf2D_dict.items():
        if reference_pf is None:
            reference_pf = a_pf
        else:
            contextual_pf2D_dict[name], did_update = a_pf.conform_to_position_bins(reference_pf)
            if debug_print:
                print(f"  conform_to_position_bins({name}): did_update={did_update}")

    # PfND.build_merged_directional_placefields concatenates the tuning curves
    # along a new "context" axis — the same mechanism used for the directional
    # (long/short) decoder, now applied to maze1/maze2.
    contextual_pf2D: PfND = PfND.build_merged_directional_placefields(
        contextual_pf2D_dict, debug_print=debug_print
    )
    contextual_pf2D_Decoder: BasePositionDecoder = BasePositionDecoder(
        contextual_pf2D, setup_on_init=True, post_load_on_init=True, debug_print=False
    )
    if debug_print:
        print(
            f"Merged context decoder: {contextual_pf2D_Decoder.pf.ratemap.n_neurons} neurons, "
            f"context axis size = {contextual_pf2D.xbin.shape}"
        )

    # ── 3. Decode laps for each maze individually ──────────────────────────────
    global_spikes_df = deepcopy(curr_active_pipeline.sess.spikes_df)

    per_maze_laps_decoder_result: Dict[str, DecodedFilterEpochsResult] = {}
    per_maze_laps_marginals_df: Dict[str, pd.DataFrame] = {}
    per_maze_context_correctness: Dict[str, CompleteDecodedContextCorrectness] = {}
    per_maze_combined_rows: List[pd.DataFrame] = []

    for maze_name in maze_epoch_names:
        if debug_print:
            print(f"\n── Decoding laps for {maze_name} ──")

        # 3a. Get the lap epochs for this maze's filtered session
        filtered_sess = curr_active_pipeline.filtered_sessions[maze_name]
        laps_obj: Laps = deepcopy(filtered_sess.laps)
        laps_epoch_obj: Epoch = ensure_Epoch(laps_obj)

        # Clamp time-bin size to the shortest lap so we always get ≥1 bin/lap
        min_lap_duration: float = find_minimum_time_bin_duration(
            ensure_dataframe(laps_epoch_obj)['duration'].to_numpy()
        )
        effective_time_bin_size: float = min(laps_decoding_time_bin_size, min_lap_duration)
        if debug_print:
            print(
                f"  {maze_name}: min_lap_duration={min_lap_duration:.3f}s, "
                f"effective_time_bin_size={effective_time_bin_size:.3f}s"
            )

        # 3b. Decode
        a_decoder_result: DecodedFilterEpochsResult = (
            contextual_pf2D_Decoder.decode_specific_epochs(
                spikes_df=deepcopy(global_spikes_df),
                filter_epochs=laps_epoch_obj,
                decoding_time_bin_size=effective_time_bin_size,
                debug_print=False,
            )
        )
        per_maze_laps_decoder_result[maze_name] = a_decoder_result

        # 3c. Build a per-epoch context marginal summary df
        marginals_df: pd.DataFrame = _build_bapun_context_marginals_df(
            decoder_result=a_decoder_result,
            maze_epoch_names=maze_epoch_names,
        )
        per_maze_laps_marginals_df[maze_name] = marginals_df

        # 3d. Evaluate correctness (the true context IS this maze)
        correctness: CompleteDecodedContextCorrectness = _check_bapun_context_correctness(
            marginals_df=marginals_df,
            true_maze_name=maze_name,
            maze_epoch_names=maze_epoch_names,
        )
        per_maze_context_correctness[maze_name] = correctness

        pct = correctness.percent_correct_tuple.percent_laps_track_identity_estimated_correctly
        if debug_print:
            n_laps = len(marginals_df)
            print(f"  {maze_name}: {int(round(pct * n_laps))}/{n_laps} laps correct ({pct:.1%})")

        # 3e. Annotate and collect for the combined df
        row_df = marginals_df.copy()
        row_df['source_maze'] = maze_name
        row_df['true_context_is_maze0'] = (maze_name == maze_epoch_names[0])
        row_df['is_context_correct'] = correctness.correctness_arrays_tuple.is_decoded_track_correct
        per_maze_combined_rows.append(row_df)

    # ── 4. Combined summary across both mazes ─────────────────────────────────
    combined_laps_df = pd.concat(per_maze_combined_rows, axis='index', ignore_index=True)

    overall_percent_correct: float = float(
        combined_laps_df['is_context_correct'].mean()
    )
    if debug_print:
        n_total = len(combined_laps_df)
        n_correct = int(combined_laps_df['is_context_correct'].sum())
        print(
            f"\nOverall: {n_correct}/{n_total} laps decoded correctly "
            f"({overall_percent_correct:.1%})"
        )

    return BapunContextDecoderPerformanceResult(
        maze_epoch_names=maze_epoch_names,
        pf2D_Decoder_dict=pf2D_Decoder_dict,
        contextual_pf2D=contextual_pf2D,
        contextual_pf2D_Decoder=contextual_pf2D_Decoder,
        per_maze_laps_decoder_result=per_maze_laps_decoder_result,
        per_maze_laps_marginals_df=per_maze_laps_marginals_df,
        per_maze_context_correctness=per_maze_context_correctness,
        combined_laps_df=combined_laps_df,
        overall_percent_correct=overall_percent_correct,
    )
```

---

### Design notes

#### How this mirrors existing infrastructure

| Step | Analogous existing code |
|---|---|
| Per-maze `pf2D_Decoder` extraction | [`build_contextual_pf2D_decoder`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L553-580) |
| `conform_to_position_bins` + `build_merged_directional_placefields` | Same function, lines [565–575](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L565-575) |
| Time-bin clamping via `find_minimum_time_bin_duration` | [`_perform_variable_time_bin_lap_groud_truth_performance_testing`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5922-5929) |
| `decode_specific_epochs` per maze | Same call in `_perform_variable_time_bin_lap_groud_truth_performance_testing`, line [5929](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5929) |
| Per-epoch context marginals | [`_build_bapun_context_marginals_df`] mirrors the `marginal_z` logic in [`decode_using_contextual_pf2D_decoder`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L636-652) |
| Correctness check | [`_check_result_laps_epochs_df_performance`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5820-5851), producing `CompleteDecodedContextCorrectness` |
| `included_neuron_IDs` subsetting | [`_perform_variable_time_bin_lap_groud_truth_performance_testing`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5885-5906) |

#### Key differences from the long/short pipeline

- No `add_laps_groundtruth_information_to_dataframe` call — that function calls `find_LongShortDelta_times()` and assumes `maze_id ∈ {0,1}` meaning long/short. For Bapun sessions the ground truth is simpler: the lap epochs obtained from `filtered_sessions[maze_name].laps` *already are* the maze1 or maze2 laps, so the true context is the maze name itself.
- The "context axis" in the posterior is size-2 (maze1, maze2) rather than size-4 (long_LR, long_RL, short_LR, short_RL), matching the 2-decoder case detected by `is_track_identity_only_pseudo2D_decoder` in [`perform_compute_specific_marginals`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1682-1690).
- `DecodedContextCorrectnessArraysTuple.is_decoded_dir_correct` is set to all-True as direction is not being discriminated; only context identity is evaluated.

## Sources

- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L544-700](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L544-700)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py@26f673d?L80-300](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py?L80-300)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L5910-5960](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L5910-5960)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis build_merged_directional_placefields pf2D_Decoder decode_specific_epochs laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20build_merged_directional_placefields%20pf2D_Decoder%20decode_specific_epochs%20laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis add_groundtruth_information maze_id is_LR_dir laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20add_groundtruth_information%20maze_id%20is_LR_dir%20laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis perform_compute_marginals laps_directional_marginals laps_track_identity_marginals](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20perform_compute_marginals%20laps_directional_marginals%20laps_track_identity_marginals)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L100-166](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L100-166)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L1621-1720](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1621-1720)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L1720-1830](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1720-1830)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis find_minimum_time_bin_duration laps as_epoch_obj](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20find_minimum_time_bin_duration%20laps%20as_epoch_obj)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis find_minimum_time_bin_duration import](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20find_minimum_time_bin_duration%20import)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis computation_results pf2D_Decoder maze1 maze2 filtered_sessions laps](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20computation_results%20pf2D_Decoder%20maze1%20maze2%20filtered_sessions%20laps)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis compute_marginals epoch_idx_col_name DecodedFilterEpochsResult](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20compute_marginals%20epoch_idx_col_name%20DecodedFilterEpochsResult)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L1831-1920](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1831-1920)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L4063-4110](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L4063-4110)

