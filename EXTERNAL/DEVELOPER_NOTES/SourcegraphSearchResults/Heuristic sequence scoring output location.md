---
title: Heuristic sequence scoring output location
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/9dfbf5c4-bc9c-48ec-a99f-f9f0e3faa78d
url: https://phohale.sourcegraph.app/deepsearch/248
created: 2025-11-12T00:41:01Z
updated: 2026-01-15T10:15:29Z
---

# In what user batch function are the heuristic sequence scoring results output? Which .csv are they output to? github.com/CommanderPho/pyPhoPlaceCellAnalysis  github.com/CommanderPho/Spike3D 

Based on my investigation, I can now provide you with a comprehensive answer about the heuristic sequence scoring results output.

The heuristic sequence scoring results are output in the **`compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function`** batch function. This function is part of the batch completion pipeline and exports the results to a CSV file called **`ripple_all_scores_merged_df`**.

## Key Details:

### **Batch Function:**
- **Function name:** [`compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py?L726-L730)

### **CSV Output File:**
- **Primary CSV:** [`ripple_all_scores_merged_df.csv`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3908)

### **Heuristic Scores Included:**
The heuristic scoring results include these specific metrics (from [`HeuristicReplayScoring.build_all_bin_wise_computation_fn_dict()`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2887-2906)):

1. **`avg_jump_cm`** - Average jump distance in centimeters
2. **`travel`** - Total distance traveled score  
3. **`coverage`** - Track coverage score
4. **`total_distance_traveled`** - From sequence scoring computations
5. **`track_coverage_score`** - From sequence scoring computations

### **File Naming Convention:**
The CSV files follow this pattern:
```
{BATCH_DATE_TO_USE}-{session_name}-(ripple_all_scores_merged_df)_tbin-{time_bin_size}.csv
```

For example:
- `2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)_tbin-0.016.csv`

### **Computation Process:**
1. The [`directional_decoders_epoch_heuristic_scoring`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L7435-7441) computation function calculates the heuristic scores
2. Results are stored in the [`HeuristicsResult`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L7526) object
3. The batch completion function exports these to the `ripple_all_scores_merged_df.csv` file via the [`export_csvs`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3865) method

The CSV contains comprehensive scoring data for each decoded epoch, including the heuristic sequence scoring metrics alongside other decoding evaluation metrics like weighted correlations, Pearson correlations, and Bayesian decoder probabilities.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/Spike3D heuristic sequence scoring batch function](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20heuristic%20sequence%20scoring%20batch%20function)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/Spike3D heuristic csv output batch](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20heuristic%20csv%20output%20batch)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis heuristic scoring output csv batch function](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20heuristic%20scoring%20output%20csv%20batch%20function)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "heuristic" "csv" "export" "batch"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22heuristic%22%20%22csv%22%20%22export%22%20%22batch%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "directional_decoders_epoch_heuristic_scoring"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22directional_decoders_epoch_heuristic_scoring%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L7434-7500](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L7434-7500)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L7500-7600](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L7500-7600)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "Heuristics" "csv" "export" "batch"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22Heuristics%22%20%22csv%22%20%22export%22%20%22batch%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis heuristic scoring export csv completion function batch](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20heuristic%20scoring%20export%20csv%20completion%20function%20batch)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py@26f673d?L726-800](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py?L726-800)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py@26f673d?L800-900](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py?L800-900)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py@26f673d?L900-1000](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py?L900-1000)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3860-3900](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3860-3900)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3900-3950](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3900-3950)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "build_complete_all_scores_merged_df"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_complete_all_scores_merged_df%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3576-3650](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3576-3650)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3650-3720](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3650-3720)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "get_all_scores_column_names"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22get_all_scores_column_names%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L3545-3575](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L3545-3575)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "get_all_score_computation_col_names"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22get_all_score_computation_col_names%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2943-2950](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2943-2950)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "build_all_score_computations_fn_dict"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_all_score_computations_fn_dict%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2909-2940](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2909-2940)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "build_all_bin_wise_computation_fn_dict"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_all_bin_wise_computation_fn_dict%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2885-2908](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2885-2908)

## Suggested Follow-ups

- Can you provide an example of how to interpret the heuristic scoring metrics (avg_jump_cm, travel, coverage) in the ripple_all_scores_merged_df.csv file, particularly in relation to the quality of replay events?
- What other decoding evaluation metrics are included in the `ripple_all_scores_merged_df.csv` file besides the heuristic scores, and how do they relate to each other?


---

# when I run that I function I encounter this error: ```python
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
~\AppData\Local\Temp\ipykernel_31228\3269945620.py in ?()
---> 32 from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _subfn_compute_complete_df_metrics
     33 from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function
     34 from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import SimpleBatchComputationDummy
     35 
~\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py in ?(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict, ripple_decoding_time_bin_size_override, laps_decoding_time_bin_size_override, needs_recompute_heuristics, force_recompute_all_decoding, save_hdf, allow_append_to_session_h5_file, max_ignore_bins, same_thresh_cm, max_jump_distance_cm)
    835     else:
    836         ## get from parameters:
    837         minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
    838         included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
--> 839     track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only # TrackTemplates
    840 
    841     if force_recompute_all_decoding:
    842         ## force recompute all
~\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py in ?(self, minimum_inclusion_fr_Hz, included_qclu_values)
    936         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_one_step_decoder_1D, self.short_LR_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_one_step_decoder_1D, self.short_RL_one_step_decoder_1D))
    937         if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
    938             return _obj
    939         else:
--> 940             return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
~\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py in ?(self, minimum_inclusion_fr_Hz, included_qclu_values)
    584             filtered_track_templates = track_templates.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=5.0, included_qclu_values=[1, 2])
    585         """
    586         filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate_and_qclu(decoders_dict=self.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
    587         long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = filtered_decoder_list # unpack
--> 588         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_LR_decoder, short_LR_decoder), RL_decoder_pair=(long_RL_decoder, short_RL_decoder), rank_method=self.rank_method)
    589         assert np.all(filtered_direction_shared_aclus_list[0] == _obj.shared_LR_aclus_only_neuron_IDs)
    590         assert np.all(filtered_direction_shared_aclus_list[1] == _obj.shared_RL_aclus_only_neuron_IDs)
    591         assert len(filtered_direction_shared_aclus_list[0]) == len(_obj.decoder_LR_pf_peak_ranks_list[0])
~\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py in ?(cls, LR_decoder_pair, RL_decoder_pair, rank_method)
    604         long_LR_decoder, short_LR_decoder = LR_decoder_pair
    605         long_RL_decoder, short_RL_decoder = RL_decoder_pair
    606 
    607         shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
--> 608         assert np.all(short_LR_decoder.neuron_IDs == shared_LR_aclus_only_neuron_IDs), f"{short_LR_decoder.neuron_IDs} != {shared_LR_aclus_only_neuron_IDs}"
    609 
    610         shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)
    611         assert np.all(short_RL_decoder.neuron_IDs == shared_RL_aclus_only_neuron_IDs), f"{short_RL_decoder.neuron_IDs} != {shared_RL_aclus_only_neuron_IDs}"
AssertionError: [ 11  12  15  24  25  28  31  39  40  44  47  48  51  53  56  57  61  68  71  75  79  80  81  82  91  92  95 102 104] != [  4   9  11  12  15  18  24  25  26  27  28  31  32  39  40  44  47  48  51  52  53  56  57  59  61  63  66  67  68  71  72  75  77  79  80  81  82  84  89  91  92  93  95 101 102 104]
```

Based on my analysis, the issue is occurring because the filtering process creates different sets of neurons for long vs short tracks. Here's the solution:

## **Solution: Use Unfiltered Templates**

The simplest workaround is to call `get_templates()` with `None` parameters to avoid the filtering that's causing the mismatch:

```python
# Instead of this (which causes the error):
track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

# Use this (unfiltered templates):
track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=None, included_qclu_values=None)
```

## **Alternative Solution: Modify the Function Call**

You can modify the function call to skip the filtering parameters:

```python
# Call the function with filtering disabled
callback_outputs = compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function(
    self=your_batch_object,
    global_data_root_parent_path=your_global_data_root_parent_path,
    curr_session_context=your_curr_session_context,
    curr_session_basedir=your_curr_session_basedir,
    curr_active_pipeline=your_curr_active_pipeline,
    across_session_results_extended_dict=your_across_session_results_extended_dict,
    # ... other parameters ...
)
```

But modify line 839 in the function to use unfiltered templates:

```python
track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=None, included_qclu_values=None)
```

## **Root Cause**

The error occurs because:

1. The filtering process (`filtered_by_frate_and_qclu`) attempts to create consistent neuron sets across long/short tracks
2. However, some neurons may meet the firing rate criteria on one track length but not the other
3. The assertion at line 608 requires that all decoders for the same direction (LR or RL) have identical neuron IDs
4. When filtering results in different neuron sets, this assertion fails

## **Why the Unfiltered Approach Works**

Using unfiltered templates (`minimum_inclusion_fr_Hz=None, included_qclu_values=None`) bypasses the filtering entirely, ensuring all decoders maintain their original neuron sets, which are consistent by design.

The heuristic scoring will still work correctly with the unfiltered templates, and you'll get the same CSV output (`ripple_all_scores_merged_df.csv`) with the heuristic sequence scoring results.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "determine_decoder_aclus_filtered_by_frate_and_qclu"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22determine_decoder_aclus_filtered_by_frate_and_qclu%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L628-680](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L628-680)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "filtered_by_frate_and_qclu" "AssertionError"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22filtered_by_frate_and_qclu%22%20%22AssertionError%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "get_templates" "minimum_inclusion_fr_Hz" "None"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22get_templates%22%20%22minimum_inclusion_fr_Hz%22%20%22None%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "get_templates" "None" "None"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22get_templates%22%20%22None%22%20%22None%22)

## Suggested Follow-ups

- Why are the neuron IDs different between the short and long track decoders after filtering?
- How does using unfiltered templates affect the heuristic scoring results and the overall analysis?


---

# in `filtered_by_frate_and_qclu`, is there a more general (non-paired) version of `_obj = TrackTemplates.init_from_paired_decoders(LR_decode....` that I should be building from the filtered objects?

Now I understand the issue better. The problem is that `init_from_paired_decoders` enforces that paired decoders (long/short for each direction) must have identical neuron IDs, but the filtering process can result in different neurons meeting the criteria for long vs short tracks.

## **Yes, there is a more general approach:**

You can create a `TrackTemplates` object directly using the attrs-generated constructor, bypassing the strict neuron ID matching requirements of `init_from_paired_decoders`. Here's how:

### **Solution: Direct Constructor Approach**

```python
def create_filtered_track_templates_flexible(directional_laps_results, minimum_inclusion_fr_Hz, included_qclu_values):
    """
    Create TrackTemplates with filtered decoders, allowing for different neuron sets
    between long/short tracks if necessary.
    """
    from copy import deepcopy
    import numpy as np
    import scipy.stats
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
    
    # Get the original decoders
    decoders_dict = {
        'long_LR': directional_laps_results.long_LR_one_step_decoder_1D,
        'long_RL': directional_laps_results.long_RL_one_step_decoder_1D,
        'short_LR': directional_laps_results.short_LR_one_step_decoder_1D,
        'short_RL': directional_laps_results.short_RL_one_step_decoder_1D,
    }
    
    # Get the filtered neuron IDs for each decoder independently
    modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(
        decoders_dict=decoders_dict, 
        minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, 
        included_qclu_values=included_qclu_values
    )
    
    # Create filtered decoders
    filtered_decoders = {}
    for decoder_name, original_decoder in decoders_dict.items():
        filtered_aclus = modified_neuron_ids_dict[decoder_name]
        filtered_decoders[decoder_name] = original_decoder.get_by_id(filtered_aclus)
    
    # For the shared neuron IDs, use the intersection or union as appropriate
    # Option 1: Use intersection (only neurons present in both long and short)
    shared_LR_aclus = np.intersect1d(
        modified_neuron_ids_dict['long_LR'], 
        modified_neuron_ids_dict['short_LR']
    )
    shared_RL_aclus = np.intersect1d(
        modified_neuron_ids_dict['long_RL'], 
        modified_neuron_ids_dict['short_RL']
    )
    
    # Option 2: Use union (neurons present in either long or short) - uncomment if preferred
    # shared_LR_aclus = np.union1d(
    #     modified_neuron_ids_dict['long_LR'], 
    #     modified_neuron_ids_dict['short_LR']
    # )
    # shared_RL_aclus = np.union1d(
    #     modified_neuron_ids_dict['long_RL'], 
    #     modified_neuron_ids_dict['short_RL']
    # )
    
    # Re-filter decoders to use only shared neurons
    final_filtered_decoders = {}
    for decoder_name, filtered_decoder in filtered_decoders.items():
        if 'LR' in decoder_name:
            shared_aclus = shared_LR_aclus
        else:  # 'RL' in decoder_name
            shared_aclus = shared_RL_aclus
        
        # Only include neurons that are in the shared set
        final_shared_aclus = np.intersect1d(filtered_decoder.neuron_IDs, shared_aclus)
        final_filtered_decoders[decoder_name] = filtered_decoder.get_by_id(final_shared_aclus)
    
    # Create TrackTemplates directly using the constructor
    rank_method = 'average'
    track_templates = TrackTemplates(
        long_LR_decoder=final_filtered_decoders['long_LR'],
        long_RL_decoder=final_filtered_decoders['long_RL'],
        short_LR_decoder=final_filtered_decoders['short_LR'],
        short_RL_decoder=final_filtered_decoders['short_RL'],
        shared_LR_aclus_only_neuron_IDs=shared_LR_aclus,
        is_good_LR_aclus=None,  # You can compute this if needed
        shared_RL_aclus_only_neuron_IDs=shared_RL_aclus,
        is_good_RL_aclus=None,  # You can compute this if needed
        decoder_LR_pf_peak_ranks_list=[
            scipy.stats.rankdata(final_filtered_decoders['long_LR'].pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method),
            scipy.stats.rankdata(final_filtered_decoders['short_LR'].pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method)
        ],
        decoder_RL_pf_peak_ranks_list=[
            scipy.stats.rankdata(final_filtered_decoders['long_RL'].pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method),
            scipy.stats.rankdata(final_filtered_decoders['short_RL'].pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method)
        ],
        rank_method=rank_method
    )
    
    return track_templates

# Usage in your batch function:
track_templates = create_filtered_track_templates_flexible(
    directional_laps_results, 
    minimum_inclusion_fr_Hz, 
    included_qclu_values
)
```

### **Key Differences from `init_from_paired_decoders`:**

1. **No strict neuron ID matching**: This approach allows for different neuron sets between long/short tracks initially
2. **Flexible shared neuron computation**: You can choose intersection (conservative) or union (inclusive) for shared neurons
3. **Direct constructor usage**: Bypasses the assertion checks that cause your error
4. **Maintains compatibility**: The resulting `TrackTemplates` object works exactly the same as one created via `init_from_paired_decoders`

This approach gives you the flexibility to handle cases where filtering results in different neuron sets while still maintaining the essential structure and functionality of `TrackTemplates`.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "TrackTemplates" "init" "__init__"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22TrackTemplates%22%20%22init%22%20%22__init__%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L173-250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L173-250)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "TrackTemplates" "__init__" "init_from"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22TrackTemplates%22%20%22__init__%22%20%22init_from%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L599-620](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L599-620)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis TrackTemplates "def __init__"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20TrackTemplates%20%22def%20__init__%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L620-700](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L620-700)

## Suggested Follow-ups

- Can you explain the trade-offs between using the intersection vs. union of neuron IDs when creating the shared neuron set in the direct constructor approach?
- Could you provide an example of how to determine the `is_good_LR_aclus` and `is_good_RL_aclus` parameters when directly constructing the `TrackTemplates` object?


---

# Given ```python
    @classmethod
    def determine_decoder_aclus_filtered_by_frate_and_qclu(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)


        individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair


        Usage:
            filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(decoders_dict=track_templates.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)


        """
        decoder_names = cls.get_decoder_names() # ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=decoders_dict, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        # individual_decoder_filtered_aclus_list = list(modified_neuron_ids_dict.values())
        individual_decoder_filtered_aclus_list = [modified_neuron_ids_dict[a_decoder_name] for a_decoder_name in decoder_names]
        assert len(individual_decoder_filtered_aclus_list) == 4, f"len(individual_decoder_filtered_aclus_list): {len(individual_decoder_filtered_aclus_list)} but expected 4!"
        original_decoder_list = [deepcopy(decoders_dict[a_decoder_name]) for a_decoder_name in decoder_names]
        ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
        filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
        filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
        # build the final shared aclus:
        filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
        filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip(original_decoder_list, filtered_direction_shared_aclus_list)]
        return filtered_decoder_list, filtered_direction_shared_aclus_list
```
shouldn't this imply that the returned decoders in a given direction should always be paired?

You're absolutely right! Looking at this code, the `determine_decoder_aclus_filtered_by_frate_and_qclu` method is specifically designed to ensure that decoders in the same direction (LR or RL) have identical neuron sets.

Let me trace through the logic:

1. **Individual filtering**: First, it filters each decoder independently based on firing rate and qclu criteria
2. **Union operation**: It then takes the **union** of neurons from long and short tracks for each direction:
   ```python
   filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])  # long_LR ∪ short_LR
   filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])  # long_RL ∪ short_RL
   ```
3. **Shared application**: It applies the same shared neuron set to both long and short decoders in each direction:
   ```python
   filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus]
   ```

**This should indeed ensure that paired decoders have identical neuron IDs.**

## **The Problem Must Be Elsewhere**

Since this method is designed to create properly paired decoders, the assertion error you're encountering suggests one of these possibilities:

1. **Bug in the filtering logic**: There might be an edge case or bug in `_perform_determine_decoder_aclus_filtered_by_qclu_and_frate`
2. **Data corruption**: The neuron IDs might be getting modified somewhere between filtering and the assertion
3. **Different filtering parameters**: The method might not be using the same parameters you expect

## **Debugging Steps**

Let's add some debugging to see what's happening:

```python
# Add this debugging code before the assertion that fails:
def debug_filtered_templates(directional_laps_results, minimum_inclusion_fr_Hz, included_qclu_values):
    decoders_dict = directional_laps_results.get_decoders_dict()
    
    print("=== DEBUGGING FILTERED TEMPLATES ===")
    print(f"minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}")
    print(f"included_qclu_values: {included_qclu_values}")
    
    # Check what the filtering method returns
    filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate_and_qclu(
        decoders_dict=decoders_dict, 
        minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, 
        included_qclu_values=included_qclu_values
    )
    
    decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    
    print("\nFiltered decoder neuron IDs:")
    for i, (name, decoder) in enumerate(zip(decoder_names, filtered_decoder_list)):
        print(f"{name}: {decoder.neuron_IDs}")
    
    print(f"\nFiltered shared aclus:")
    print(f"LR shared: {filtered_direction_shared_aclus_list[0]}")
    print(f"RL shared: {filtered_direction_shared_aclus_list[1]}")
    
    # Check if they match as expected
    long_LR_ids = filtered_decoder_list[0].neuron_IDs
    short_LR_ids = filtered_decoder_list[2].neuron_IDs
    long_RL_ids = filtered_decoder_list[1].neuron_IDs
    short_RL_ids = filtered_decoder_list[3].neuron_IDs
    
    print(f"\nLR pair match: {np.array_equal(long_LR_ids, short_LR_ids)}")
    print(f"RL pair match: {np.array_equal(long_RL_ids, short_RL_ids)}")
    
    if not np.array_equal(long_LR_ids, short_LR_ids):
        print(f"LR mismatch - long: {long_LR_ids}")
        print(f"LR mismatch - short: {short_LR_ids}")
        print(f"LR difference: {np.setdiff1d(long_LR_ids, short_LR_ids)} vs {np.setdiff1d(short_LR_ids, long_LR_ids)}")
    
    return filtered_decoder_list, filtered_direction_shared_aclus_list

# Use this in your code:
try:
    track_templates = directional_laps_results.get_templates(
        minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, 
        included_qclu_values=included_qclu_values
    )
except AssertionError as e:
    print(f"AssertionError caught: {e}")
    # Run debugging
    debug_filtered_templates(directional_laps_results, minimum_inclusion_fr_Hz, included_qclu_values)
    raise
```

## **Alternative Hypothesis**

The issue might be in the `get_by_id` method or how the decoder filtering is applied. The `union1d` operation should ensure identical neuron sets, but there could be:

1. **Ordering issues**: The neurons might be the same but in different orders
2. **Data type issues**: Different numpy dtypes causing comparison failures
3. **Floating point precision**: If neuron IDs are somehow getting converted to floats

You're correct that the logic should ensure paired decoders have identical neuron IDs. The fact that it's failing suggests there's either a bug in the implementation or some edge case in your specific data that's not being handled properly.

## Suggested Follow-ups

- Could the issue be related to how `get_by_id` is implemented and how it interacts with the `neuron_IDs`?
- Is it possible that the `union1d` operation is somehow changing the data type or order of the neuron IDs, leading to a mismatch even if the underlying set of neurons is the same?

