from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

""" 
Call from a Jupyter Notebook like:

```python

%run -i "SCRATCH/test_external_script.py"

```

It should recieve all of the variables from the notebook.

"""
# your_script.py expects var1 and var2 to be defined
# print(f"var1: {var1}, var2: {var2}")


## RankOrderRastersDebugger: 
spikes_df = curr_active_pipeline.sess.spikes_df
rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

#TODO 2023-11-17 19:57: - [ ] Find other expansions of this kinda and replace it
# LR_laps_epoch_ranked_aclus_stats_dict, LR_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_laps_long_z_score_values, LR_laps_short_z_score_values, LR_laps_long_short_z_score_diff_values = rank_order_results.LR_laps
# RL_laps_epoch_ranked_aclus_stats_dict, RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, RL_laps_long_z_score_values, RL_laps_short_z_score_values, RL_laps_long_short_z_score_diff_values = rank_order_results.RL_laps

# LR_ripple_evts_epoch_ranked_aclus_stats_dict, LR_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_ripple_evts_long_z_score_values, LR_ripple_evts_short_z_score_values, LR_ripple_evts_long_short_z_score_diff_values = rank_order_results.LR_ripple
# RL_ripple_evts_epoch_ranked_aclus_stats_dict, RL_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, RL_ripple_evts_long_z_score_values, RL_ripple_evts_short_z_score_values, RL_ripple_evts_long_short_z_score_diff_values = rank_order_results.RL_ripple

ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple

_out = RankOrderRastersDebugger.init_rank_order_debugger(spikes_df, ripple_result_tuple.active_epochs, track_templates, rank_order_results.RL_ripple.selected_spikes_fragile_linear_neuron_IDX_dict, rank_order_results.LR_ripple.selected_spikes_fragile_linear_neuron_IDX_dict)
# _out = RankOrderRastersDebugger.init_rank_order_debugger(spikes_df, ripple_result_tuple.active_epochs, track_templates, rank_order_results.RL_ripple.selected_spikes_df, rank_order_results.LR_ripple.selected_spikes_df)
# _out = RankOrderRastersDebugger.init_rank_order_debugger(spikes_df, ripple_result_tuple.active_epochs, track_templates, None, None)
n_epochs: int = np.shape(_out.active_epochs_df)[0] # 626
print(f'n_epochs: {n_epochs}')

_out