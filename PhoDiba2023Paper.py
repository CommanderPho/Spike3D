import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum
from neuropy.plotting.ratemaps import BackgroundRenderingOptions
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots

_bak_rcParams = mpl.rcParams.copy()
# mpl.rcParams['toolbar'] = 'None' # disable toolbars
%matplotlib qt






curr_active_pipeline.reload_default_display_functions()




# ==================================================================================================================== #
# Perform `batch_perform_all_plots`                                                                                    #
# ==================================================================================================================== #
neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True)



# ==================================================================================================================== #
# Extract Relevent Specific Data Needed for Figure Display                                                             #
# ==================================================================================================================== #
## long_short_decoding_analyses:
curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
## Extract variables from results object:
long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

# (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D
decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333

## Get global 'long_short_post_decoding' results:
curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result['Flat_epoch_time_bins_mean'], expected_v_observed_result['Flat_decoder_time_bin_centers'], expected_v_observed_result['num_neurons'], expected_v_observed_result['num_timebins_in_epoch'], expected_v_observed_result['num_total_flat_timebins'], expected_v_observed_result['is_short_track_epoch'], expected_v_observed_result['is_long_track_epoch'], expected_v_observed_result['short_short_diff'], expected_v_observed_result['long_long_diff']

jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
(epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


# ==================================================================================================================== #
# Figure 1) pf1D Ratemaps, Active set, etc                                                                             #
# ==================================================================================================================== #
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot, plot_raster_plot
from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
from PendingNotebookCode import PAPER_FIGURE_figure_1_full, PAPER_FIGURE_figure_1_add_replay_epoch_rasters
from PendingNotebookCode import build_shared_sorted_neuronIDs

pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = PAPER_FIGURE_figure_1_full(curr_active_pipeline) # did not display the pf1






# Critical new code:
ratemap = long_pf1D.ratemap
included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())


# ==================================================================================================================== #
# Figure 2) Firing Rate Bar Graphs                                                                                     #
# ==================================================================================================================== #


# Instantaneous versions:
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
from PendingNotebookCode import PaperFigureTwo

_out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
_out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
_out_fig_2.display(defer_show=True)


# ==================================================================================================================== #
# Figure 3) `PAPER_FIGURE_figure_3`: Firing Rate Index and Long/Short Firing Rate Replays v. Laps                      #
# ==================================================================================================================== #
_out, _out2 = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True)




# ==================================================================================================================== #
# HELPERS: Interactive Components                                                                                      #
# ==================================================================================================================== #
from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
# fig, ax, rect_selector, set_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)

grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True, should_apply_updates_to_pipeline=False)
print(f'grid_bin_bounds: {grid_bin_bounds}')
print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")


# ==================================================================================================================== #
# DEBUGGING:                                                                                                           #
# ==================================================================================================================== #
### Testing `plot_kourosh_activity_style_figure` for debugging:
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import plot_kourosh_activity_style_figure
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.helpers import _helper_make_scatterplot_clickable

# plot_aclus = EITHER_subset.track_exclusive_aclus.copy()
plot_aclus = EITHER_subset.track_exclusive_aclus[new_all_aclus_sort_indicies].copy()
_out_A = plot_kourosh_activity_style_figure(long_results_obj, long_session, plot_aclus, unit_sort_order=new_all_aclus_sort_indicies, epoch_idx=13, callout_epoch_IDXs=None, skip_rendering_callouts=False)
app, win, plots, plots_data = _out_A
# plots


