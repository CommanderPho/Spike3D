## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt

from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

should_force_recompute_placefields = True
should_display_2D_plots = True

# ==================================================================================================================== #
# 2022-12-08 Batch Programmatic Figures (Currently only Jonathan-style)                                                #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership # needed for batch_programmatic_figures
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import create_daily_programmatic_display_function_testing_folder_if_needed

def session_context_to_relative_path(parent_path, session_ctx):
    """_summary_

    Args:
        parent_path (_type_): _description_
        session_ctx (_type_): _description_

    Returns:
        _type_: _description_

    Usage:

        curr_sess_ctx = local_session_contexts_list[0]
        # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
        session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)

    """
    parent_path = Path(parent_path)
    subset_whitelist=['format_name','animal','exper_name', 'session_name']
    all_keys_found, found_keys, missing_keys = session_ctx.check_keys(subset_whitelist, debug_print=False)
    if not all_keys_found:
        print(f'WARNING: missing {len(missing_keys)} keys from context: {missing_keys}. Building path anyway.')
    curr_sess_ctx_tuple = session_ctx.as_tuple(subset_whitelist=subset_whitelist, drop_missing=True) # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')
    return parent_path.joinpath(*curr_sess_ctx_tuple).resolve()
    



from enum import unique # SessionBatchProgress
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # required for SessionBatchProgress

@unique
class SessionBatchProgress(ExtendedEnum):
    """Indicates the progress state for a given session in a batch processing queue """
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"

def batch_programmatic_figures(curr_active_pipeline):
    """ programmatically generates and saves the batch figures 2022-12-07 
        curr_active_pipeline is the pipeline for a given session with all computations done.

    ## TODO: curr_session_parent_out_path


    """
    ## 🗨️🟢 2022-10-26 - Jonathan Firing Rate Analyses
    # Perform missing global computations                                                                                  #
    curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_jonathan_replay_firing_rate_analyses', '_perform_short_long_pf_overlap_analyses'], fail_on_exception=True, debug_print=True)

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']

    # ==================================================================================================================== #
    # Batch Output of Figures                                                                                              #
    # ==================================================================================================================== #
    ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
    # %matplotlib qt
    short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
    short_only_aclus = short_only_df.index.values.tolist()
    long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
    long_only_aclus = long_only_df.index.values.tolist()
    shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
    shared_aclus = shared_df.index.values.tolist()
    print(f'shared_aclus: {shared_aclus}')
    print(f'long_only_aclus: {long_only_aclus}')
    print(f'short_only_aclus: {short_only_aclus}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    curr_sess_ctx = local_session_contexts_list[0]
    # curr_sess_ctx # IdentifyingContext<('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')>
    figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
    curr_session_parent_out_path = session_context_to_relative_path(figures_parent_out_path, curr_sess_ctx)
    print(f'curr_session_parent_out_path: {curr_session_parent_out_path}')
    curr_session_parent_out_path.mkdir(exist_ok=True) # make folder if needed



    # ==================================================================================================================== #
    # Output Figures to File                                                                                               #
    # ==================================================================================================================== #
    ## PDF Output
    # %matplotlib qtagg
    import matplotlib
    # configure backend here
    # matplotlib.use('Qt5Agg')
    # backend_qt5agg
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!

    n_max_page_rows = 10
    _batch_plot_kwargs_list = _build_batch_plot_kwargs(long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=n_max_page_rows)
    active_out_figures_list = _perform_batch_plot(curr_active_pipeline, _batch_plot_kwargs_list, figures_parent_out_path=curr_session_parent_out_path, write_pdf=False, write_png=True, progress_print=True, debug_print=False)

    return active_identifying_session_ctx, active_out_figures_list


# ==================================================================================================================== #
# 2022-12-07 - batch_load_session - Computes Entire Pipeline                                                           #
# ==================================================================================================================== #
from neuropy.core.epoch import NamedTimerange
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # for batch_load_session

def batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, force_reload=False, **kwargs):
    """Loads and runs the entire pipeline for a session folder located at the path 'basedir'.

    Args:
        global_data_root_parent_path (_type_): _description_
        active_data_mode_name (_type_): _description_
        basedir (_type_): _description_

    Returns:
        _type_: _description_
    """

    epoch_name_whitelist = kwargs.get('epoch_name_whitelist', ['maze1','maze2','maze'])
    debug_print = kwargs.get('debug_print', False)
    skip_save = kwargs.get('skip_save', True)

    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
        override_basepath=Path(basedir), override_post_load_functions=[], force_reload=force_reload, active_pickle_filename='loadedSessPickle.pkl', skip_save=skip_save)

    active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_whitelist=epoch_name_whitelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
    if debug_print:
        print(f'active_session_filter_configurations: {active_session_filter_configurations}')

    active_session_computation_configs = active_data_mode_registered_class.build_default_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=0.03333) #1.0/30.0 # decode at 30fps to match the position sampling frequency
    curr_active_pipeline.filter_sessions(active_session_filter_configurations, debug_print=True)

    # Whitelist Mode:
    computation_functions_name_whitelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                        '_perform_position_decoding_computation', 
                                        '_perform_firing_rate_trends_computation',
                                        '_perform_pf_find_ratemap_peaks_computation',
                                        # '_perform_two_step_position_decoding_computation',
                                        # '_perform_recursive_latent_placefield_decoding'
                                     ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'
    computation_functions_name_blacklist=None

    # # Blacklist Mode:
    # computation_functions_name_whitelist=None
    # computation_functions_name_blacklist=['_perform_spike_burst_detection_computation','_perform_recursive_latent_placefield_decoding']

    curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=True, debug_print=debug_print) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)

    # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_blacklist=['_perform_spike_burst_detection_computation'], debug_print=False, fail_on_exception=False) # whitelist: ['_perform_baseline_placefield_computation']

    curr_active_pipeline.prepare_for_display(root_output_dir=global_data_root_parent_path.joinpath('Output'), should_smooth_maze=True) # TODO: pass a display config

    if not skip_save:
        curr_active_pipeline.save_pipeline()
    else:
        print(f'skip_save == True, so not saving at the end of batch_load_session')
    return curr_active_pipeline


# ==================================================================================================================== #
# 2022-12-07 - Finding Local Session Paths
# ==================================================================================================================== #

def find_local_session_paths(local_session_parent_path, blacklist=[], debug_print=True):
    try:
        found_local_session_paths_list = [x for x in local_session_parent_path.iterdir() if x.is_dir()]
        local_session_names_list = [a_path.name for a_path in found_local_session_paths_list if a_path.name not in blacklist]
        if debug_print:
            print(f'local_session_names_list: {local_session_names_list}')
        local_session_paths_list = [local_session_parent_path.joinpath(a_name).resolve() for a_name in local_session_names_list]
        
    except Exception as e:
        print(f"Error processing path: '{local_session_parent_path}' due to exception: {e}. Skipping...")
        local_session_paths_list = None
        local_session_names_list = None
        
    return local_session_paths_list, local_session_names_list


# ==================================================================================================================== #
# 2022-12-01 - Automated programmatic output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`                                                                                                          #
# ==================================================================================================================== #

## PDF Output 
from matplotlib.backends import backend_pdf
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_metadata_from_display_context, create_daily_programmatic_display_function_testing_folder_if_needed # newer version of build_pdf_export_metadata

from neuropy.utils.misc import compute_paginated_grid_config # for paginating shared aclus
def _subfn_batch_plot_automated(curr_active_pipeline, included_unit_neuron_IDs=None, active_identifying_ctx=None, fignum=None, fig_idx=0, n_max_page_rows=10):
    """ the a programmatic wrapper for automated output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`. The specific plot function called. 
    Called ONLY by `_perform_batch_plot(...)`

    """
    # size_dpi = 100.0,
    # single_subfigure_size_px = np.array([1920.0, 220.0])
    single_subfigure_size_inches = np.array([19.2,  2.2])

    num_cells = len(included_unit_neuron_IDs)
    desired_figure_size_inches = single_subfigure_size_inches.copy()
    desired_figure_size_inches[1] = desired_figure_size_inches[1] * num_cells
    graphics_output_dict = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', active_identifying_ctx,
                                                        n_max_plot_rows=n_max_page_rows, included_unit_neuron_IDs=included_unit_neuron_IDs,
                                                        show_inter_replay_frs=False, spikes_color=(0.1, 0.0, 0.1), spikes_alpha=0.5, fignum=fignum, fig_idx=fig_idx, figsize=desired_figure_size_inches)
    fig, subfigs, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['subfigs'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
    fig.suptitle(active_identifying_ctx.get_description()) # 'kdiba_2006-6-08_14-26-15_[4, 13, 36, 58, 60]'
    return fig

def _build_batch_plot_kwargs(long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=10):
    """ builds the list of kwargs for all aclus """
    _batch_plot_kwargs_list = [] # empty list to start
    ## {long_only, short_only} plot configs (doesn't include the shared_aclus)
    if len(long_only_aclus) > 0:        
        _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=long_only_aclus,
        active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
            display_fn_name='batch_plot_test', plot_result_set='long_only', aclus=f"{long_only_aclus}"
        ),
        fignum='long_only', n_max_page_rows=len(long_only_aclus)))
    else:
        print(f'WARNING: long_only_aclus is empty, so not adding kwargs for these.')
    
    if len(short_only_aclus) > 0:
         _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=short_only_aclus,
        active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
            display_fn_name='batch_plot_test', plot_result_set='short_only', aclus=f"{short_only_aclus}"
        ),
        fignum='short_only', n_max_page_rows=len(short_only_aclus)))
    else:
        print(f'WARNING: short_only_aclus is empty, so not adding kwargs for these.')

    ## Build Pages for Shared ACLUS:    
    nAclusToShow = len(shared_aclus)
    if nAclusToShow > 0:        
        # Paging Management: Constrain the subplots values to just those that you need
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nAclusToShow, max_num_columns=1, max_subplots_per_page=n_max_page_rows, data_indicies=shared_aclus, last_figure_subplots_same_layout=True)
        num_pages = len(included_combined_indicies_pages)
        ## paginated outputs for shared cells
        included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
        paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
            active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name='batch_plot_test', plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{curr_included_unit_indicies}"),
            fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]
        _batch_plot_kwargs_list.extend(paginated_shared_cells_kwarg_list) # add paginated_shared_cells_kwarg_list to the list
    else:
        print(f'WARNING: shared_aclus is empty, so not adding kwargs for these.')
    return _batch_plot_kwargs_list

def _perform_batch_plot(curr_active_pipeline, active_kwarg_list, figures_parent_out_path=None, write_pdf=False, write_png=True, progress_print=True, debug_print=False):
    """ Plots everything using the kwargs provided in `active_kwarg_list`

    Args:
        active_kwarg_list (_type_): generated by `_build_batch_plot_kwargs(...)`
        figures_parent_out_path (_type_, optional): _description_. Defaults to None.
        write_pdf (bool, optional): _description_. Defaults to False.
        write_png (bool, optional): _description_. Defaults to True.
        progress_print (bool, optional): _description_. Defaults to True.
        debug_print (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if figures_parent_out_path is None:
        figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()
    active_out_figures_list = [] # empty list to hold figures
    num_pages = len(active_kwarg_list)
    for i, curr_batch_plot_kwargs in enumerate(active_kwarg_list):
        curr_active_identifying_ctx = curr_batch_plot_kwargs['active_identifying_ctx']
        # print(f'curr_active_identifying_ctx: {curr_active_identifying_ctx}')
        active_pdf_metadata, active_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx)
        # print(f'active_pdf_save_filename: {active_pdf_save_filename}')
        curr_pdf_save_path = figures_parent_out_path.joinpath(active_pdf_save_filename) # build the final output pdf path from the pdf_parent_out_path (which is the daily folder)
        # One plot at a time to PDF:
        if write_pdf:
            with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                a_fig = _subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
                active_out_figures_list.append(a_fig)
                # Save out PDF page:
                pdf.savefig(a_fig)
        else:
            a_fig = _subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
            active_out_figures_list.append(a_fig)

        # Also save .png versions:
        if write_png:
            # curr_page_str = f'pg{i+1}of{num_pages}'
            fig_png_out_path = curr_pdf_save_path.with_suffix('.png')
            # fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
            a_fig.savefig(fig_png_out_path)
            if progress_print:
                print(f'\t saved {fig_png_out_path}')

    return active_out_figures_list










# ==================================================================================================================== #
# 2022-11-09                                                                                                           #
# ==================================================================================================================== #
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

def scale_title_label(ax, curr_title_obj, curr_im, debug_print=False):
    """ Scales some matplotlib-based figures titles to be reasonable. I remember that this was important and hard to make, but don't actually remember what it does as of 2022-10-24. It needs to be moved in to somewhere else.
    """
    ### USAGE:
    # ## Scale all:
    # _display_outputs = widget.last_added_display_output
    # curr_graphics_objs = _display_outputs.graphics[0]

    # """ curr_graphics_objs is:
    # {2: {'axs': [<Axes:label='2'>],
    # 'image': <matplotlib.image.AxesImage at 0x1630c4556d0>,
    # 'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c4559a0>},
    # 4: {'axs': [<Axes:label='4'>],
    # 'image': <matplotlib.image.AxesImage at 0x1630c455f70>,
    # 'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463280>},
    # 5: {'axs': [<Axes:label='5'>],
    # 'image': <matplotlib.image.AxesImage at 0x1630c463850>,
    # 'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463b20>},
    # ...
    # """
    # for aclu, curr_neuron_graphics_dict in curr_graphics_objs.items():
    #     curr_title_obj = curr_neuron_graphics_dict['title_obj'] # matplotlib.offsetbox.AnchoredText
    #     curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
    #     curr_im = curr_neuron_graphics_dict['image'] # matplotlib.image.AxesImage
    #     curr_ax = curr_neuron_graphics_dict['axs'][0]
    #     scale_title_label(curr_ax, curr_title_obj, curr_im)


    transform = ax.transData

    curr_label_extent = curr_title_obj.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1028.49144862968, 2179.0555555555566], [1167.86644862968, 2193.0555555555566]])
    curr_label_width = curr_label_extent.width # 139.375
    # curr_im.get_extent() # (-85.75619321393464, 112.57838773103435, -96.44772761274268, 98.62205280781535)
    img_extent = curr_im.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1049.76842294452, 2104.7727272727284], [1146.5894743148401, 2200.000000000001]])
    curr_img_width = img_extent.width # 96.82105137032022
    needed_scale_factor = curr_img_width / curr_label_width
    if debug_print:
        print(f'curr_label_width: {curr_label_width}, curr_img_width: {curr_img_width}, needed_scale_factor: {needed_scale_factor}')
    needed_scale_factor = min(needed_scale_factor, 1.0) # Only scale up, don't scale down
    
    curr_font_props = curr_title_obj.prop # FontProperties
    curr_font_size_pts = curr_font_props.get_size_in_points() # 10.0
    curr_scaled_font_size_pts = needed_scale_factor * curr_font_size_pts
    if debug_print:
        print(f'curr_font_size_pts: {curr_font_size_pts}, curr_scaled_font_size_pts: {curr_scaled_font_size_pts}')

    if isinstance(curr_title_obj, AnchoredText):
        curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
    else:
        curr_title_text_obj = curr_title_obj
    
    curr_title_text_obj.set_fontsize(curr_scaled_font_size_pts)
    font_foreground = 'white'
    # font_foreground = 'black'
    curr_title_text_obj.set_color(font_foreground)
    # curr_title_text_obj.set_fontsize(6)
    
    stroke_foreground = 'black'
    # stroke_foreground = 'gray'
    # stroke_foreground = 'orange'
    strokewidth = 4
    curr_title_text_obj.set_path_effects([withStroke(foreground=stroke_foreground, linewidth=strokewidth)])
    ## Disable path effects:
    # curr_title_text_obj.set_path_effects([])

#     ## Figure computation
#     fig: plt.Figure = ax.get_figure()
#     dpi = fig.dpi
#     rect_height_inch = rect_height / dpi
#     # Initial fontsize according to the height of boxes
#     fontsize = rect_height_inch * 72
#     print(f'rect_height_inch: {rect_height_inch}, fontsize: {fontsize}')

# #     text: Annotation = ax.annotate(txt, xy, ha=ha, va=va, xycoords=transform, **kwargs)

# #     # Adjust the fontsize according to the box size.
# #     text.set_fontsize(fontsize)
#     bbox: Bbox = text.get_window_extent(fig.canvas.get_renderer())
#     adjusted_size = fontsize * rect_width / bbox.width
#     print(f'bbox: {bbox}, adjusted_size: {adjusted_size}')
#     text.set_fontsize(adjusted_size)
    
    

    


# ==================================================================================================================== #
# 2022-08-18                                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow # required for display_all_eloy_pf_density_measures_results

def display_all_eloy_pf_density_measures_results(active_pf_2D, active_eloy_analysis, active_simpler_pf_densities_analysis, active_peak_prominence_2d_results):
    """ 
    Usage:
        out_all_eloy_pf_density_fig = display_all_eloy_pf_density_measures_results(active_pf_2D, active_eloy_analysis, active_simpler_pf_densities_analysis, active_peak_prominence_2d_results)
        
    """
    # active_xbins = active_pf_2D.xbin
    # active_ybins = active_pf_2D.ybin
    
    # # *bin_indicies:
    # xbin_indicies = active_pf_2D.xbin_labels -1
    # ybin_indicies = active_pf_2D.ybin_labels -1
    # active_xbins = xbin_indicies
    # active_ybins = ybin_indicies
    
    # *bin_centers: these seem to work
    active_xbins = active_pf_2D.xbin_centers
    active_ybins = active_pf_2D.ybin_centers
    
    out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_xbins, active_ybins, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity')
    out.add_data(row=2, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_xbins, ybins=active_ybins, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')
    out.add_data(row=3, col=0, matrix=active_pf_2D.ratemap.occupancy, xbins=active_xbins, ybins=active_ybins, name='occupancy_seconds', title='Seconds Occupancy', variable_label='seconds')
    out.add_data(row=4, col=0, matrix=active_simpler_pf_densities_analysis.n_neurons_meeting_firing_critiera_by_position_bins_2D, xbins=active_xbins, ybins=active_ybins, name='n_neurons_meeting_firing_critiera_by_position_bins_2D', title='# neurons > 1Hz per Pos (X, Y)', variable_label='# neurons')
    # out.add_data(row=5, col=0, matrix=active_peak_prominence_2d_results.peak_counts.raw, xbins=active_pf_2D.xbin_labels, ybins=active_pf_2D.ybin_labels, name='pf_peak_counts_map', title='# pf peaks per Pos (X, Y)', variable_label='# pf peaks')
    # out.add_data(row=6, col=0, matrix=active_peak_prominence_2d_results.peak_counts.gaussian_blurred, xbins=active_pf_2D.xbin_labels, ybins=active_pf_2D.ybin_labels, name='pf_peak_counts_map_blurred gaussian', title='Gaussian blurred # pf peaks per Pos (X, Y)', variable_label='Gaussian blurred # pf peaks')
    out.add_data(row=5, col=0, matrix=active_peak_prominence_2d_results.peak_counts.raw, xbins=active_xbins, ybins=active_ybins, name='pf_peak_counts_map', title='# pf peaks per Pos (X, Y)', variable_label='# pf peaks')
    out.add_data(row=6, col=0, matrix=active_peak_prominence_2d_results.peak_counts.gaussian_blurred, xbins=active_xbins, ybins=active_ybins, name='pf_peak_counts_map_blurred gaussian', title='Gaussian blurred # pf peaks per Pos (X, Y)', variable_label='Gaussian blurred # pf peaks')

    return out
    

# ==================================================================================================================== #
# Pre 2022-08-16 Figure Docking                                                                                        #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget

def _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, extant_dockAreaWidget=None, debug_print=False):
    """ Combines the active_pf_2D individual figures into a single widget, with each item being docked and modifiable.
    Requies figures to already be created and passed in the appropriate format.
    
    # TODO: On close should close the figure handles that are currently open. Can use figure_manager to do this.
    
    
    # TODO: Shouldnt' this be a widget instead of a function? Maybe it doesn't matter though.
    
    if should_nest_figures_on_filter is True, the figures are docked in a nested dockarea for each filter (e.g. ['maze1', 'maze2']. Otherwise they are returned flat.
        
    Unique to nested:
        all_nested_dock_area_widgets = {}
        all_nested_dock_area_widget_display_items = {}

    NOTE: This builds a brand-new independent dockAreaWindow, with no option to reuse an extant one.

    Usage:
    
        def _display_specified__display_2d_placefield_result_plot_ratemaps_2D(filter_name):
            active_filter_pf_2D_figures = {}
            active_filter_pf_2D_figures['SPIKES_MAPS'] = curr_active_pipeline.display('_display_2d_placefield_result_plot_ratemaps_2D', filter_name, plot_variable=enumTuningMap2DPlotVariables.SPIKES_MAPS, fignum=plots_fig_nums_dict[filter_name][0], **figure_format_config)[0]
            active_filter_pf_2D_figures['TUNING_MAPS'] = curr_active_pipeline.display('_display_2d_placefield_result_plot_ratemaps_2D', filter_name, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=plots_fig_nums_dict[filter_name][1], **figure_format_config)[0]
            return active_filter_pf_2D_figures

        active_pf_2D_figures = {}
        ## Plots for each maze programmatically:
        for i, filter_name in enumerate(curr_active_pipeline.active_config_names):
            active_pf_2D_figures[filter_name] = _display_specified__display_2d_placefield_result_plot_ratemaps_2D(filter_name=filter_name)

        active_pf_2D_figures
        # {'maze1': {'SPIKES_MAPS': <Figure size 1728x1080 with 88 Axes>,
        #   'TUNING_MAPS': <Figure size 1728x1080 with 88 Axes>},
        #  'maze2': {'SPIKES_MAPS': <Figure size 1728x864 with 71 Axes>,
        #   'TUNING_MAPS': <Figure size 1728x864 with 71 Axes>}}

        win, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items = _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, debug_print=False)

        win, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items = _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, debug_print=False)
        
        
    """
    min_width = 500
    min_height = 500
    if extant_dockAreaWidget is None:
        created_new_main_widget = True
        active_containing_dockAreaWidget, app = DockAreaWrapper._build_default_dockAreaWindow(title='active_pf_2D_figures', defer_show=False)
    else:
        created_new_main_widget = False
        active_containing_dockAreaWidget = extant_dockAreaWidget

    all_dock_display_items = {}
    all_item_widths_list = []
    all_item_heights_list = []

    if should_nest_figures_on_filter:
        all_nested_dock_area_widgets = {}
        all_nested_dock_area_widget_display_items = {}

        _last_dock_outer_nested_item = None
        for filter_name, a_figures_dict in active_pf_2D_figures.items():
            # For each filter, create a new NestedDockAreaWidget
            all_nested_dock_area_widgets[filter_name] = NestedDockAreaWidget()
            # Once done with a given filter, add its nested dockarea widget to the window
            if _last_dock_outer_nested_item is not None:
                #NOTE: to stack two dock widgets on top of each other, do area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
                dockAddLocationOpts = ['above', _last_dock_outer_nested_item] # position relative to the _last_dock_outer_nested_item for this figure
            else:
                dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
            nested_out_widget_key = f'Nested Outer Widget: {filter_name}'
            if debug_print:
                print(f'nested_out_widget_key: {nested_out_widget_key}')
            _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(nested_out_widget_key, dockSize=(min_width, min_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=all_nested_dock_area_widgets[filter_name], dockAddLocationOpts=dockAddLocationOpts)
            all_nested_dock_area_widget_display_items[filter_name] = dDisplayItem
            _last_dock_outer_nested_item = dDisplayItem

            ## Add the sub-items for this filter:
            _last_dock_item = None
            for a_figure_name, a_figure in a_figures_dict.items():
                # individual figures
                figure_key = f'{filter_name}_{a_figure_name}'
                if debug_print:
                    print(f'figure_key: {figure_key}')
                fig_window = a_figure.canvas.window()
                fig_geom = fig_window.window().geometry() # get the QTCore PyRect object
                fig_x, fig_y, fig_width, fig_height = fig_geom.getRect() # Note: dx & dy refer to width and height
                all_item_widths_list.append(fig_width)
                all_item_heights_list.append(fig_height)

                # Add the dock and keep the display item:
                if _last_dock_item is not None:
                    dockAddLocationOpts = ['above', _last_dock_item] # position relative to the _last_dock_item for this figure
                else:
                    dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
                _, dDisplayItem = all_nested_dock_area_widgets[filter_name].add_display_dock(figure_key, dockSize=(fig_width, fig_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
                dDisplayItem.setOrientation('horizontal') # want orientation of outer dockarea to be opposite of that of the inner one. # 'auto', 'horizontal', or 'vertical'.
                all_dock_display_items[figure_key] = dDisplayItem
                _last_dock_item = dDisplayItem

    else:
        ## Flat (non-nested)
        all_nested_dock_area_widgets = None
        all_nested_dock_area_widget_display_items = None
        
        for filter_name, a_figures_dict in active_pf_2D_figures.items():
            _last_dock_item = None
            for a_figure_name, a_figure in a_figures_dict.items():
                # individual figures
                figure_key = f'{filter_name}_{a_figure_name}'
                if debug_print:
                    print(f'figure_key: {figure_key}')
                fig_window = a_figure.canvas.window()
                fig_geom = fig_window.window().geometry() # get the QTCore PyRect object
                fig_x, fig_y, fig_width, fig_height = fig_geom.getRect() # Note: dx & dy refer to width and height
                all_item_widths_list.append(fig_width)
                all_item_heights_list.append(fig_height)
                
                # Add the dock and keep the display item:
                if _last_dock_item is not None:
                    #NOTE: to stack two dock widgets on top of each other, do area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
                    dockAddLocationOpts = ['above', _last_dock_item] # position relative to the _last_dock_item for this figure
                else:
                    dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
                    
                display_config = CustomDockDisplayConfig()
            
                _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(figure_key, dockSize=(fig_width, fig_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
                all_dock_display_items[figure_key] = dDisplayItem

                _last_dock_item = dDisplayItem

    # Resize window to largest figure size:
    if created_new_main_widget:
        # Only resize if we created this widget, otherwise don't change the size
        all_item_widths_list = np.array(all_item_widths_list)
        all_item_heights_list = np.array(all_item_heights_list)
        max_width = np.max(all_item_widths_list)
        max_height = np.max(all_item_heights_list)
        active_containing_dockAreaWidget.resize(max_width, max_height)
    
    return active_containing_dockAreaWidget, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items
    

# ==================================================================================================================== #
# 2022-07-20                                                                                                           #
# ==================================================================================================================== #

def old_timesynchronized_plotter_testing():
    # # Test PfND_TimeDependent Class

    # ## Old TimeSynchronized*Plotter Testing

    # CELL ==================================================================================================================== #    t = curr_occupancy_plotter.active_time_dependent_placefields.last_t + 7 # add one second
    # with np.errstate(divide='ignore', invalid='ignore'):
    # active_time_dependent_placefields.update(t)
    print(f't: {t}')
    curr_occupancy_plotter.on_window_changed(0.0, t)
    curr_placefields_plotter.on_window_changed(0.0, t)


    # CELL ==================================================================================================================== #    from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image, pyqtplot_plot_image_array
    import time

    def _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ plots a single result at a given time.
        
        Creates the figures if needed, otherwise updates the existing ones.
        
        """
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}} # make a new dictionary to hold the plot objects.

        # images = curr_ratemap.tuning_curves # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        # occupancy = curr_ratemap.curr_raw_occupancy_map

        imv = output_plots_dict.get('occupancy', {}).get('imv', None)
        if imv is None:
            # Otherwise build the plotter:
            occupancy_app, occupancy_win, imv = pyqtplot_plot_image(active_time_dependent_placefields2D.xbin, active_time_dependent_placefields2D.ybin, occupancy)
            output_plots_dict['occupancy'] = dict(zip(('app', 'win', 'imv'), (occupancy_app, occupancy_win, imv)))   
            occupancy_win.show()
        else:
            # Update the existing one:
            imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)

        pg.QtGui.QApplication.processEvents() # call to ensure the occupancy gets updated before starting the placefield plots:
        
        img_item_array = output_plots_dict.get('placefields', {}).get('img_item_array', None)
        if img_item_array is None:
            # Create a new one:
            placefields_app, placefields_win, root_render_widget, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_time_dependent_placefields2D.xbin, active_time_dependent_placefields2D.ybin,
                                                                                                                                            active_time_dependent_placefields2D.ratemap.normalized_tuning_curves, active_time_dependent_placefields2D.curr_raw_occupancy_map)#, 
            output_plots_dict['placefields'] = dict(zip(('app', 'win', 'root_render_widget', 'plot_array', 'img_item_array', 'other_components_array'), (placefields_app, placefields_win, root_render_widget, plot_array, img_item_array, other_components_array)))
            placefields_win.show()

        else:
            # Update the placefields plot if needed:
            images = curr_ratemap.tuning_curves # (43, 63, 63)
            for i, an_img_item in enumerate(img_item_array):
                image = np.squeeze(images[i,:,:])
                # Pre-filter the data:
                # image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
                if drop_below_threshold is not None:
                    image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy        
                # an_img_item.setImage(np.squeeze(images[i,:,:]))
                an_img_item.setImage(image)

        return output_plots_dict


    # CELL ==================================================================================================================== #
    def pre_build_iterative_results(num_iterations=50, t_list=[], ratemaps_list=[]):
        """ 
        build up historical data arrays:
        
        Usage:
            t_list, ratemaps_list = pre_build_iterative_results(num_iterations=50, t_list=t_list, ratemaps_list=ratemaps_list)
        """
        # t_list = []
        # ratemaps_list = []
        
        def _step_plot(time_step_seconds):
            t = active_time_dependent_placefields2D.last_t + time_step_seconds # add one second
            t_list.append(t)
            with np.errstate(divide='ignore', invalid='ignore'):
                active_time_dependent_placefields2D.update(t)
            # Loop through and update the plots:
            # Get flat list of images:
            curr_ratemap = active_time_dependent_placefields2D.ratemap
            # images = curr_ratemap.tuning_curves # (43, 63, 63)
            # images = active_time_dependent_placefields2D.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
            # occupancy = curr_ratemap.occupancy
            ratemaps_list.append(curr_ratemap)
        #     for i, an_img_item in enumerate(img_item_array):
        #     # for i, a_plot in enumerate(plot_array):
        #         # image = np.squeeze(images[i,:,:])
        #         # Pre-filter the data:
        #         # image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        # #         if drop_below_threshold is not None:
        # #             image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy        
        #         an_img_item.setImage(np.squeeze(images[i,:,:]))
        
        for i in np.arange(num_iterations):
            _step_plot(time_step_seconds=1.0)
        
        return t_list, ratemaps_list


    # Loop through the historically collected ratemaps and plot them:
    def _test_plot_historical_iterative_pf_results(t_list, ratemaps_list, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ Uses the previously built-up t_list and ratemaps_list (as computed by pre_build_iterative_results(...)) to plot the time-dependent results.
        requires:
        imv: a previously created single-image plotter:
        """
        num_historical_results = len(ratemaps_list)
        assert len(t_list) == len(ratemaps_list), f"len(t_list): {len(t_list)} needs to equal len(ratemaps_list): {len(ratemaps_list)}"
        
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {},
                                'placefields': {}} # make a new dictionary to hold the plot objects.
            
        for i in np.arange(num_historical_results):
            curr_t = t_list[i]
            # Set up
            # print(f'curr_t: {curr_t}')
            curr_ratemap = ratemaps_list[i]
            output_plots_dict = _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
        
            pg.QtGui.QApplication.processEvents()
            time.sleep(0.1) # Sleep for 0.5 seconds

        return output_plots_dict

    # Build the Historical Results:
    t_list, ratemaps_list = pre_build_iterative_results(num_iterations=50, t_list=t_list, ratemaps_list=ratemaps_list)
    # Plot the historical results:
    if output_plots_dict is None:
        output_plots_dict = {'occupancy': {}, 'placefields': {}}
    output_plots_dict = _test_plot_historical_iterative_pf_results(t_list, ratemaps_list, output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    # Compute the time-dependent ratemap info in real-time and plot them:
    def _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, t, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ 
        requires:
        imv: a previously created single-image plotter:
        """
        # Compute the updated placefields/occupancy for the time t:
        with np.errstate(divide='ignore', invalid='ignore'):
            active_time_dependent_placefields2D.update(t)
        # Update the plots:
        curr_t = active_time_dependent_placefields2D.last_t
        curr_ratemap = active_time_dependent_placefields2D.ratemap

        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}} # make a new dictionary to hold the plot objects.
            
        # Plot the results directly from the active_time_dependent_placefields2D
        output_plots_dict = _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
        pg.QtGui.QApplication.processEvents()
        
        return output_plots_dict

    def _test_live_iterative_pf_results_plot(active_time_dependent_placefields2D, num_iterations=50, time_step_seconds=1.0, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ performs num_iterations time steps of size time_step_seconds and plots the results. """
        for i in np.arange(num_iterations):
            t = active_time_dependent_placefields2D.last_t + time_step_seconds # add one second
            output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, t, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
            time.sleep(0.1) # Sleep for 0.5 seconds


    # CELL ==================================================================================================================== #
    try:
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}}
    except NameError:
        output_plots_dict = {'occupancy': {}, 'placefields': {}}

    output_plots_dict = _test_live_iterative_pf_results_plot(active_time_dependent_placefields2D, num_iterations=50, time_step_seconds=1.0, output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    output_plots_dict = {'occupancy': {}, 'placefields': {}} # clear the output plots dict
    output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, spike_raster_window.spikes_window.active_time_window[1], output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    def _on_window_updated(window_start, window_end):
        # print(f'_on_window_updated(window_start: {window_start}, window_end: {window_end})')
        global output_plots_dict
        ## Update only version:
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     active_time_dependent_placefields2D.update(window_end) # advance the placefield display to the end of the window.
        ## Update and plot version:
        # t = window_end
        output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, window_end, output_plots_dict=output_plots_dict)
        
    # spike_raster_window.connect_additional_controlled_plotter(_on_window_updated)

    _on_window_updated(spike_raster_window.spikes_window.active_time_window[0], spike_raster_window.spikes_window.active_time_window[1])
    sync_connection = spike_raster_window.spike_raster_plt_2d.window_scrolled.connect(_on_window_updated) # connect the window_scrolled event to the _on_window_updated function


    # CELL ==================================================================================================================== #
    active_time_dependent_placefields2D.plot_occupancy()


    # CELL ==================================================================================================================== #
    # active_time_dependent_placefields2D.plot_ratemaps_2D(enable_spike_overlay=False) # Works
    active_time_dependent_placefields2D.plot_ratemaps_2D(enable_spike_overlay=True)


    # CELL ==================================================================================================================== #
    # t_list
    active_time_dependent_placefields2D.plot_ratemaps_2D(enable_saving_to_disk=False, enable_spike_overlay=False)


    # CELL ==================================================================================================================== #
    # ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, active_time_dependent_placefields2D, active_config_name)
    occupancy_fig, occupancy_ax = active_time_dependent_placefields2D.plot_occupancy(identifier_details_list=[])


    # CELL ==================================================================================================================== #
    i = 0
    while (i < len(t_list)):
        curr_t = t_list[i]
        # Set up
        print(f'curr_t: {curr_t}')
        curr_ratemap = ratemaps_list[i]
        # images = curr_ratemap.tuning_curves # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        # occupancy = curr_ratemap.curr_raw_occupancy_map
        imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)
        i += 1
        pg.QtGui.QApplication.processEvents()
        
    print(f'done!')


    # CELL ==================================================================================================================== #
    # Timer Update Approach:
    timer = pg.QtCore.QTimer()
    i = 0
    def update():
        if (i < len(t_list)):
            curr_t = t_list[i]
            # Set up
            print(f'curr_t: {curr_t}')
            curr_ratemap = ratemaps_list[i]
            # images = curr_ratemap.tuning_curves # (43, 63, 63)
            occupancy = curr_ratemap.occupancy
            # occupancy = curr_ratemap.curr_raw_occupancy_map
            imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)
            i += 1
        else:
            print(f'done!')
        # pw.plot(x, y, clear=True)

    timer.timeout.connect(update)


    # CELL ==================================================================================================================== #
    # timer.start(16)
    timer.start(500)


    # CELL ==================================================================================================================== #
    timer.stop()


    # CELL ==================================================================================================================== #
    t_list


    # CELL ==================================================================================================================== #
    # get properties from spike_raster_window:

    active_curve_plotter_3d = test_independent_vedo_raster_widget # use separate vedo plotter
    # active_curve_plotter_3d = spike_raster_window.spike_raster_plt_3d
    curr_computations_results = curr_active_pipeline.computation_results[active_config_name]


    # CELL ==================================================================================================================== #
    ## Spike Smoothed Moving Average Rate:
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.Specific3DTimeCurves import Specific3DTimeCurvesHelper
    binned_spike_moving_average_rate_curve_datasource = Specific3DTimeCurvesHelper.add_unit_time_binned_spike_visualization_curves(curr_computations_results, active_curve_plotter_3d, spike_visualization_mode='mov_average')            


    # CELL ==================================================================================================================== #
    # Get current plot items:
    curr_plot3D_active_window_data = active_curve_plotter_3d.params.time_curves_datasource.get_updated_data_window(active_curve_plotter_3d.spikes_window.active_window_start_time, active_curve_plotter_3d.spikes_window.active_window_end_time) # get updated data for the active window from the datasource
    is_data_series_mode = active_curve_plotter_3d.params.time_curves_datasource.has_data_series_specs
    if is_data_series_mode:
        data_series_spaital_values_list = active_curve_plotter_3d.params.time_curves_datasource.data_series_specs.get_data_series_spatial_values(curr_plot3D_active_window_data)
        num_data_series = len(data_series_spaital_values_list)

    curr_data_series_index = 0
    curr_data_series_dict = data_series_spaital_values_list[curr_data_series_index]

    curr_plot_column_name = curr_data_series_dict.get('name', f'series[{curr_data_series_index}]') # get either the specified name or the generic 'series[i]' name otherwise
    curr_plot_name = active_curve_plotter_3d.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
    # points for the current plot:
    pts = np.column_stack([curr_data_series_dict['x'], curr_data_series_dict['y'], curr_data_series_dict['z']])
    pts


    # CELL ==================================================================================================================== #
    ## Add the new filled plot item:
    plot_args = ({'color_name':'white','line_width':0.5,'z_scaling_factor':1.0})
    _test_fill_plt = gl.GLLinePlotItem(pos=points, color=line_color, width=plot_args.setdefault('line_width',0.5), antialias=True)
    _test_fill_plt.scale(1.0, 1.0, plot_args.setdefault('z_scaling_factor',1.0)) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.            
    # plt.scale(1.0, 1.0, self.data_z_scaling_factor) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.
    active_curve_plotter_3d.ui.main_gl_widget.addItem(_test_fill_plt)


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots.keys()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.render_epochs


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_datasource


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_enable_baseline_grid = True
    active_curve_plotter_3d.params.time_curves_baseline_grid_alpha = 0.9
    # add_3D_time_curves_baseline_grid_mesh


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.add_3D_time_curves_baseline_grid_mesh()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.remove_3D_time_curves_baseline_grid_mesh()


    # CELL ==================================================================================================================== #
    list(active_curve_plotter_3d.plots.keys())


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots.time_curve_helpers


    # CELL ==================================================================================================================== #
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.Render3DTimeCurvesBaseGridMixin import BaseGrid3DTimeCurvesHelper, Render3DTimeCurvesBaseGridMixin


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.init_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.add_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.remove_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_main_alpha = 0.5
    active_curve_plotter_3d.update_3D_time_curves()


    # CELL ==================================================================================================================== #
    # Add default params if needed:
    # active_curve_plotter_3d.params


    # CELL ==================================================================================================================== #
    list(active_curve_plotter_3d.params.keys())


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_z_baseline


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots


    # CELL ==================================================================================================================== #
    'time_curve_helpers' not in active_curve_plotter_3d.plots


    # CELL ==================================================================================================================== #
    'plots_grid_3dCurveBaselines_Grid' not in active_curve_plotter_3d.plots.time_curve_helpers


    # CELL ==================================================================================================================== #
    time_curves_z_baseline = 5.0 

    data_series_baseline
    # z_map_fn = lambda v_main: v_main + 5.0 # returns the un-transformed primary value

    5.0


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.axes_walls_z_height = 15.0


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d._update_axes_plane_graphics()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_axis_length


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_zoom_factor # 2.6666666666666665


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_to_spatial(temporal_data=[1.0])

    # CELL ==================================================================================================================== #
    line_color = pg.mkColor(plot_args.setdefault('color_name', 'white'))
    line_color.setAlphaF(0.8)

# ==================================================================================================================== #
# Pre- 2022-07-11                                                                                                      #
# ==================================================================================================================== #

def process_by_good_placefields(session, active_config, active_placefields):
    """  Filters the session by the units in active_placefields that have good placefields and return an updated session. Also adds generated colors for each good unit to active_config """
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    print('good_placefield_neuronIDs: {}; ({} good)'.format(good_placefield_neuronIDs, len(good_placefield_neuronIDs)))

    ## Filter by neurons with good placefields only:
    good_placefields_session = session.get_by_id(good_placefield_neuronIDs) # Filter by good placefields only, and this fetch also ensures they're returned in the order of sorted ascending index ([ 2  3  5  7  9 12 18 21 22 23 26 27 29 34 38 45 48 53 57])

    pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    active_config.plotting_config.pf_sort_ind = pf_sort_ind
    active_config.plotting_config.pf_colors = pf_colors
    active_config.plotting_config.active_cells_colormap = pf_colormap
    active_config.plotting_config.active_cells_listed_colormap = ListedColormap(active_config.plotting_config.active_cells_colormap)
    
    return good_placefields_session, active_config, good_placefield_neuronIDs

def build_placefield_multiplotter(nfields, linear_plot_data=None):
    linear_plotter_indicies = np.arange(nfields)
    fixed_columns = 5
    needed_rows = int(np.ceil(nfields / fixed_columns))
    row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
    mp = pvqt.MultiPlotter(nrows=needed_rows, ncols=fixed_columns, show=False, title='Muliplotter', toolbar=False, menu_bar=False, editor=False)
    print('linear_plotter_indicies: {}\n row_column_indicies: {}\n'.format(linear_plotter_indicies, row_column_indicies))
    # mp[0, 0].add_mesh(pv.Sphere())
    # mp[0, 1].add_mesh(pv.Cylinder())
    # mp[1, 0].add_mesh(pv.Cube())
    # mp[1, 1].add_mesh(pv.Cone())
    for a_linear_index in linear_plotter_indicies:
        print('a_linear_index: {}, row_column_indicies[0][a_linear_index]: {}, row_column_indicies[1][a_linear_index]: {}'.format(a_linear_index, row_column_indicies[0][a_linear_index], row_column_indicies[1][a_linear_index]))
        curr_row = row_column_indicies[0][a_linear_index]
        curr_col = row_column_indicies[1][a_linear_index]
        if linear_plot_data is None:
            mp[curr_row, curr_col].add_mesh(pv.Sphere())
        else:
            mp[curr_row, curr_col].add_mesh(linear_plot_data[a_linear_index], name='maze_bg', color="black", render=False)
            # mp[a_row_column_index[0], a_row_column_index[1]].add_mesh(pv.Sphere())
    return mp, linear_plotter_indicies, row_column_indicies

#TODO: Works, but need to convert into the computation function format or find a new place to put it. It operates on the entire pipeline while currently computation functions are limited to operating on one stage at a time.
def _perform_PBE_stats(active_pipeline, debug_print = False):
    """ # Analyze PBEs by looping through the filtered epochs:
        This whole implementation seems silly and inefficient        
        Can't I use .agg(['count', 'mean']) or something? 
        
        
    Usage:
        from PendingNotebookCode import _perform_PBE_stats
        pbe_analyses_result_df, [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] = _perform_PBE_stats(curr_active_pipeline, debug_print=False) # all_epochs_n_pbes: [206, 31, 237], all_epochs_mean_pbe_durations: [0.2209951456310722, 0.23900000000001073, 0.22335021097046923], all_epochs_cummulative_pbe_durations: [45.52500000000087, 7.409000000000333, 52.934000000001205], all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, 1910.1600048116618]
        pbe_analyses_result_df

    """
    all_epochs_labels = []
    all_epochs_total_durations = []
    all_epochs_n_pbes = []
    all_epochs_pbe_duration_lists = []
    all_epochs_cummulative_pbe_durations = []
    all_epochs_mean_pbe_durations = []
    all_epochs_full_pbe_spiketrain_lists = []
    all_epochs_pbe_num_spikes_lists = []
    all_epochs_intra_pbe_interval_lists = []
    
    for (name, filtered_sess) in active_pipeline.filtered_sessions.items():
        # interested in analyzing both the filtered_sess.pbe and the filtered_sess.spikes_df (as they relate to the PBEs)
        all_epochs_labels.append(name)
        curr_named_time_range = active_pipeline.sess.epochs.get_named_timerange(name) # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
        
        if not np.isscalar(curr_named_time_range.duration):
            # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
            curr_named_time_range = NamedTimerange(name='maze', start_end_times=[active_pipeline.sess.epochs['maze1'][0], active_pipeline.sess.epochs['maze2'][1]])
        
        curr_epoch_duration = curr_named_time_range.duration
        all_epochs_total_durations.append(curr_epoch_duration) # TODO: this should be in seconds (or at least the same units as the PBE durations)... actually this might be right.
        # Computes the intervals between each PBE:
        curr_intra_pbe_intervals = filtered_sess.pbe.starts[1:] - filtered_sess.pbe.stops[:-1]
        all_epochs_intra_pbe_interval_lists.append(curr_intra_pbe_intervals)
        all_epochs_n_pbes.append(filtered_sess.pbe.n_epochs)
        all_epochs_pbe_duration_lists.append(filtered_sess.pbe.durations)
        all_epochs_cummulative_pbe_durations.append(np.sum(filtered_sess.pbe.durations))
        all_epochs_mean_pbe_durations.append(np.nanmean(filtered_sess.pbe.durations))
        # filtered_sess.spikes_df.PBE_id
        curr_pbe_only_spikes_df = filtered_sess.spikes_df[filtered_sess.spikes_df.PBE_id > -1].copy()
        unique_PBE_ids = np.unique(curr_pbe_only_spikes_df['PBE_id'])
        flat_PBE_ids = [int(id) for id in unique_PBE_ids]
        num_unique_PBE_ids = len(flat_PBE_ids)
        # groups the spikes_df by PBEs:
        curr_pbe_grouped_spikes_df = curr_pbe_only_spikes_df.groupby(['PBE_id'])
        curr_spiketrains = list()
        curr_PBE_spiketrain_num_spikes = list()
        for i in np.arange(num_unique_PBE_ids):
            curr_PBE_id = flat_PBE_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_PBE_dataframe = curr_pbe_grouped_spikes_df.get_group(curr_PBE_id)
            curr_PBE_num_spikes = np.shape(curr_PBE_dataframe)[0] # the number of spikes in this PBE
            curr_PBE_spiketrain_num_spikes.append(curr_PBE_num_spikes)
            curr_spiketrains.append(curr_PBE_dataframe['t'].to_numpy())

        curr_PBE_spiketrain_num_spikes = np.array(curr_PBE_spiketrain_num_spikes)
        all_epochs_pbe_num_spikes_lists.append(curr_PBE_spiketrain_num_spikes)
        curr_spiketrains = np.array(curr_spiketrains, dtype='object')
        all_epochs_full_pbe_spiketrain_lists.append(curr_spiketrains)
        if debug_print:
            print(f'name: {name}, filtered_sess.pbe: {filtered_sess.pbe}')

    if debug_print:
        print(f'all_epochs_n_pbes: {all_epochs_n_pbes}, all_epochs_mean_pbe_durations: {all_epochs_mean_pbe_durations}, all_epochs_cummulative_pbe_durations: {all_epochs_cummulative_pbe_durations}, all_epochs_total_durations: {all_epochs_total_durations}')
        # all_epochs_n_pbes: [3152, 561, 1847, 832, 4566], all_epochs_mean_pbe_durations: [0.19560881979695527, 0.22129233511594312, 0.19185056848946497, 0.2333112980769119, 0.1987152869032212]

    all_epochs_pbe_occurance_rate = [(float(all_epochs_total_durations[i]) / float(all_epochs_n_pbes[i])) for i in np.arange(len(all_epochs_n_pbes))]
    all_epochs_pbe_percent_duration = [(float(all_epochs_total_durations[i]) / float(all_epochs_cummulative_pbe_durations[i])) for i in np.arange(len(all_epochs_n_pbes))]    
    all_epoch_mean_num_pbe_spikes = [np.nanmean(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [3151, 561, 1847, 831, 4563]
    all_epoch_std_num_pbe_spikes = [np.nanstd(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [11.638970035733648, 15.013817202645336, 15.5123897729991, 15.113395025612247, 11.473087401691878]
    # [20.429704855601397, 27.338680926916222, 23.748781808337846, 25.673886883273166, 20.38614946307254]
    # Build the final output result dataframe:
    pbe_analyses_result_df = pd.DataFrame({'n_pbes':all_epochs_n_pbes, 'mean_pbe_durations': all_epochs_mean_pbe_durations, 'cummulative_pbe_durations':all_epochs_cummulative_pbe_durations, 'epoch_total_duration':all_epochs_total_durations,
                'pbe_occurance_rate':all_epochs_pbe_occurance_rate, 'pbe_percent_duration':all_epochs_pbe_percent_duration,
                'mean_num_pbe_spikes':all_epoch_mean_num_pbe_spikes, 'stddev_num_pbe_spikes':all_epoch_std_num_pbe_spikes}, index=all_epochs_labels)
    # temporary: this isn't how the returns work for other computation functions:
    all_epochs_info = [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] # list version
    # all_epochs_info = {'all_epochs_full_pbe_spiketrain_lists':all_epochs_full_pbe_spiketrain_lists, 'all_epochs_pbe_num_spikes_lists':all_epochs_pbe_num_spikes_lists, 'all_epochs_intra_pbe_interval_lists':all_epochs_intra_pbe_interval_lists} # dict version
    return pbe_analyses_result_df, all_epochs_info

