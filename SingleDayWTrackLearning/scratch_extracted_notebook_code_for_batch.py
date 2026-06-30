



## Manual Recomputations:
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import final_process_bapun_all_comps


curr_active_pipeline = final_process_bapun_all_comps(curr_active_pipeline=curr_active_pipeline, active_data_mode_name=active_data_mode_name,
                                                    posthoc_save=False,
                                                    time_bin_size = 0.500,
                                                    # time_bin_size=0.250,
                                                    # overwrite_extant = False,
                                                    overwrite_extant = True,
                                                    # fail_on_exception = False,
                                                    fail_on_exception=True,
)

## 9m
# desired_time_bin_size = 0.010 # 10ms
desired_time_bin_size = 0.250 # 250ms
curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

## 13m at 250ms
curr_active_pipeline.rerun_failed_computations()
curr_active_pipeline.get_failed_computations()


# DISPLAY PORTION

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlacefieldsPlotter import TimeSynchronizedPlacefieldsPlotter

_restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

#  Create a new `SpikeRaster2D` instance using `_display_spike_raster_pyqtplot_2D` and capture its outputs:
curr_active_pipeline.reload_default_display_functions()
curr_active_pipeline.prepare_for_display()



## ✅ 2025-09-19 - Clean programmmatic figure outputs 

from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D, capture_new_figures_decorator
fig_man = PhoActiveFigureManager2D(name=f'fig_man') # Initialize a new figure manager
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import programmatic_render_to_file, programmatic_display_to_PDF, extract_figures_from_display_function_output
from neuropy.core.session.Formats.BaseDataSessionFormats import HardcodedProcessingParameters
from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

hardcoded_params: HardcodedProcessingParameters = NWBDataSessionFormatRegisteredClass._get_session_specific_parameters(session_context=curr_active_pipeline.get_session_context())
# hardcoded_params

# hardcoded_params.decoder_building_session_names
# hardcoded_params.non_global_activity_session_names

fig_man.close_all()

# subset_includelist = ['maze1', 'maze2', 'maze_GLOBAL'] # Day5TwoNovel
# subset_includelist = ['roam', 'sprinkle'] # Day4

subset_includelist = hardcoded_params.decoder_building_session_names
print(f'subset_includelist: {subset_includelist}')

display_fn_kwargs = dict(subplots=(None, 9),
    fig_column_width=None,   # key fix — uses data aspect ratio for width
    fig_row_height=1.0,
    resolution_multiplier=1.0,
)

# display_fn_kwargs = dict(subplots=(None, 5))

# _out = dict()
# _out['_display_2d_placefield_result_plot_ratemaps_2D'] = curr_active_pipeline.display(display_function='_display_2d_placefield_result_plot_ratemaps_2D', active_session_configuration_context=IdentifyingContext(format_name='bapun',animal='RatS',session_name='Day5TwoNovel',filter_name='maze1'), **display_fn_kwargs) # _display_2d_placefield_result_plot_ratemaps_2D
# _out['_display_2d_placefield_result_plot_ratemaps_2D'] = curr_active_pipeline.display(display_function='_display_2d_placefield_result_plot_ratemaps_2D', active_session_configuration_context=IdentifyingContext(format_name='bapun',animal='RatS',session_name='Day5TwoNovel',filter_name='maze2'), **display_fn_kwargs) # _display_2d_placefield_result_plot_ratemaps_2D
_out_list = programmatic_render_to_file(curr_active_pipeline=curr_active_pipeline, curr_display_function_name='_display_2d_placefield_result_plot_ratemaps_2D', subset_includelist=subset_includelist, 
                                        write_vector_format=True, write_png=True, debug_print=True, **display_fn_kwargs)
_out_list = programmatic_render_to_file(curr_active_pipeline=curr_active_pipeline, curr_display_function_name='_display_2d_placefield_occupancy', subset_includelist=subset_includelist, 
                                        write_vector_format=True, write_png=True, debug_print=True)
_out_list = programmatic_render_to_file(curr_active_pipeline=curr_active_pipeline, curr_display_function_name='_display_1d_placefields', subset_includelist=subset_includelist, 
                                        write_vector_format=True, write_png=True, debug_print=True, **display_fn_kwargs)



## `Spike3DRasterWindowWidget` Cell
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _setup_spike_raster_window_for_debugging
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import ScatterItemData
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import Spike3DRasterWindowWidget # used in `NewSimpleRaster`
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_proper_epoch_intervals

global_context: IdentifyingContext = curr_active_pipeline.filtered_contexts['maze_GLOBAL']
global_context

# Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
spike_raster_window, (active_2d_plot, active_3d_plot, *_all_outputs_dict) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline, force_create_new=True, allow_replace_hardcoded_main_plots_with_tracks=True,
     active_session_configuration_context=global_context,
)

## set correct intervals
a_rect_item, an_interval_ds = build_proper_epoch_intervals(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, height=1.5)
# preview_overview_scatter_plot: pg.ScatterPlotItem  = active_2d_plot.plots.preview_overview_scatter_plot # ScatterPlotItem 
# preview_overview_scatter_plot.setDownsampling(auto=True, method='subsample', dsRate=10)
# main_graphics_layout_widget: pg.GraphicsLayoutWidget = active_2d_plot.ui.main_graphics_layout_widget
wrapper_layout: pg.QtWidgets.QVBoxLayout = active_2d_plot.ui.wrapper_layout
# main_content_splitter = active_2d_plot.ui.main_content_splitter # QSplitter
layout = active_2d_plot.ui.layout
background_static_scroll_window_plot = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
# active_window_container_layout = active_2d_plot.ui.active_window_container_layout # GraphicsLayout, first item of `main_graphics_layout_widget` -- just the active raster window I think, there is a strange black space above it


# from pyphoplacecellanalysis.External.pyqtgraph_extensions.trapezoid_callout import TrapezoidOverlay

# _out_overlays: Dict[Tuple, TrapezoidOverlay] = TrapezoidOverlay.add_overview_indicator_trapazoids_to_timeline(active_2d_plot=active_2d_plot)


## add context decoder
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import (
    AddNewDecodedEpochMarginal_MatplotlibPlotCommand,
    decoding_continuous_cache_key,
)

# 1) Get the SpikeRaster2D widget
spike_raster_window, (active_2d_plot, *_rest) = Spike3DRasterWindowWidget.find_or_create_if_needed(
    curr_active_pipeline,
    force_create_new=False,
    allow_replace_hardcoded_main_plots_with_tracks=True,
    active_session_configuration_context='maze_GLOBAL',
)

# 2) Match menu defaults — for Bapun, usually only the context-marginal row
active_2d_plot.params.enable_non_marginalized_raw_result = False
active_2d_plot.params.enable_marginal_over_direction = False
active_2d_plot.params.enable_marginal_over_track_ID = True

# 3) Same as menu: whitelist must use the CACHE KEY (tuple), not bare 0.25
cache_key = decoding_continuous_cache_key(0.25, None)  # -> (0.25, 0.25)

cmd = AddNewDecodedEpochMarginal_MatplotlibPlotCommand(
    spike_raster_window,
    curr_active_pipeline,
    active_time_bin_sizes_whitelist=[cache_key],  # NOT [0.25] and NOT [[0.25, 0.25]]
)
cmd.execute()
from copy import deepcopy
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedEpochMarginal_MatplotlibPlotCommand

dd = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
cache_key = (0.25, 0.25)
continuously_decoded_dict = dd.continuously_decoded_result_cache_dict[cache_key]
info_string = f" - t_bin_size: {cache_key}"

output_dict = AddNewDecodedEpochMarginal_MatplotlibPlotCommand.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(
    curr_active_pipeline=curr_active_pipeline,
    active_2d_plot=active_2d_plot,
    continuously_decoded_dict=deepcopy(continuously_decoded_dict),
    info_string=info_string,
    enable_non_marginalized_raw_result=False,
    enable_marginal_over_direction=False,
    enable_marginal_over_track_ID=True,
)
output_dict  # keys: 'marginal_over_track_ID', etc.
identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict = output_dict['marginal_over_track_ID']
a_dock = active_2d_plot.find_display_dock(identifier_name)

a_dock.size() # PyQt5.QtCore.QSize(3778, 129)
a_dock.height() # 129
a_dock.setFixedHeight(130)


# ⚓✅💾 Export ALL tracks (both plotting backends)
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mimage
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FigureToImageHelpers
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum

# relative_data_output_parent_folder = Path('data').resolve()

relative_data_output_parent_folder = curr_active_pipeline.get_output_path().resolve()
Assert.path_exists(relative_data_output_parent_folder)

## INPUTS: im_posterior_x_stack, track_labels, 
output_pdf_path: Path = relative_data_output_parent_folder.joinpath('2026-06-30_all_timeline_tracks_exported_stack.pdf')

# included_track_dock_identifiers = None
included_track_dock_identifiers = [
	# 'interval_overview',
	'intervals',
	# 'rasters[raster_overview]',
	'rasters[raster_window]',
	'new_curves_separate_plot',
	# 'mpl_position_curves',
	#  'MenuCommand_display_plot_marginal_1D_most_likely_position_comparisons',
	#  'global context',
	#  'global context (overview),
    'marginal_over_track_ID_ContinuousDecode - t_bin_size: (0.25, 0.25)',
]
included_track_dock_identifiers = list(reversed(included_track_dock_identifiers))

# track_labels: List[str] = list(included_track_dock_identifiers_to_track_labels_dict.values())
track_labels = None
saved_output_pdf_path = FigureToImageHelpers.export_wrapped_tracks_to_paged_df(active_2d_plot, output_pdf_path=output_pdf_path, included_track_dock_identifiers=included_track_dock_identifiers, track_labels=track_labels, debug_max_num_pages=250)

## OUTPUTS: output_pdf_path, included_track_dock_identifiers

out_path = None
_render_export_all_time_tracks = active_2d_plot.export_all_tracks_to_image(custom_figure_output_path=out_path, curr_active_pipeline=curr_active_pipeline)
_render_export_all_time_tracks

