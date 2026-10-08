# -*- coding: utf-8 -*-
"""Restructure RadonTransform + OverlayLabels providers per rename plan."""
from pathlib import Path

p = Path(r'h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\DisplayFunctions\DecoderPredictionError.py')
text = p.read_text(encoding='utf-8')

old_radon_header = '''class RadonTransformPlotDataProvider(PaginatedPlotDataProvider):
    """ Adds the yellow Radon Transform fit line and scoring band to the posterior heatmap.

    `.add_data_to_pagination_controller(...)` adds the result to the pagination controller

    Data:
        plots_data.radon_transform_data
    Plots:
        plots['radon_transform']
        
            _out_pagination_controller.plots_data.radon_transform_data = radon_transform_data
        _out_pagination_controller.plots['radon_transform'] = {}


    Usage:

    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider


    """
    callback_identifier_string: str = 'plot_radon_transform_line_data'
    plots_group_identifier_key: str = 'radon_transform' # _out_pagination_controller.plots['overlay_labels']
    plots_group_data_identifier_key: str = 'radon_transform_data'
    
    provided_params: Dict[str, Any] = dict(enable_radon_transform_info=True, enable_radon_transform_line=True, enable_radon_transform_scoring_band=True, radon_transform_margin=4.0, radon_transform_n_neighbours=None)
    provided_plots_data: Dict[str, Any] = {'radon_transform_data': None}
    provided_plots: Dict[str, Any] = {'radon_transform': {}}

    column_names: List[str] = ['radon', 'velocity', 'intercept', 'speed']

    # Above heuristic sequence hlines (~5) and direction-change lines (~22)
    OVERLAY_LABEL_ZORDER: int = 50
    # Radon label sits one step above wcorr so a transient overlap still draws radon on top
    OVERLAY_RADON_LABEL_ZORDER: int = OVERLAY_LABEL_ZORDER + 1
    

    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        return {'on_render_page_callbacks': 
                {'plot_radon_transform_line_data': cls._callback_update_curr_single_epoch_slice_plot}
        }


    @classmethod
    def _axes_y_below_anchored_artist(cls, curr_ax, anchored_artist, pad_axes: float = 0.0, fallback_y: float = 0.78) -> float:
        """Return axes-fraction Y for the top of the next stacked label below `anchored_artist`.

        Prefers the last TextArea of AnchoredCustomText so AnchoredOffsetbox borderpad does not inflate the gap.
        """
        if anchored_artist is None:
            return float(fallback_y)
        try:
            fig = curr_ax.get_figure()
            renderer = fig.canvas.get_renderer()
            # Prefer last text line extent (tighter stack) when available
            text_areas = getattr(anchored_artist, 'text_areas', None)
            if text_areas:
                bbox_disp = text_areas[-1]._text.get_window_extent(renderer=renderer)
            else:
                bbox_disp = anchored_artist.get_window_extent(renderer=renderer)
            bbox_axes = bbox_disp.transformed(curr_ax.transAxes.inverted())
            y = float(bbox_axes.y0) - float(pad_axes)
            return max(y, 0.05)
        except Exception:
            return float(fallback_y)


    @classmethod
    def _overlay_stack_anchor_artist(cls, curr_ax, plots, wcorr_anchored_text=None, heuristic_anchored_text=None):
        """Bottom-most extant overlay label: heuristic (green) if present, else wcorr (blue)."""
        plots_dict = plots.get('weighted_corr', {}).get(curr_ax, {})
        heuristic_text = heuristic_anchored_text if (heuristic_anchored_text is not None) else plots_dict.get('heuristic_text', None)
        wcorr_text = wcorr_anchored_text if (wcorr_anchored_text is not None) else plots_dict.get('wcorr_text', None)
        for artist in (heuristic_text, wcorr_text):
            if (artist is not None) and (getattr(artist, 'axes', None) is not None):
                return artist
        ## END for artist in (heuristic_text, wcorr_text)...
        return None


    @classmethod
    def _resolve_radon_label_bbox_y(cls, curr_ax, plots, params, pad_axes: float = 0.0) -> float:
        """Axes-fraction Y for the radon label: below extant overlay stack, else top-right (1.0)."""
        fallback_y: float = float(params.setdefault('radon_label_bbox_y', 0.78))
        anchor = cls._overlay_stack_anchor_artist(curr_ax, plots)
        if anchor is not None:
            return cls._axes_y_below_anchored_artist(curr_ax, anchor, pad_axes=pad_axes, fallback_y=fallback_y)
        return 1.0


    @classmethod
    def _reposition_radon_text_below_wcorr(cls, curr_ax, plots, data_idx, params, wcorr_anchored_text=None, heuristic_anchored_text=None, pad_axes: float = 0.0) -> None:
        """After wcorr/heuristic label heights are known, move extant radon_text just below the stack."""
        radon_text_artist = plots.get(cls.plots_group_identifier_key, {}).get(data_idx, {}).get('radon_text', None)
        if radon_text_artist is None:
            return
        fallback_y: float = float(params.setdefault('radon_label_bbox_y', 0.78))
        anchor = cls._overlay_stack_anchor_artist(curr_ax, plots, wcorr_anchored_text=wcorr_anchored_text, heuristic_anchored_text=heuristic_anchored_text)
        if anchor is not None:
            y = cls._axes_y_below_anchored_artist(curr_ax, anchor, pad_axes=pad_axes, fallback_y=fallback_y)
        else:
            y = 1.0
        radon_text_artist.set_bbox_to_anchor((1.0, y), transform=curr_ax.transAxes)
        radon_text_artist.set_zorder(cls.OVERLAY_RADON_LABEL_ZORDER)

    @classmethod
    def _subfn_build_radon_transform_plotting_data'''

new_radon_header = '''class RadonTransformPlotDataProvider(PaginatedPlotDataProvider):
    """ Adds the yellow Radon Transform fit line and scoring band to the posterior heatmap.

    Corner-score text (`radon` / speed / intercept labels) is owned by `OverlayLabelsPaginatedPlotDataProvider`;
    this provider still builds `RadonTransformPlotData.radon_text` strings for that consumer.

    `.add_data_to_pagination_controller(...)` adds the result to the pagination controller

    Data:
        plots_data.radon_transform_data
    Plots:
        plots['radon_transform']  # {'line', 'band'} per data_idx

            _out_pagination_controller.plots_data.radon_transform_data = radon_transform_data
        _out_pagination_controller.plots['radon_transform'] = {}


    Usage:

    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import RadonTransformPlotDataProvider

    History:
        Corner `radon_text` artists were previously drawn here; stacking helpers
        `_reposition_radon_text_below_wcorr` / `_overlay_stack_anchor_artist` / `_resolve_radon_label_bbox_y`
        / `_axes_y_below_anchored_artist` removed when label ownership moved to OverlayLabels.
    """
    callback_identifier_string: str = 'plot_radon_transform_line_data'
    plots_group_identifier_key: str = 'radon_transform' # _out_pagination_controller.plots['radon_transform']
    plots_group_data_identifier_key: str = 'radon_transform_data'
    
    provided_params: Dict[str, Any] = dict(enable_radon_transform_info=True, enable_radon_transform_line=True, enable_radon_transform_scoring_band=True, radon_transform_margin=4.0, radon_transform_n_neighbours=None)
    provided_plots_data: Dict[str, Any] = {'radon_transform_data': None}
    provided_plots: Dict[str, Any] = {'radon_transform': {}}

    column_names: List[str] = ['radon', 'velocity', 'intercept', 'speed']


    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        return {'on_render_page_callbacks': 
                {'plot_radon_transform_line_data': cls._callback_update_curr_single_epoch_slice_plot}
        }


    @classmethod
    def _subfn_build_radon_transform_plotting_data'''

assert old_radon_header in text, 'radon header block not found'
text = text.replace(old_radon_header, new_radon_header, 1)
print('replaced radon header+helpers')

marker_start = '    @classmethod\n    def _callback_update_curr_single_epoch_slice_plot(cls, curr_ax, params: "VisualizationParameters", plots_data: "RenderPlotsData", plots: "RenderPlots", ui: "PhoUIContainer", data_idx:int, curr_time_bins, *args, epoch_slice=None, curr_time_bin_container=None, **kwargs): # curr_posterior, curr_most_likely_positions, debug_print:bool=False\n        """ 2023-05-30 - Based off of `_helper_update_decoded_single_epoch_slice_plot` to enable plotting radon transform lines on paged decoded epochs'
idx = text.find(marker_start)
assert idx != -1, 'radon callback start not found'
end_marker = '\n# ====================================================================================================================\n# Score: Weighted Correlation'
end_idx = text.find(end_marker, idx)
assert end_idx != -1, 'score section not found'

new_radon_callback = '''    @classmethod
    def _callback_update_curr_single_epoch_slice_plot(cls, curr_ax, params: "VisualizationParameters", plots_data: "RenderPlotsData", plots: "RenderPlots", ui: "PhoUIContainer", data_idx:int, curr_time_bins, *args, epoch_slice=None, curr_time_bin_container=None, **kwargs): # curr_posterior, curr_most_likely_positions, debug_print:bool=False
        """ Draw radon fit line + scoring band only. Corner labels are owned by OverlayLabelsPaginatedPlotDataProvider.
        """
        from matplotlib.patches import Polygon

        def _subfn_build_kwargs(curr_ax):
            radon_theme_color: str = '#ffee00' ## a dark yellow/orange
            # Solid yellow line (same theme as former radon score text stroke)
            line_kwargs = dict(scalex=False, scaley=False, label='computed radon transform', linestyle='-', linewidth=0.5, color=radon_theme_color, alpha=0.85, marker=None, zorder=3)

            fn_rgb255_to_rgbF = lambda arr: [(float(v)/float(255)) for v in arr]
            fRGB_band_facecolor = fn_rgb255_to_rgbF([235, 192, 52])
            fRGB_band_edgecolor = fn_rgb255_to_rgbF([235, 177, 52])
            
            band_kwargs = dict(facecolor=(*fRGB_band_facecolor, 0.35), edgecolor=(*fRGB_band_edgecolor, 0.85), linewidth=1.0)

            return line_kwargs, band_kwargs


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        line_kwargs, band_kwargs = _subfn_build_kwargs(curr_ax)
        debug_print = kwargs.pop('debug_print', True)

        ## Extract the visibility:
        should_enable_radon_transform_info: bool = params.enable_radon_transform_info
        should_enable_radon_transform_line: bool = bool(params.setdefault('enable_radon_transform_line', True)) and should_enable_radon_transform_info
        should_enable_radon_transform_scoring_band: bool = bool(params.setdefault('enable_radon_transform_scoring_band', True)) and should_enable_radon_transform_info

        if debug_print:
            print(f'{params.name}: _callback_update_curr_single_epoch_slice_plot(..., data_idx: {data_idx}, curr_time_bins: {curr_time_bins})')
        
        extant_plots = plots[cls.plots_group_identifier_key].get(data_idx, {})
        extant_line = extant_plots.get('line', None)
        extant_band = extant_plots.get('band', None)
        # Legacy key may remain from older pages; drop orphaned radon_text if present
        extant_radon_text = extant_plots.get('radon_text', None)
        if (extant_line is not None) or (extant_band is not None) or (extant_radon_text is not None):
            if extant_line is not None:
                extant_line.remove()
            extant_line = None
            if extant_radon_text is not None:
                extant_radon_text.remove()
            extant_radon_text = None
            if extant_band is not None:
                for band_artist in list(extant_band):
                    band_artist.remove()
                ## END for band_artist in list(extant_band)....
            extant_band = None

        if curr_time_bin_container is not None:
            actual_time_bins = curr_time_bin_container.centers
            if debug_print:
                print(f'actual_time_bins: {actual_time_bins}')
                print(f'curr_time_bins: {curr_time_bins}')

        else:
            if debug_print:
                print(f'No actual time bins!')
                print(f'curr_time_bins: {curr_time_bins}')
                
            actual_time_bins = deepcopy(curr_time_bins)

        # Scoring band (orange stair-step polygons) — below the yellow line
        radon_band_artists = []
        if should_enable_radon_transform_scoring_band:
            band_polygons = getattr(plots_data.radon_transform_data[data_idx], 'band_polygons', None)
            if band_polygons is not None:
                for xs, ys in band_polygons:
                    poly = Polygon(np.column_stack([xs, ys]), closed=True, **band_kwargs, zorder=2)
                    curr_ax.add_patch(poly)
                    radon_band_artists.append(poly)
                ## END for xs, ys in band_polygons....

        ## Plot the yellow geometric radon line
        if should_enable_radon_transform_line:
            curr_line_y = plots_data.radon_transform_data[data_idx].line_y
            real_line_extrapolated_t = np.squeeze(curr_time_bins)
            real_line_extrapolated_x = np.interp(real_line_extrapolated_t, xp=np.squeeze(actual_time_bins), fp=np.squeeze(curr_line_y))
            radon_transform_plot, = curr_ax.plot(real_line_extrapolated_t, real_line_extrapolated_x, **line_kwargs)
        else:
            radon_transform_plot = None

        # Store the plot objects for future updates (text labels live under plots['overlay_labels']):
        plots[cls.plots_group_identifier_key][data_idx] = {'line': radon_transform_plot, 'band': radon_band_artists}
        
        if debug_print:
            print(f'\\t success!')

        return params, plots_data, plots, ui




'''

text = text[:idx] + new_radon_callback + text[end_idx:]
print('replaced radon callback')

old_section = '''# ==================================================================================================================== #
# Score: Weighted Correlation                                                                                          #
# ==================================================================================================================== #
is_integer = lambda v: isinstance(v, (int, np.integer))
is_float = lambda v: isinstance(v, (float, np.floating))

@define(slots=False, repr=False)
class OverlayLabelsPlotData:

    data_values_dict: Dict[str, Optional[Any]] = field(factory=dict)'''

new_section = '''# ==================================================================================================================== #
# Overlay Labels (wcorr / heuristic / radon corner text)                                                               #
# ==================================================================================================================== #
is_integer = lambda v: isinstance(v, (int, np.integer))
is_float = lambda v: isinstance(v, (float, np.floating))

@define(slots=False, repr=False)
class OverlayLabelsPlotData:
    """ Per-epoch DF-derived corner-label strings (wcorr / heuristic scores).

    History:
        Formerly named `WeightedCorrelationPlotData`.
    """

    data_values_dict: Dict[str, Optional[Any]] = field(factory=dict)'''

assert old_section in text, 'overlay plot data section not found'
text = text.replace(old_section, new_section, 1)
print('updated OverlayLabelsPlotData docstring')

old_provider_doc = '''class OverlayLabelsPaginatedPlotDataProvider(PaginatedPlotDataProvider):
    """ NOTE: This class currently provides more than just weighted correlation data, in fact it is suitable for rendering any subplot-dependent computed quantity.
    Currently displays: WCorr, P_decoder (1D decoder probability of the four different decoders), simple correlation pearson r value

    Data:
        plots_data.overlay_labels_data
    Plots:
        plots['overlay_labels']
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import OverlayLabelsPaginatedPlotDataProvider, OverlayLabelsPlotData
    """
    plots_group_identifier_key: str = 'overlay_labels' # _out_pagination_controller.plots['overlay_labels']

    # text_color: str = '#ff886a' # an orange
    # text_color: str = '#42D142' # a light green
    # text_color: str = '#66FF00' # a light green
    text_color: str = '#013220' # a very dark forest green
    theme_stroke_color: str = '#00B0FF' # bright blue outline for wcorr block (mirrors radon's yellow stroke style)
    heuristic_theme_stroke_color: str = '#00C853' # green outline for heuristic score lines (coverage, mseq_*, …)

    provided_params: Dict[str, Any] = dict(enable_overlay_labels_info=True, enable_overlay_labels_modify_axes_rect=False, overlay_labels_text_original_figure_rect=None, overlay_labels_text_was_axes_rect_modified=False)
    provided_plots_data: Dict[str, Any] = {'overlay_labels_data': None}
    provided_plots: Dict[str, Any] = {'overlay_labels': {}}
    column_names: List[str] = ['wcorr', 'P_decoder', 'pearsonr', 'travel', 'coverage', 'total_congruent_direction_change', 'longest_sequence_length', 'avg_jump_cm']'''

new_provider_doc = '''class OverlayLabelsPaginatedPlotDataProvider(PaginatedPlotDataProvider):
    """ Corner-score overlay labels for paginated decoded-epoch slices.

    Draws stacked AnchoredCustomText blocks: wcorr (blue), heuristic scores (green), radon (yellow).
    Radon line/band geometry remains on `RadonTransformPlotDataProvider`; radon label strings are read
    from `plots_data.radon_transform_data[data_idx].build_display_text(...)`.

    Data:
        plots_data.overlay_labels_data  # DF-derived wcorr/heuristic (optional)
        plots_data.radon_transform_data  # optional; used for radon corner text only
    Plots:
        plots['overlay_labels']  # {'wcorr_text', 'heuristic_text', 'radon_text'} keyed by curr_ax
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import OverlayLabelsPaginatedPlotDataProvider, OverlayLabelsPlotData

    History:
        Formerly `WeightedCorrelationPaginatedPlotDataProvider` / `WeightedCorrelationPlotData`.
        Former plots key `weighted_corr`, plots_data key `weighted_corr_data`.
        Former params: `enable_weighted_correlation_info`, `enable_weighted_corr_data_provider_modify_axes_rect`,
        `weighted_corr_text_original_figure_rect`, `weighted_corr_text_was_axes_rect_modified`.
        Former builder: `decoder_build_single_weighted_correlation_data`.
        Former callback id: `plot_wcorr_data`.
        Radon corner text previously lived on `RadonTransformPlotDataProvider`.
    """
    plots_group_identifier_key: str = 'overlay_labels' # _out_pagination_controller.plots['overlay_labels']

    # text_color: str = '#ff886a' # an orange
    # text_color: str = '#42D142' # a light green
    # text_color: str = '#66FF00' # a light green
    text_color: str = '#013220' # a very dark forest green
    theme_stroke_color: str = '#00B0FF' # bright blue outline for wcorr block (mirrors radon's yellow stroke style)
    heuristic_theme_stroke_color: str = '#00C853' # green outline for heuristic score lines (coverage, mseq_*, …)
    radon_theme_stroke_color: str = '#ffee00' # yellow outline for radon score lines

    # Above heuristic sequence hlines (~5) and direction-change lines (~22)
    OVERLAY_LABEL_ZORDER: int = 50
    OVERLAY_RADON_LABEL_ZORDER: int = OVERLAY_LABEL_ZORDER + 1

    provided_params: Dict[str, Any] = dict(enable_overlay_labels_info=True, enable_overlay_labels_modify_axes_rect=False, overlay_labels_text_original_figure_rect=None, overlay_labels_text_was_axes_rect_modified=False)
    provided_plots_data: Dict[str, Any] = {'overlay_labels_data': None}
    provided_plots: Dict[str, Any] = {'overlay_labels': {}}
    column_names: List[str] = ['wcorr', 'P_decoder', 'pearsonr', 'travel', 'coverage', 'total_congruent_direction_change', 'longest_sequence_length', 'avg_jump_cm']'''

assert old_provider_doc in text, 'provider docstring block not found'
text = text.replace(old_provider_doc, new_provider_doc, 1)
print('updated OverlayLabels provider docstring')

old_get_cols = '''    @classmethod
    def get_column_names(cls) -> List[str]:
        return OverlayLabelsPlotData.get_column_names()
    

    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        return {'on_render_page_callbacks': 
                {'plot_overlay_labels_data': cls._callback_update_curr_single_epoch_slice_plot}
        }
        

    @classmethod
    def decoder_build_single_overlay_labels_data(cls, curr_results_obj, included_columns=None):'''

new_get_cols = '''    @classmethod
    def get_column_names(cls) -> List[str]:
        return OverlayLabelsPlotData.get_column_names()


    @classmethod
    def _axes_y_below_anchored_artist(cls, curr_ax, anchored_artist, pad_axes: float = 0.0, fallback_y: float = 0.78) -> float:
        """Return axes-fraction Y for the top of the next stacked label below `anchored_artist`.

        Prefers the last TextArea of AnchoredCustomText so AnchoredOffsetbox borderpad does not inflate the gap.
        """
        if anchored_artist is None:
            return float(fallback_y)
        try:
            fig = curr_ax.get_figure()
            renderer = fig.canvas.get_renderer()
            # Prefer last text line extent (tighter stack) when available
            text_areas = getattr(anchored_artist, 'text_areas', None)
            if text_areas:
                bbox_disp = text_areas[-1]._text.get_window_extent(renderer=renderer)
            else:
                bbox_disp = anchored_artist.get_window_extent(renderer=renderer)
            bbox_axes = bbox_disp.transformed(curr_ax.transAxes.inverted())
            y = float(bbox_axes.y0) - float(pad_axes)
            return max(y, 0.05)
        except Exception:
            return float(fallback_y)


    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        return {'on_render_page_callbacks': 
                {'plot_overlay_labels_data': cls._callback_update_curr_single_epoch_slice_plot}
        }
        

    @classmethod
    def decoder_build_single_overlay_labels_data(cls, curr_results_obj, included_columns=None):'''

assert old_get_cols in text, 'get_column_names block not found'
text = text.replace(old_get_cols, new_get_cols, 1)
print('moved _axes_y_below_anchored_artist onto OverlayLabels')

p.write_text(text, encoding='utf-8')
print('wrote', p)
