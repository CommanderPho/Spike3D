## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from typing import  List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt

from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

from collections import Counter # debug_detect_repeated_values

import scipy # pho_compute_rank_order

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses # for _compute_single_rank_order_shuffle


# ==================================================================================================================== #
# 2024-01-17 - Lap performance validation                                                                              #
# ==================================================================================================================== #
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalMergedDecodersResult

def _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size: float = 0.5, desired_ripple_decoding_time_bin_size: float = 0.1):
    """ 2024-01-17 - Pending refactor from ReviewOfWork_2024-01-17.ipynb 

    Makes a copy of the 'DirectionalMergedDecoders' result
    from PendingNotebookCode import _perform_variable_time_bin_lap_groud_truth_performance_testing    

    """
    ## Copy the default result:
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result = deepcopy(directional_merged_decoders_result)

    owning_pipeline_reference = curr_active_pipeline
    all_directional_pf1D_Decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder

    # Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result

    # Modifies alt_directional_merged_decoders_result, a copy of the original result, with new timebins
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

    ## Decode Laps:
    global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
    min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(global_any_laps_epochs_obj.to_dataframe()['duration'].to_numpy())
    laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002

    alt_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, debug_print=False)

    ## Decode Ripples:        
    global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
    min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
    ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_time_bin_size) # 10ms # 0.002
    alt_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(owning_pipeline_reference.sess.spikes_df), global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size)

    ## Post Compute Validations:
    alt_directional_merged_decoders_result.perform_compute_marginals()


    # global_any_laps_epochs_obj

    from neuropy.core.session.dataSession import Laps

    # takes 'laps_df' and 'result_laps_epochs_df' to add the ground_truth and the decoded posteriors:

    # Ensure it has the 'lap_track' column
    t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_df = laps_obj.to_dataframe()
    ## Compute the ground-truth information using the position information:
    laps_df = Laps._update_dataframe_maze_id_if_needed(laps_df, t_start, t_delta, t_end) # 'maze_id'
    # laps_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df, global_session=global_session)
    laps_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df, global_session=curr_active_pipeline.sess) # 'is_LR_dir', global_session is missing the last two laps
    laps_df


    ## 2024-01-17 - Updates the `a_directional_merged_decoders_result.laps_epochs_df` with both the ground-truth values and the decoded predictions

    ## Inputs: a_directional_merged_decoders_result, laps_df

    a_directional_merged_decoders_result: DirectionalMergedDecodersResult = alt_directional_merged_decoders_result

    ## Get the most likely direction/track from the decoded posteriors:
    all_directional_laps_filter_epochs_decoder_result_value: DecodedFilterEpochsResult = a_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = a_directional_merged_decoders_result.laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = a_directional_merged_decoders_result.laps_track_identity_marginals_tuple

    # a_directional_merged_decoders_result.ripple_epochs_df
    # a_directional_merged_decoders_result.laps_epochs_df
    result_laps_epochs_df: pd.DataFrame = a_directional_merged_decoders_result.laps_epochs_df

    ## Add the ground-truth results to the laps df:
    # add the 'maze_id' groud-truth column in:
    # result_laps_epochs_df['maze_id'] = laps_df['maze_id'] # this works despite the different size because of the index matching
    # ## add the 'is_LR_dir' groud-truth column in:
    # result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'] # this works despite the different size because of the index matching

    result_laps_epochs_df['maze_id'] = laps_df['maze_id'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
    ## add the 'is_LR_dir' groud-truth column in:
    result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching

    ## Add the decoded results to the laps df:
    result_laps_epochs_df['is_most_likely_track_identity_Long'] = laps_is_most_likely_track_identity_Long
    result_laps_epochs_df['is_most_likely_direction_LR'] = laps_is_most_likely_direction_LR_dir
    result_laps_epochs_df

    # np.sum(result_laps_epochs_df['is_LR_dir'] == result_laps_epochs_df['lap_dir'])/np.shape(result_laps_epochs_df)[0]
    np.sum(result_laps_epochs_df['is_LR_dir'] == result_laps_epochs_df['is_most_likely_direction_LR'])/np.shape(result_laps_epochs_df)[0]
    laps_decoding_time_bin_size = alt_directional_merged_decoders_result.laps_decoding_time_bin_size
    print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}')


    ## Uses only 'result_laps_epochs_df'

    def _check_result_laps_epochs_df_performance(result_laps_epochs_df: pd.DataFrame, debug_print=True):
        """ 2024-01-17 - Validates the performance of the pseudo2D decoder posteriors using the laps data.
        
        """
        # Check 'maze_id' decoding accuracy
        n_laps = np.shape(result_laps_epochs_df)[0]
        is_decoded_track_correct = (result_laps_epochs_df['maze_id'] == result_laps_epochs_df['is_most_likely_track_identity_Long'].apply(lambda x: 0 if x else 1))
        percent_laps_track_identity_estimated_correctly = (np.sum(is_decoded_track_correct) / n_laps)
        if debug_print:
            print(f'percent_laps_track_identity_estimated_correctly: {percent_laps_track_identity_estimated_correctly}')
        # Check 'is_LR_dir' decoding accuracy:
        is_decoded_dir_correct = (result_laps_epochs_df['is_LR_dir'].apply(lambda x: 0 if x else 1) == result_laps_epochs_df['is_most_likely_direction_LR'].apply(lambda x: 0 if x else 1))
        percent_laps_direction_estimated_correctly = (np.sum(is_decoded_dir_correct) / n_laps)
        if debug_print:
            print(f'percent_laps_direction_estimated_correctly: {percent_laps_direction_estimated_correctly}')

        # Both should be correct
        are_both_decoded_properties_correct = np.logical_and(is_decoded_track_correct, is_decoded_dir_correct)
        percent_laps_estimated_correctly = (np.sum(are_both_decoded_properties_correct) / n_laps)
        if debug_print:
            print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')

        return (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)


    (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)

    # laps_decoding_time_bin_size: 1.668
    # percent_laps_track_identity_estimated_correctly: 0.9875
    # percent_laps_direction_estimated_correctly: 0.5125
    # percent_laps_estimated_correctly: 0.5


    # laps_decoding_time_bin_size: 0.1
    # percent_laps_track_identity_estimated_correctly: 0.9875
    # percent_laps_direction_estimated_correctly: 0.4875
    # percent_laps_estimated_correctly: 0.4875

    # laps_decoding_time_bin_size: 0.5
    # percent_laps_track_identity_estimated_correctly: 1.0
    # percent_laps_direction_estimated_correctly: 0.5
    # percent_laps_estimated_correctly: 0.5














# ==================================================================================================================== #
# 2023-12-21 - Inversion Count Concept                                                                                 #
# ==================================================================================================================== #

class InversionCount:
    """ 2023-12-21 - "Inversion Count" Quantification of Order (as an alternative to Spearman?

        computes the number of swap operations required to sort the list `arr` 



    # Example usage

        from PendingNotebookCode import InversionCount
        # list1 = [3, 1, 5, 2, 4]
        list1 = [1, 2, 4, 3, 5] # 1
        list1 = [1, 3, 4, 5, 2] # 3
        num_swaps = count_swaps_to_sort(list1)
        print("Number of swaps required:", num_swaps)

        >>> Number of swaps required: 3



    """
    @classmethod
    def merge_sort_and_count(cls, arr):
        """ Inversion Count - computes the number of swap operations required to sort the list `arr` 
        """
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, count_left = cls.merge_sort_and_count(arr[:mid])
        right, count_right = cls.merge_sort_and_count(arr[mid:])
        merged, count_split = cls.merge_and_count(left, right)

        return merged, (count_left + count_right + count_split)

    @classmethod
    def merge_and_count(cls, left, right):
        """ Inversion Count """
        merged = []
        count = 0
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                count += len(left) - i
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, count

    @classmethod
    def count_swaps_to_sort(cls, arr):
        _, swaps = cls.merge_sort_and_count(arr)
        return swaps



class CurrTesting:
    
    # Pre-2023-21 ________________________________________________________________________________________________________ #

    def _plot_directional_likelihoods_df(directional_likelihoods_df):
        """ 2023-12-21 - Now 

        """
        df = deepcopy(directional_likelihoods_df)

        fig = plt.figure(num='directional_likelihoods_df Matplotlib figure')
        plt.plot(df.index, df["long_relative_direction_likelihoods"], label="Long Direction")
        plt.plot(df.index, df["short_relative_direction_likelihoods"], label="Short Direction")

        for i, idx in enumerate(df["long_best_direction_indices"]):
            if idx == 0:
                plt.annotate("↑", (df.index[i], df["long_relative_direction_likelihoods"][i]), textcoords="offset points", xytext=(0, 10))
            elif idx == 1:
                plt.annotate("↓", (df.index[i], df["long_relative_direction_likelihoods"][i]), textcoords="offset points", xytext=(0, -10))

        plt.xlabel("Index")
        plt.ylabel("Likelihood")
        plt.legend()
        plt.show()


    def pho_compute_rank_order(track_templates, curr_epoch_spikes_df: pd.DataFrame, rank_method="average", stats_nan_policy='omit') -> Dict[str, Tuple]:
        """ 2023-12-20 - Actually working spearman rank-ordering!! 

        Usage:
            curr_epoch_spikes_df = deepcopy(active_plotter.get_active_epoch_spikes_df())[['t_rel_seconds', 'aclu', 'shank', 'cluster', 'qclu', 'maze_id', 'flat_spike_idx', 'Probe_Epoch_id']]
            curr_epoch_spikes_df["spike_rank"] = curr_epoch_spikes_df["t_rel_seconds"].rank(method="average")
            # Sort by column: 'aclu' (ascending)
            curr_epoch_spikes_df = curr_epoch_spikes_df.sort_values(['aclu'])
            curr_epoch_spikes_df

        """
        curr_epoch_spikes_df["spike_rank"] = curr_epoch_spikes_df["t_rel_seconds"].rank(method=rank_method)
        # curr_epoch_spikes_df = curr_epoch_spikes_df.sort_values(['aclu'], inplace=False) # Sort by column: 'aclu' (ascending)

        n_spikes = np.shape(curr_epoch_spikes_df)[0]
        curr_epoch_spikes_aclus = deepcopy(curr_epoch_spikes_df.aclu.to_numpy())
        curr_epoch_spikes_aclu_ranks = deepcopy(curr_epoch_spikes_df.spike_rank.to_numpy())
        # curr_epoch_spikes_aclu_rank_map = dict(zip(curr_epoch_spikes_aclus, curr_epoch_spikes_aclu_ranks)) # could build a map equiv to template versions
        n_unique_aclus = np.shape(curr_epoch_spikes_df.aclu.unique())[0]
        assert n_spikes == n_unique_aclus, f"there is more than one spike in curr_epoch_spikes_df for an aclu! n_spikes: {n_spikes}, n_unique_aclus: {n_unique_aclus}"

        # decoder_LR_pf_peak_ranks_list = [scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_LR_decoder, short_LR_decoder)]
        # decoder_RL_pf_peak_ranks_list = [scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_RL_decoder, short_RL_decoder)]

        # rank_method: str = "dense"
        # rank_method: str = "average"

        track_templates.rank_method = rank_method
        # decoder_rank_dict = {a_decoder_name:scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder_name, a_decoder in track_templates.get_decoders_dict().items()}
        # decoder_aclu_peak_rank_dict_dict = {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method))) for a_decoder_name, a_decoder in track_templates.get_decoders_dict().items()}
        # decoder_aclu_peak_rank_dict_dict = {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense'))) for a_decoder_name, a_decoder in track_templates.decoder_peak_rank_list_dict.items()}

        # decoder_rank_dict = track_templates.decoder_peak_rank_list_dict
        decoder_aclu_peak_rank_dict_dict = track_templates.decoder_aclu_peak_rank_dict_dict

        template_spearman_real_results = {}
        for a_decoder_name, a_decoder_aclu_peak_rank_dict in decoder_aclu_peak_rank_dict_dict.items():
            # template_corresponding_aclu_rank_list: the list of template ranks for each aclu present in the `curr_epoch_spikes_aclus`
            template_corresponding_aclu_rank_list = np.array([a_decoder_aclu_peak_rank_dict.get(key, np.nan) for key in curr_epoch_spikes_aclus]) #  if key in decoder_aclu_peak_rank_dict_dict['long_LR']
            # curr_epoch_spikes_aclu_rank_list = np.array([curr_epoch_spikes_aclu_rank_map.get(key, np.nan) for key in curr_epoch_spikes_aclus])
            curr_epoch_spikes_aclu_rank_list = curr_epoch_spikes_aclu_ranks
            n_missing_aclus = np.isnan(template_corresponding_aclu_rank_list).sum()
            real_long_rank_stats = scipy.stats.spearmanr(curr_epoch_spikes_aclu_rank_list, template_corresponding_aclu_rank_list, nan_policy=stats_nan_policy)
            print(f'real_long_rank_stats: {real_long_rank_stats}')
            # _alt_real_long_rank_stats = CurrTesting.calculate_spearman_rank_correlation(curr_epoch_spikes_aclu_rank_list, template_corresponding_aclu_rank_list, rank_method=rank_method)
            # print(f'_alt_real_long_rank_stats: {_alt_real_long_rank_stats}')
            # print(f"Spearman rank correlation coefficient: {correlation}")
            template_spearman_real_results[a_decoder_name] = (*real_long_rank_stats, n_missing_aclus)
        
        return template_spearman_real_results


    def debug_detect_repeated_values(data, exceeding_count:int=1):
        """
        Identify and return a map of all repeated values in a list-like or NumPy array.

        Args:
            data: Any list-like or NumPy array.
            min_repeat: Max number of times a value can be used before it is included (default: 1).
            
        Returns:
            A dictionary mapping each repeated value to its count.
        """
        if isinstance(data, np.ndarray):
            data = data.flatten()
        return {key: value for key, value in Counter(data).items() if value > exceeding_count}




# ==================================================================================================================== #
# 2023-12-19 PyQtGraphCrosshairs                                                                                       #
# ==================================================================================================================== #

"""
Demonstrates some customized mouse interaction by drawing a crosshair that follows 
the mouse.
"""

from attrs import define, field
import pyphoplacecellanalysis.External.pyqtgraph as pg

@define(slots=False, repr=False)
class PyQtGraphCrosshairs:
    """ a class wrapper for the simple hover crosshairs shown in the pyqtgraph examples
    
    """
    vLine: pg.InfiniteLine = field()
    hLine: pg.InfiniteLine = field()
    proxy: pg.SignalProxy = field(init=False) 
    p1: pg.PlotItem = field(init=False)
    label: Optional[pg.LabelItem] = field(init=False)
        
    @classmethod
    def init_from_plot_item(cls, p1, a_label):
        _obj = cls(vLine=pg.InfiniteLine(angle=90, movable=False), 
             hLine=pg.InfiniteLine(angle=0, movable=False))
        _obj.p1 = p1
        _obj.label = a_label
        # _obj.vLine = pg.InfiniteLine(angle=90, movable=False)
        # _obj.hLine = pg.InfiniteLine(angle=0, movable=False)
        p1.addItem(_obj.vLine, ignoreBounds=True)
        p1.addItem(_obj.hLine, ignoreBounds=True)
        _obj.proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=_obj.mouseMoved)
        return _obj
  

    def mouseMoved(self, evt):
        """ captures `label` """
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        vb = self.p1.vb
        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            if index > 0 and index < len(data1):
                print(f"<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
                if self.label is not None:
                    self.label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())



# #generate layout
# app = pg.mkQApp("Crosshair Example")
# win = pg.GraphicsLayoutWidget(show=True)
# win.setWindowTitle('pyqtgraph example: crosshair')
# label = pg.LabelItem(justify='right')
# win.addItem(label)
# p1 = win.addPlot(row=1, col=0)
# # customize the averaged curve that can be activated from the context menu:
# p1.avgPen = pg.mkPen('#FFFFFF')
# p1.avgShadowPen = pg.mkPen('#8080DD', width=10)

# p2 = win.addPlot(row=2, col=0)

# region = pg.LinearRegionItem()
# region.setZValue(10)
# # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
# # item when doing auto-range calculations.
# p2.addItem(region, ignoreBounds=True)

# #pg.dbg()
# p1.setAutoVisible(y=True)

# #create numpy arrays
# #make the numbers large to show that the range shows data from 10000 to all the way 0
# data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
# data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)

# p1.plot(data1, pen="r")
# p1.plot(data2, pen="g")

# p2d = p2.plot(data1, pen="w")
# # bound the LinearRegionItem to the plotted data
# region.setClipItem(p2d)

# def update():
#     region.setZValue(10)
#     minX, maxX = region.getRegion()
#     p1.setXRange(minX, maxX, padding=0)    

# region.sigRegionChanged.connect(update)

# def updateRegion(window, viewRange):
#     rgn = viewRange[0]
#     region.setRegion(rgn)

# p1.sigRangeChanged.connect(updateRegion)

# region.setRegion([1000, 2000])

# #cross hair
# vLine = pg.InfiniteLine(angle=90, movable=False)
# hLine = pg.InfiniteLine(angle=0, movable=False)
# p1.addItem(vLine, ignoreBounds=True)
# p1.addItem(hLine, ignoreBounds=True)
# vb = p1.vb

# a_crosshairs = PyQtGraphCrosshairs.init_from_plot_item(p1=p1, a_label=label)

# #p1.scene().sigMouseMoved.connect(mouseMoved)







# ==================================================================================================================== #
# OLD                                                                                                                  #
# ==================================================================================================================== #



## Laps Stuff:

should_force_recompute_placefields = True
should_display_2D_plots = True
_debug_print = False


# ==================================================================================================================== #
# Programmatic Attr Class Generation with attr.ib                                                                      #
# ==================================================================================================================== #

import attr
import attrs

def create_class_from_dict(class_name, input_dict):
    """ 
    TempGraphicsOutput = create_class_from_dict('TempGraphicsOutput', _out)
    TempGraphicsOutput
    """
    attributes = {}
    for key, value in input_dict.items():
        attributes[key] = attr.ib(type=type(value), default=value) # , repr=False

    return attrs.make_class(class_name, attributes)




# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #

from copy import deepcopy
import numpy as np
from neuropy.utils.mixins.binning_helpers import transition_matrix


class TransitionMatrixComputations:
    """ 
    from PendingNotebookCode import TransitionMatrixComputations
    
    # Visualization ______________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    out = BasicBinnedImageRenderingWindow(binned_x_transition_matrix_higher_order_list[0], pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix', title="Transition Matrix for binned x (from, to)", variable_label='Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)
    
    
    """
    ### 1D Transition Matrix:

    def _compute_position_transition_matrix(xbin_labels, binned_x: np.ndarray, n_powers:int=3):
        """  1D Transition Matrix from binned positions (e.g. 'binned_x')

            pf1D.xbin_labels # array([  1,   2,   3,   4,  ...)
            pf1D.filtered_pos_df['binned_x'].to_numpy() # array([116, 115, 115, ...,  93,  93,  93], dtype=int64)
            
        Usage:
        
            # pf1D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf1D'])
            pf1D = deepcopy(global_pf1D)
            # pf1D = deepcopy(short_pf1D)
            # pf1D = deepcopy(long_pf1D)
            binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix(pf1D.xbin_labels, pf1D.filtered_pos_df['binned_x'].to_numpy())

        """
        num_position_states = len(xbin_labels)
        # binned_x = pos_1D.to_numpy()
        binned_x_indicies = binned_x - 1
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1, max_state_index=num_position_states)
        # binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, transition_matrix(deepcopy(binned_x_indicies), markov_order=2, max_state_index=num_position_states), transition_matrix(deepcopy(binned_x_indicies), markov_order=3, max_state_index=num_position_states)]

        binned_x_transition_matrix[np.isnan(binned_x_transition_matrix)] = 0.0
        binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [np.linalg.matrix_power(binned_x_transition_matrix, n) for n in np.arange(2, n_powers+1)]
        # , np.linalg.matrix_power(binned_x_transition_matrix, 2), np.linalg.matrix_power(binned_x_transition_matrix, 3)
        # binned_x_transition_matrix.shape # (64, 64)
        return binned_x_transition_matrix_higher_order_list

    def _build_decoded_positions_transition_matrix(active_one_step_decoder):
        """ Compute the transition_matrix from the decoded positions 

        TODO: make sure that separate events (e.g. separate replays) are not truncated creating erronious transitions

        """
        # active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
        # active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
        # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
        active_one_step_decoder.most_likely_position_flat_indicies
        # active_most_likely_positions = active_one_step_decoder.revised_most_likely_positions.T
        # active_most_likely_positions #.shape # (36246,)

        most_likely_position_indicies = np.squeeze(np.array(np.unravel_index(active_one_step_decoder.most_likely_position_flat_indicies, active_one_step_decoder.original_position_data_shape))) # convert back to an array
        most_likely_position_xbins = most_likely_position_indicies + 1 # add 1 to convert back to a bin label from an index
        # most_likely_position_indicies # (1, 36246)

        xbin_labels = np.arange(active_one_step_decoder.original_position_data_shape[0]) + 1

        decoded_binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix(xbin_labels, most_likely_position_indicies)
        return decoded_binned_x_transition_matrix_higher_order_list, xbin_labels

# ==================================================================================================================== #
# 2023-10-31 - Debug Plotting for Directional Placefield Templates                                                     #
# ==================================================================================================================== #

# from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import build_shared_sorted_neuronIDs
# from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph

# ratemap = long_pf1D.ratemap
# included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
# rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())

# heatmap_pf1D_win, heatmap_pf1D_img = visualize_heatmap_pyqtgraph(rediculous_final_sorted_all_included_pfmap, show_yticks=False, title=f"pf1D Sorted Visualization", defer_show=True)
# active_curves_sorted = long_pf1D.ratemap.normalized_tuning_curves[is_included][included_new_all_aclus_sort_indicies]
# heatmap_pf1D_win, heatmap_pf1D_img = visualize_heatmap_pyqtgraph(active_curves_sorted, show_yticks=False, title=f"pf1D Sorted Visualization", defer_show=True)

# _out = visualize_heatmap_pyqtgraph(np.vstack([odd_shuffle_helper.long_pf_peak_ranks, odd_shuffle_helper.short_pf_peak_ranks, even_shuffle_helper.long_pf_peak_ranks, even_shuffle_helper.short_pf_peak_ranks]), show_value_labels=True, show_xticks=True, show_yticks=True, show_colorbar=False)



from scipy import stats # _recover_samples_per_sec_from_laps_df

def _recover_samples_per_sec_from_laps_df(global_laps_df, time_start_column_name='start_t_rel_seconds', time_stop_column_name='end_t_rel_seconds',
            extra_indexed_column_start_column_name='start_position_index', extra_indexed_column_stop_column_name='end_position_index') -> float:
    """ Recovers the index/Denoting with λ(Θ) the probability that a neuron be active in a given room, the null hypothesis was therefore that all neurons could be assigned the same value λ (which would depend on Θ)time relation for the specified index columns by computing both the time duration and the number of indicies spanned by a given epoch.

    returns the `mode_samples_per_sec` corresponding to that column.

    ASSUMES REGULAR SAMPLEING!

    Usage:

        global_laps_df = global_laps.to_dataframe()
        position_mode_samples_per_sec = _recover_samples_per_sec_from_laps_df(global_laps_df, time_start_column_name='start_t_rel_seconds', time_stop_column_name='end_t_rel_seconds',
                    extra_indexed_column_start_column_name='start_position_index', extra_indexed_column_stop_column_name='end_position_index')

        position_mode_samples_per_sec # 29.956350269267112


     """
    duration_sec = global_laps_df[time_stop_column_name] - global_laps_df[time_start_column_name]
    num_position_samples = global_laps_df[extra_indexed_column_stop_column_name] - global_laps_df[extra_indexed_column_start_column_name]
    samples_per_sec = (num_position_samples/duration_sec).to_numpy()
    mode_samples_per_sec = stats.mode(samples_per_sec)[0] # take the mode of all the epochs
    return mode_samples_per_sec



# ==================================================================================================================== #
# 2023-10-26 - Directional Placefields to generate four templates                                                      #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers


# ==================================================================================================================== #
# 2023-10-19 Weighted Correlation                                                                                      #
# ==================================================================================================================== #
from neuropy.core import Epoch
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

@function_attributes(short_name=None, tags=['maze', 'maze_id', 'epochs', 'column'], input_requires=[], output_provides=[], uses=['Epoch'], used_by=[], creation_date='2023-10-19 08:01', related_items=[])
def _add_maze_id_to_epochs(active_filter_epochs: Epoch, track_change_time: float):
    """ adds a 'maze_id' columns to the `active_filter_epochs`'s internal dataframe.
    
     Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:

        Usage:
     
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        track_change_time = short_session.t_start
        active_filter_epochs: Epoch = long_results_obj.active_filter_epochs ## no copy
        active_filter_epochs = _add_maze_id_to_epochs(active_filter_epochs, track_change_time)
        active_filter_epochs

    """
    active_filter_epochs._df['maze_id'] = np.nan
    long_active_filter_epochs = active_filter_epochs.time_slice(None, track_change_time)
    active_filter_epochs._df.loc[np.isin(active_filter_epochs.labels, long_active_filter_epochs.labels), 'maze_id'] = 0

    short_active_filter_epochs = active_filter_epochs.time_slice(track_change_time, None)
    active_filter_epochs._df.loc[np.isin(active_filter_epochs.labels, short_active_filter_epochs.labels), 'maze_id'] = 1
    active_filter_epochs._df = active_filter_epochs._df.astype({'maze_id': 'int8'}) # Change column type to int8 for column: 'maze_id'

    return active_filter_epochs


@function_attributes(short_name=None, tags=['weighted_correlation', 'decoder', 'epoch'], input_requires=[], output_provides=[], uses=['WeightedCorr'], used_by=['add_weighted_correlation_result'], creation_date='2023-10-19 07:54', related_items=[])
def compute_epoch_weighted_correlation(xbin_centers, curr_time_bins, curr_long_epoch_p_x_given_n, method='spearman') -> List[float]:
    """ computes the weighted_correlation for the epoch given the decoded posterior

    # FLATTEN for WeightedCorr calculation, filling as appropriate:
    X, Y are vectors
    W is a matrix containing the posteriors
    
    """
    from neuropy.utils.external.WeightedCorr import WeightedCorr
    
    n_xbins = len(xbin_centers)
    curr_n_time_bins = len(curr_time_bins)
    curr_flat_length = int(float(curr_n_time_bins) * float(n_xbins))
    
    X = np.repeat(curr_time_bins, n_xbins)
    Y = np.tile(xbin_centers, curr_n_time_bins)
    assert np.shape(X) == np.shape(Y)
    W = np.reshape(curr_long_epoch_p_x_given_n, newshape=curr_flat_length, order='F') # order='F' means take the first axis (xbins) as changing the fastest
    assert np.allclose(curr_long_epoch_p_x_given_n[:,1], W[n_xbins:n_xbins+n_xbins]) # compare the lienarlly-index second timestamp with the 2D-indexed version to ensure flattening was done correctly.
    data_df = pd.DataFrame({'x':X, 'y':Y, 'w':W})
    
    weighted_corr_results = []

    if isinstance(method, str):
        # wrap in single element list:
        method = (method, )
        
    a_weighted_corr_obj = WeightedCorr(xyw=data_df[['x', 'y', 'w']])
    for a_method in method:
        weighted_corr_results.append(a_weighted_corr_obj(method=a_method)) # append the scalar

    return weighted_corr_results



@function_attributes(short_name=None, tags=['weighted_correlation', 'decoder'], input_requires=[], output_provides=[], uses=['compute_epoch_weighted_correlation'], used_by=[], creation_date='2023-10-19 07:54', related_items=[])
def add_weighted_correlation_result(xbin_centers, a_long_decoder_result: DecodedFilterEpochsResult, a_short_decoder_result: DecodedFilterEpochsResult, method=('pearson', 'spearman'), debug_print = False):
    """ builds the weighted correlation for each epoch respective to the posteriors decoded by each decoder (long/short) """
    epoch_long_weighted_corr_results = []
    epoch_short_weighted_corr_results = []

    for decoded_epoch_idx in np.arange(a_long_decoder_result.num_filter_epochs):
        # decoded_epoch_idx:int = 0
        curr_epoch_time_bin_container = a_long_decoder_result.time_bin_containers[decoded_epoch_idx]
        curr_time_bins = curr_epoch_time_bin_container.centers
        curr_n_time_bins = len(curr_time_bins)
        if debug_print:
            print(f'curr_n_time_bins: {curr_n_time_bins}')


        ## Long Decoding:
        curr_long_epoch_p_x_given_n = a_long_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
        print(f'np.shape(curr_long_epoch_p_x_given_n): {np.shape(curr_long_epoch_p_x_given_n)}')
        
        weighted_corr_result = compute_epoch_weighted_correlation(xbin_centers, curr_time_bins, curr_long_epoch_p_x_given_n, method=method)
        epoch_long_weighted_corr_results.append(weighted_corr_result)

        ## Short Decoding:
        curr_short_epoch_p_x_given_n = a_short_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
        weighted_corr_result = compute_epoch_weighted_correlation(xbin_centers, curr_time_bins, curr_short_epoch_p_x_given_n, method=method)
        epoch_short_weighted_corr_results.append(weighted_corr_result)

    # ## Build separate result dataframe:
    # epoch_weighted_corr_results_df = pd.DataFrame({'weighted_corr_LONG': np.array(epoch_long_weighted_corr_results), 'weighted_corr_SHORT': np.array(epoch_short_weighted_corr_results)})
    # epoch_weighted_corr_results_df

    return np.array(epoch_long_weighted_corr_results), np.array(epoch_short_weighted_corr_results)
    
    




# ==================================================================================================================== #
# 2023-10-11                                                                                                           #
# ==================================================================================================================== #
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables

def build_and_merge_all_sessions_joined_neruon_fri_df(global_data_root_parent_path, BATCH_DATE_TO_USE):
    """ captures a lot of stuff still, don't remember what. 
    
    Usage:    
        # BATCH_DATE_TO_USE = '2023-10-05_NewParameters'
        BATCH_DATE_TO_USE = '2023-10-07'
        all_sessions_joined_neruon_fri_df, out_path = build_and_merge_all_sessions_joined_neruon_fri_df(global_data_root_parent_path, BATCH_DATE_TO_USE)


    TODO: seems like it should probably go into AcrossSessionResults or AcrossSessionTables
    
    """


    # Rootfolder mode:
    # joined_neruon_fri_df_file_paths = [global_data_root_parent_path.joinpath(f'{BATCH_DATE_TO_USE}_{a_ctxt.get_description(separator="-", include_property_names=False)}_joined_neruon_fri_df.pkl') for a_ctxt in included_session_contexts]

    # Subfolder mode:
    # joined_neruon_fri_df_file_paths = [global_data_root_parent_path.joinpath(BATCH_DATE_TO_USE, f'{a_ctxt.get_description(separator="-", include_property_names=False)}_joined_neruon_fri_df.pkl') for a_ctxt in included_session_contexts]

    # Both mode:
    joined_neruon_fri_df_file_paths = [global_data_root_parent_path.joinpath(BATCH_DATE_TO_USE, f'{BATCH_DATE_TO_USE}_{a_ctxt.get_description(separator="-", include_property_names=False)}_joined_neruon_fri_df.pkl') for a_ctxt in included_session_contexts]
    joined_neruon_fri_df_file_paths = [a_path for a_path in joined_neruon_fri_df_file_paths if a_path.exists()] # only get the paths that exist

    data_frames = [AcrossSessionTables.load_table_from_file(global_data_root_parent_path=a_path.parent, output_filename=a_path.name) for a_path in joined_neruon_fri_df_file_paths]
    # data_frames = [df for df in data_frames if df is not None] # remove empty results
    print(f'joined_neruon_fri_df: concatenating dataframes from {len(data_frames)}')
    all_sessions_joined_neruon_fri_df = pd.concat(data_frames, ignore_index=True)

    ## Finally save out the combined result:
    all_sessions_joined_neruon_fri_df_basename = f'{BATCH_DATE_TO_USE}_MERGED_joined_neruon_fri_df'
    out_path = global_data_root_parent_path.joinpath(all_sessions_joined_neruon_fri_df_basename).resolve()
    AcrossSessionTables.write_table_to_files(all_sessions_joined_neruon_fri_df, global_data_root_parent_path=global_data_root_parent_path, output_basename=all_sessions_joined_neruon_fri_df_basename)
    print(f'>>\t done with {out_path}')
    return all_sessions_joined_neruon_fri_df, out_path







#TODO 2023-08-10 16:50: - [ ] 



from enum import Enum, auto
from attrs import define

@define(slots=False)
class SwiftLikeEnum:
    """ # can enums store associated data?
    # some properties only make sense for certain enum values, like .
    """
    value: int
    attribute: str

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

class MyEnum(Enum):
    CASE1 = SwiftLikeEnum(value=auto(), attribute="attribute1")
    CASE2 = SwiftLikeEnum(value=auto(), attribute="attribute2")

    @property
    def attribute(self):
        return self.value.attribute




# 2023-07-13 - Helpers for future swapping of the x and y axis on many of the plots like Kamran suggested.

def _swap_x_and_y_axis(x_frs, y_frs, should_swap_axis:bool=True):
    """ swaps the order of the arguments depending on the value of `should_swap_axis`. Can be used to reverse the x and y axes for a plot."""
    if not should_swap_axis:
        return (x_frs, y_frs) # return in same order
    else:
        return (y_frs, x_frs)

# 2023-07-05 - MiracleWrapper Idea
"""
# Wish it was easier to get things in and out of functions.
def a_fn(...):
    pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = a_computation_fn(...)
    miracle_wrapper = MiracleWrapper(pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics)
    miracle_wrapper.add(pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics)
    return miracle_wrapper

miracle_wrapper = a_fn(...)
# Ideally, you could get them out "by magic" by specifying the same name that they were put in with on the LHS of an assignment eqn:
pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = miracle_wrapper.magic_unwrap()
"""


# ==================================================================================================================== #
# 2023-05-16 - Manual Post-hoc Conformance for Laps and Long/Short Bins                                                #
# ==================================================================================================================== #

def _update_computation_configs_with_laps_and_shared_grid_bins(curr_active_pipeline, enable_interactive_bounds_selection:bool = False):
    """ 2023-05-16 - A post-hoc version of updating the computation configs and recomputing with the laps as the computation_epochs and the shared bins as the grid_bin_bounds.
            In the future shouldn't need this, as I updated the KDiba default active computation configs to determine these properties prior to computation by default.
    """
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    # curr_active_pipeline.computation_results['maze1'].computation_config.pf_params.grid_bin = refined_grid_bin_bounds

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]


    active_computation_configs_dict = {'default': curr_active_pipeline.computation_results[global_epoch_name].computation_config} # get the old pf_params from global

    ## Duplicate the default computation config to modify it:
    temp_comp_params = deepcopy(active_computation_configs_dict['default'])

    # Determine the grid_bin_bounds from the long session:
    grid_bin_bounding_session = long_session
    grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(grid_bin_bounding_session.position.x, grid_bin_bounding_session.position.y)
    grid_bin_bounds # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))

    if enable_interactive_bounds_selection:
        # Interactive grid_bin_bounds selector (optional):
        from neuropy.utils.matplotlib_helpers import add_rectangular_selector
        # Show an interactive rectangular selection for the occupancy:
        fig, ax = curr_active_pipeline.computation_results['maze'].computed_data.pf2D.plot_occupancy()
        rect_selector, set_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds) # (24.82, 257.88), (125.52, 149.19)
        # TODO: allow the user to customize selection (block) before continuing
        # refined_grid_bin_bounds
        # final_grid_bin_bounds = refined_grid_bin_bounds # TODO 2023-05-16 - implement
    else:
        # no interactive selection/refinement:
        final_grid_bin_bounds = grid_bin_bounds

    # refined_grid_bin_bounds = ((24.12, 259.80), (130.00, 150.09))
    # temp_comp_params = PlacefieldComputationParameters(speed_thresh=4)
    # temp_comp_params.pf_params.speed_thresh = 10 # 4.0 cm/sec
    # temp_comp_params.pf_params.grid_bin = (2, 2) # (2cm x 2cm)
    temp_comp_params.pf_params.grid_bin = (1.5, 1.5) # (1.5cm x 1.5cm)
    temp_comp_params.pf_params.grid_bin_bounds = final_grid_bin_bounds # same bounds for all
    # temp_comp_params.pf_params.smooth = (0.0, 0.0) # No smoothing
    # temp_comp_params.pf_params.frate_thresh = 1 # Minimum for non-smoothed peak is 1Hz
    temp_comp_params.pf_params.computation_epochs = global_session.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0) # laps specifically for use in the placefields with non-overlapping, duration, constraints: the lap must be at least 1 second long and at most 30 seconds long

    # Add it to the array of computation configs:
    # active_session_computation_configs.append(temp_comp_params)
    active_computation_configs_dict['custom'] = temp_comp_params
    # active_computation_configs_dict


    # Compute with the new computation config:
    computation_functions_name_includelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                        '_perform_position_decoding_computation', 
                                        '_perform_firing_rate_trends_computation',
                                        '_perform_pf_find_ratemap_peaks_computation',
                                        '_perform_time_dependent_pf_sequential_surprise_computation'
                                        '_perform_two_step_position_decoding_computation',
                                        # '_perform_recursive_latent_placefield_decoding'
                                    ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'

    # computation_functions_name_includelist=['_perform_baseline_placefield_computation']
    curr_active_pipeline.perform_computations(computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=None, fail_on_exception=True, debug_print=False, overwrite_extant_results=True) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)
    return curr_active_pipeline



# ==================================================================================================================== #
# 2023-05-08 - Paginated Plots                                                                                         #
# ==================================================================================================================== #

# from PendingNotebookCode import PaginationController
# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
# from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# From plot_paginated_decoded_epoch_slices

# ==================================================================================================================== #
# 2023-05-02 - Factor out Paginator and plotting stuff                                                                 #
# ==================================================================================================================== #

# from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D

# ==================================================================================================================== #
# 2023-05-02 - Factor out interactive matplotlib/pyqtgraph helper code (untested)                                      #
# ==================================================================================================================== #
import matplotlib
from attrs import define, Factory

@define(slots=True, eq=False) #eq=False enables hashing by object identity
class SelectionManager:
    """ Takes a list of matplotlib Axes that can have their selection toggled/un-toggled for inclusion/exclusion. 
        Adds the ability to toggle selections for each axis by clicking, and a grey background for selected objects vs. white for unselected.
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

        laps_plot_tuple = plot_decoded_epoch_slices(long_results_obj.active_filter_epochs, long_results_obj.all_included_filter_epochs_decoder_result, global_pos_df=global_session.position.df, variable_name='lin_pos', xbin=long_results_obj.original_1D_decoder.xbin,
                                                                name='stacked_epoch_slices_long_results_obj', debug_print=False, debug_test_max_num_slices=32)
        curr_viz_params, _curr_plot_data, _curr_plots, _curr_ui_container = laps_plot_tuple

        # Create a SelectionManager instance
        sm = SelectionManager(_curr_plots.axs)

    """
    axes: list
    is_selected: dict = Factory(dict)
    fig: matplotlib.figure.Figure = None # Matplotlib.Figure
    cid: int = None # Matplotlib.Figure

    def __attrs_post_init__(self):
        # Get figure from first axes:
        assert len(self.axes) > 0
        first_ax = self.axes[0]
        self.fig = first_ax.get_figure()		
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # Set initial selection to False
        for ax in self.axes:
            self.is_selected[ax] = False

    def on_click(self, event):
        # Get the clicked Axes object
        ax = event.inaxes		
        # Toggle the selection status of the clicked Axes
        self.is_selected[ax] = not self.is_selected[ax]
        # Set the face color of the clicked Axes based on its selection status
        if self.is_selected[ax]:
            ax.patch.set_facecolor('gray')
        else:
            ax.patch.set_facecolor('white')
        # Redraw the figure to show the updated selection
        event.canvas.draw()


from attrs import Factory

@define(slots=True, eq=False) #eq=False enables hashing by object identity
class PaginatedSelectionManager:
    """ Takes a list of matplotlib Axes that can have their selection toggled/un-toggled for inclusion/exclusion. 
        Adds the ability to toggle selections for each axis by clicking, and a grey background for selected objects vs. white for unselected.
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

        laps_plot_tuple = plot_decoded_epoch_slices(long_results_obj.active_filter_epochs, long_results_obj.all_included_filter_epochs_decoder_result, global_pos_df=global_session.position.df, variable_name='lin_pos', xbin=long_results_obj.original_1D_decoder.xbin,
                                                                name='stacked_epoch_slices_long_results_obj', debug_print=False, debug_test_max_num_slices=32)
        curr_viz_params, _curr_plot_data, _curr_plots, _curr_ui_container = laps_plot_tuple

        # Create a SelectionManager instance
        sel_man = PaginatedSelectionManager(axes=self.plots.axs, fig=self.plots.fig)

        def on_page_change(updated_page_idx):
            # print(f'on_page_change(updated_page_idx: {updated_page_idx})')
            # Update on page change:
            sel_man.perform_update()

        ui.mw.ui.paginator_controller_widget.jump_to_page.connect(on_page_change)

    """
    axes: list
    is_selected: dict = Factory(dict)
    fig: matplotlib.figure.Figure = None # Matplotlib.Figure
    callback_id: int = None # Matplotlib.Figure

    def __attrs_post_init__(self):
        # Get figure from first axes:
        assert len(self.axes) > 0
        first_ax = self.axes[0]
        self.fig = first_ax.get_figure()		
        self.callback_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # Set initial selection to False
        # for ax in self.axes:
        #     self.is_selected[ax] = False

    def perform_update(self):
        """ called to update the selection when the page is changed or something else happens. """        
        current_page_idx, curr_page_data_indicies = _get_current_page_data_indicies()
        assert len(self.plots.axs) == len(curr_page_data_indicies), f"len(plots.axs): {len(self.plots.axs)}, len(curr_page_data_indicies): {len(curr_page_data_indicies)}"
        for ax, found_data_idx in zip(self.plots.axs, list(curr_page_data_indicies)): # TODO: might fail for the last page?
            # print(f'found_data_idx: {found_data_idx}')
            # found_data_index = curr_page_data_indicies[found_index]
            # print(f'{current_page_idx = }, {found_data_index =}')
            is_selected = self.is_selected.get(found_data_idx, False)
            if is_selected:
                ax.patch.set_facecolor('gray')
            else:
                ax.patch.set_facecolor('white')
                
        # Redraw the figure to show the updated selection
        self.fig.canvas.draw()

    def on_click(self, event):
        # Get the clicked Axes object
        ax = event.inaxes
        # Find the axes
        found_index = safe_find_index_in_list(self.plots.axs, ax)
        # print(f'{found_index = }')
        current_page_idx, curr_page_data_indicies = _get_current_page_data_indicies()
        found_data_index = curr_page_data_indicies[found_index]
        # print(f'{current_page_idx = }, {found_data_index =}')
        # Toggle the selection status of the clicked Axes
        self.is_selected[found_data_index] = not self.is_selected.get(found_data_index, False) # if never set before, assume that it's not selected

        # self.is_selected[ax] = not self.is_selected[ax]
        # Set the face color of the clicked Axes based on its selection status
        # if self.is_selected[ax]:
        if self.is_selected[found_data_index]:
            ax.patch.set_facecolor('gray')
        else:
            ax.patch.set_facecolor('white')
        # Redraw the figure to show the updated selection
        event.canvas.draw()
        

# ==================================================================================================================== #
# 2023-04-17 - Factor out interactive diagnostic figure code                                                           #
# ==================================================================================================================== #
## Create a diagnostic plot that plots a stack of the three curves used for computations in the given epoch:

import pyphoplacecellanalysis.External.pyqtgraph as pg



# ==================================================================================================================== #
# 2023-04-10 - Long short expected surprise                                                                            #
# ==================================================================================================================== #

def _scramble_curve(pf: np.ndarray, roll_num_bins:int = 10, method='circ'):
    """ Circularly rotates the 1D placefield """
    return np.roll(pf, roll_num_bins)




# ==================================================================================================================== #
# 2023-03-09 - Parameter Sweeping                                                                                      #
# ==================================================================================================================== #

def _compute_parameter_sweep(spikes_df, active_pos, all_param_sweep_options: dict) -> dict:
    """ Computes the PfNDs for all the swept parameters (combinations of grid_bin, smooth, etc)
    
    Usage:
        from PendingNotebookCode import _compute_parameter_sweep

        smooth_options = [(None, None), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
        grid_bin_options = [(1,1),(5,5),(10,10)]
        all_param_sweep_options = cartesian_product(smooth_options, grid_bin_options)
        param_sweep_option_n_values = dict(smooth=len(smooth_options), grid_bin=len(grid_bin_options)) 
        output_pfs = _compute_parameter_sweep(spikes_df, active_pos, all_param_sweep_options)

    """
    output_pfs = {} # empty dict

    for a_sweep_dict in all_param_sweep_options:
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        output_pfs[a_sweep_tuple] = PfND(deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal'), deepcopy(active_pos.linear_pos_obj), **a_sweep_dict) # grid_bin=, etc
        
    return output_pfs




# ==================================================================================================================== #
# 2022-02-17 - Giving up on Rank-Order Sequence Analysis                                                               #
# ==================================================================================================================== #
""" 
    after convincing Kamran that the sample size of the diferent replays made them uncomparable.
"""

# # 2023-02-16 - Simple "weighted-center-of-mass" method of determing cell firing order in a timeseries

# +

class SpikesRankOrder:

    def compute_rankordered_spikes_during_epochs(active_spikes_df, active_epochs):
        """ 
        Usage:
            from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

            active_sess = curr_active_pipeline.filtered_sessions['maze']
            active_epochs = active_sess.perform_compute_estimated_replay_epochs(min_epoch_included_duration=None, max_epoch_included_duration=None, maximum_speed_thresh=None) # filter on nothing basically
            active_spikes_df = active_sess.spikes_df.spikes.sliced_by_neuron_type('pyr') # only look at pyramidal cells

            spike_trimmed_active_epochs, _extra_outputs = filter_epochs_by_num_active_units(active_spikes_df, active_epochs, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=1)
            epoch_ranked_aclus_dict, active_spikes_df, all_probe_epoch_ids, all_aclus = compute_rankordered_spikes_during_epochs(active_spikes_df, active_epochs)
    """
        from neuropy.utils.mixins.time_slicing import add_epochs_id_identity
        
        # add the active_epoch's id to each spike in active_spikes_df to make filtering and grouping easier and more efficient:
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=-1) # uses new add_epochs_id_identity

        # Get all aclus and epoch_idxs used throughout the entire spikes_df:
        all_aclus = active_spikes_df['aclu'].unique()
        all_probe_epoch_ids = active_spikes_df['Probe_Epoch_id'].unique()

        selected_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name].first() # first spikes
        # selected_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name].median() # median spikes
        
        
        # rank the aclu values by their first t value in each Probe_Epoch_id
        ranked_aclus = selected_spikes.groupby('Probe_Epoch_id').rank(method='dense') # resolve ties in ranking by assigning the same rank to each and then incrimenting for the next item
        # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        ranked_aclus_dict = {}
        for (epoch_id, aclu), rank in zip(ranked_aclus.index, ranked_aclus):
            if epoch_id not in ranked_aclus_dict:
                ranked_aclus_dict[epoch_id] = {}
            ranked_aclus_dict[epoch_id][aclu] = rank
        # ranked_aclus_dict
        return ranked_aclus_dict, active_spikes_df, all_probe_epoch_ids, all_aclus

    # -


    # +

    def compute_rankordered_stats(epoch_ranked_aclus_dict):
        """ Spearman rank-order tests:
            
        WARNING, from documentation: Although calculation of the p-value does not make strong assumptions about the distributions underlying the samples, it is only accurate for very large samples (>500 observations). For smaller sample sizes, consider a permutation test (see Examples section below).

        Usage: 
            
            epoch_ranked_aclus_stats_corr_values, epoch_ranked_aclus_stats_p_values, (outside_epochs_ranked_aclus_stats_corr_value, outside_epochs_ranked_aclus_stats_p_value) = compute_rankordered_stats(epoch_ranked_aclus_dict)

        """
        import scipy.stats
        
        epoch_ranked_aclus_stats_dict = {epoch_id:scipy.stats.spearmanr(np.array(list(rank_dict.keys())), np.array(list(rank_dict.values()))) for epoch_id, rank_dict in epoch_ranked_aclus_dict.items()}
        # epoch_ranked_aclus_stats_dict

        # Spearman statistic (correlation) values:
        epoch_ranked_aclus_stats_corr_values = np.array([np.abs(rank_stats.statistic) for epoch_id, rank_stats in epoch_ranked_aclus_stats_dict.items()])
        outside_epochs_ranked_aclus_stats_corr_value = epoch_ranked_aclus_stats_corr_values[0]
        epoch_ranked_aclus_stats_corr_values = epoch_ranked_aclus_stats_corr_values[1:] # drop the first value corresponding to the -1 index. Now they correspond only to valid epoch_ids

        # Spearman p-values:
        epoch_ranked_aclus_stats_p_values = np.array([rank_stats.pvalue for epoch_id, rank_stats in epoch_ranked_aclus_stats_dict.items()])
        outside_epochs_ranked_aclus_stats_p_value = epoch_ranked_aclus_stats_p_values[0]
        epoch_ranked_aclus_stats_p_values = epoch_ranked_aclus_stats_p_values[1:] # drop the first value corresponding to the -1 index. Now they correspond only to valid epoch_ids

        return epoch_ranked_aclus_stats_corr_values, epoch_ranked_aclus_stats_p_values, (outside_epochs_ranked_aclus_stats_corr_value, outside_epochs_ranked_aclus_stats_p_value)


    # # + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
    # # ### 2023-02-16 TODO: try to overcome issue with small sample sizes mentioned above by performing the permutation test:

    # # +
    # # def statistic(x):  # permute only `x`
    # #     return scipy.stats.spearmanr(x, y).statistic
    # # res_exact = scipy.stats.permutation_test((x,), statistic, permutation_type='pairings')
    # res_asymptotic = scipy.stats.spearmanr(x, y)
    # res_exact.pvalue, res_asymptotic.pvalue  # asymptotic pvalue is too low

    # # scipy.stats.permutation_test((x,), (lambda x: scipy.stats.spearmanr(x, y).statistic), permutation_type='pairings')

    # ## Compute the exact value using permutations:
    # # epoch_ranked_aclus_stats_exact_dict = {epoch_id:scipy.stats.permutation_test((np.array(list(rank_dict.keys())),), (lambda x: scipy.stats.spearmanr(x, np.array(list(rank_dict.values()))).statistic), permutation_type='pairings') for epoch_id, rank_dict in epoch_ranked_aclus_dict.items()}
    # epoch_ranked_aclus_stats_exact_dict = {epoch_id:scipy.stats.permutation_test((np.array(list(rank_dict.values())),), (lambda y: scipy.stats.spearmanr(np.array(list(rank_dict.keys())), y).statistic), permutation_type='pairings') for epoch_id, rank_dict in epoch_ranked_aclus_dict.items()} # ValueError: each sample in `data` must contain two or more observations along `axis`.
    # epoch_ranked_aclus_stats_exact_dict







# ==================================================================================================================== #
# 2022-12-22 - Posterior Confidences/Certainties                                                                       #
# ==================================================================================================================== #

def _compute_epoch_posterior_confidences(X_decoding_of_Y_epochs_results):
    """ average over positions to find the maximum likelihood in the posterior (value only) for each timebin. This is a rough estimate for how certain we are about each timebin.
    Usage:
        from PendingNotebookCode import _compute_epoch_posterior_confidences
        long_decoding_of_short_epochs_results = _compute_epoch_posterior_confidences(long_decoding_of_short_epochs_results)
        short_decoding_of_long_epochs_results = _compute_epoch_posterior_confidences(short_decoding_of_long_epochs_results)
    """
    # loop through each returned epoch and compute its measurez:
    X_decoding_of_Y_epochs_results.posterior_uncertainty_measure = [] # one for each decoded epoch
    ## combined_plottables variables refer to concatenating the values for each epoch so they can be plotted using a single matplotlib command:
    X_decoding_of_Y_epochs_results.combined_plottables_x = []
    X_decoding_of_Y_epochs_results.combined_plottables_y = []
    
    for i, time_bin_container, p_x_given_n in zip(np.arange(X_decoding_of_Y_epochs_results.num_filter_epochs), X_decoding_of_Y_epochs_results.time_bin_containers, X_decoding_of_Y_epochs_results.p_x_given_n_list):
        # average over positions to find the maximum likelihood in the posterior (value only) for each timebin. This is a rough estimate for how certain we are about each timebin.
        posterior_uncertainty_measure = np.max(p_x_given_n, axis=0) # each value will be between (0.0, 1.0]
        X_decoding_of_Y_epochs_results.posterior_uncertainty_measure.append(posterior_uncertainty_measure)
        X_decoding_of_Y_epochs_results.combined_plottables_x.append(time_bin_container.centers)
        X_decoding_of_Y_epochs_results.combined_plottables_y.append(posterior_uncertainty_measure)
    return X_decoding_of_Y_epochs_results



# ==================================================================================================================== #
# 2022-12-20 - Overlapping Intervals                                                                                   #
# ==================================================================================================================== #
# https://www.baeldung.com/cs/finding-all-overlapping-intervals

# def eraseOverlapIntervals(intervals):
#     """ https://leetcode.com/problems/non-overlapping-intervals/solutions/91702/python-simple-greedy-10-lines/ """
#     if len(intervals) == 0:
#         return 0
#     intervals = sorted(intervals, key = lambda x:x[1])
#     removeNum, curBorder = -1, intervals[0][1]
#     for interval in intervals:
#         if interval[0] < curBorder:
#             removeNum += 1
#         else:
#             curBorder = interval[1]
#     return removeNum

def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    """ https://leetcode.com/problems/non-overlapping-intervals/solutions/424634/python-greedy-w-optimizations-faster-then-99-5/ """
    max_count = len(intervals)
    if max_count  <= 1:
        return 0
    arr = sorted(intervals, key=lambda x: x[1])
    counter = 0
    last_end = float("-inf")
    for elem in arr:
        if elem[0] >= last_end:
            last_end = elem[1]
            counter += 1
    return max_count - counter


def removeCoveredIntervals(intervals: List[List[int]]) -> List[List[int]]:
    """ 
    https://leetcode.com/problems/remove-covered-intervals/solutions/879665/python-faster-than-99-using-dict/

    Alternatives:
        https://leetcode.com/problems/remove-covered-intervals/solutions/878478/python-simple-solution-explained-video-code-fastest/
        https://leetcode.com/problems/remove-covered-intervals/solutions/1784520/python3-sorting-explained/?orderBy=most_votes&languageTags=python3

    """
    d=dict()   
    high=-1
    ans=0
    out_intervals = []
    totalIntervals = len(intervals)
    
    for i in range(len(intervals)):
        if intervals[i][0] in d.keys():
            if d[intervals[i][0]]< intervals[i][1]:
                d[intervals[i][0]] = intervals[i][1]
        else:
            d[intervals[i][0]]  = intervals[i][1]
    for i in sorted(d):
        if d[i] > high:
            high = d[i]
            ans+=1
            out_intervals.append(d[i])
    return out_intervals


def merge(intervals: List[List[int]], in_place=False) -> List[List[int]]:
    """ NOTE: modifies initial array.
    Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
     https://leetcode.com/problems/merge-intervals/solutions/350272/python3-sort-o-nlog-n/ """
    if not in_place:
        # return a copy so the original intervals aren't modified.
        intervals = deepcopy(intervals)
    intervals.sort(key =lambda x: x[0])
    merged =[]
    for i in intervals:
        # if the list of merged intervals is empty 
        # or if the current interval does not overlap with the previous,
        # simply append it.
        if not merged or merged[-1][-1] < i[0]:
            merged.append(i)
        # otherwise, there is overlap,
        #so we merge the current and previous intervals.
        else:
            merged[-1][-1] = max(merged[-1][-1], i[-1])
    return merged


# Divide Intervals Into Minimum Number of Groups _____________________________________________________________________ #
# """You are given a 2D integer array intervals where intervals[i] = [lefti, righti] represents the inclusive interval [lefti, righti].
#     You have to divide the intervals into one or more groups such that each interval is in exactly one group, and no two intervals that are in the same group intersect each other.
#     Return the minimum number of groups you need to make.
# """

import heapq
def minGroups(intervals: List[List[int]]) -> int:
    """ https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/solutions/2560020/min-heap/?orderBy=most_votes&languageTags=python3

    Alternatives:
        https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/solutions/2568422/python-runtime-o-nlogn-96-12-memory-o-n/
    """
    pq = []
    for left, right in sorted(intervals):
        if pq and pq[0] < left:
            heapq.heappop(pq)
        heapq.heappush(pq, right)
    return pq


## overlap between ranges:
def overlap(a,b,c,d):
    """ 
    return 0 for identical intervals and None for non-overlapping intervals as required.
    https://stackoverflow.com/questions/11026167/interval-overlap-size

    For ranges, see: https://stackoverflow.com/questions/6821156/how-to-find-range-overlap-in-python 
        `range(max(x[0], y[0]), min(x[-1], y[-1])+1)`
    """
    r = 0 if a==c and b==d else min(b,d)-max(a,c)
    if r>=0: return r


# def merge(self, intervals: List[List[int]]) -> List[List[int]]:
#     """ https://dev.to/codekagei/algorithm-to-merge-overlapping-intervals-found-in-a-list-python-solution-5819 """
#     if len(intervals) <= 1:
#         return intervals
#     output = []
#     intervals.sort()
#     current = intervals[0]
#     output.append(current)
#     for i in range(len(intervals)):
#         current2 = current[1];
#         next1 = intervals[i][0]
#         next2 = intervals[i][1]

#         if current2 >= next1:
#             current[1] = max(current2, next2)
#         else:
#             current = intervals[i]
#             output.append(current)

#     return output


# ==================================================================================================================== #
# 2022-12-18 - Added Standardization of Position bins between short and long                                           #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND # for re-binning pf1D
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results

def merge_overlapping_intervals(intervals):
    """ Doesn't seem to work. Generated by Chat-GPT """
    # Sort the intervals by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    # Initialize the result with the first interval
    result = [intervals[0]]
    # Iterate through the rest of the intervals
    for interval in intervals[1:]:
        # If the current interval overlaps with the last interval in the result,
        # update the end time of the last interval to the maximum of the two end times
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        # Otherwise, append the current interval to the result
        else:
            result.append(interval)
    return np.array(result)

def split_overlapping_intervals(intervals):
    """ Doesn't seem to work. Generated by Chat-GPT """
    # Sort the intervals by start time
    intervals = sorted(intervals, key=lambda x: x[0])

    result = []
    # Iterate through the intervals
    for interval in intervals:
        # If the current interval overlaps with the last interval in the result,
        # split the current interval into two non-overlapping intervals
        if result and interval[0] <= result[-1][1]:
            result.append([result[-1][1], interval[1]])
        # Otherwise, append the current interval to the result
        else:
            result.append(interval)
    return np.array(result)


### Piso-based interval overlap removal
# ## Build non-overlapping intervals with piso. Unsure of the computation efficiency, but the ouptuts are correct.
# import piso
# piso.register_accessors()

# print(f'pre: {active_filter_epochs.shape[0]}')
# valid_intervals = pd.arrays.IntervalArray.from_arrays(left=active_filter_epochs.start.values, right=active_filter_epochs.end.values).piso.symmetric_difference()
# valid_active_filter_epochs = np.vstack([valid_intervals.left.values.T, valid_intervals.right.values.T]).T
# print(f'post: {valid_active_filter_epochs.shape[0]}') # (37, 2)

# active_filter_epochs = valid_active_filter_epochs


def interleave(list1, list2):
    """ Chat-GPT """
    return [x for pair in zip(list1, list2) for x in pair]


import itertools
def interleave(list1, list2):
    """ human solution """
    return [x for x in itertools.chain.from_iterable(itertools.izip_longest(list1,list2)) if x]




def _get_common_cell_pf_results(long_neuron_ids, short_neuron_ids):
    ## get shared neuron info:
    # this must be done after we rebuild the short_pf1D bins (if we need to) so they continue to match:
    pf_neurons_diff = _compare_computation_results(long_neuron_ids, short_neuron_ids)

    shared_aclus = pf_neurons_diff.intersection #.shape (56,)
    print(f'shared_aclus: {shared_aclus}.\t np.shape: {np.shape(shared_aclus)}')
    # curr_any_context_neurons = pf_neurons_diff.either
    long_only_aclus = pf_neurons_diff.lhs_only
    short_only_aclus = pf_neurons_diff.rhs_only
    print(f'long_only_aclus: {long_only_aclus}.\t np.shape: {np.shape(long_only_aclus)}')
    print(f'short_only_aclus: {short_only_aclus}.\t np.shape: {np.shape(short_only_aclus)}')

    ## Get the normalized_tuning_curves only for the shared aclus (that are common across (long/short/global):
    long_is_included = np.isin(long_neuron_ids, shared_aclus)  #.shape # (104, 63)
    long_incl_aclus = np.array(long_neuron_ids)[long_is_included] #.shape # (98,)
    long_incl_curves = long_pf1D.ratemap.normalized_tuning_curves[long_is_included]  #.shape # (98, 63)
    assert long_incl_aclus.shape[0] == long_incl_curves.shape[0] # (98,) == (98, 63)

    short_is_included = np.isin(short_neuron_ids, shared_aclus)
    short_incl_aclus = np.array(short_neuron_ids)[short_is_included] #.shape (98,)
    short_incl_curves = short_pf1D.ratemap.normalized_tuning_curves[short_is_included]  #.shape # (98, 40)
    assert short_incl_aclus.shape[0] == short_incl_curves.shape[0] # (98,) == (98, 63)
    # assert short_incl_curves.shape[1] == long_incl_curves.shape[1] # short and long should have the same bins

    global_is_included = np.isin(global_neuron_ids, shared_aclus)
    global_incl_aclus = np.array(global_neuron_ids)[global_is_included] #.shape (98,)
    global_incl_curves = global_pf1D.ratemap.normalized_tuning_curves[global_is_included]  #.shape # (98, 63)
    assert global_incl_aclus.shape[0] == global_incl_curves.shape[0] # (98,) == (98, 63)
    assert global_incl_curves.shape[1] == long_incl_curves.shape[1] # global and long should have the same bins

    assert np.alltrue(np.isin(long_incl_aclus, short_incl_aclus))
    assert np.alltrue(np.isin(long_incl_aclus, global_incl_aclus))
    return 


# ==================================================================================================================== #
# 2022-12-15 Importing from TestNeuropyPipeline241                                                                     #
# ==================================================================================================================== #

def _update_nearest_decoded_most_likely_position_callback(start_t, end_t):
    """ Only uses end_t
    Implicitly captures: ipspikesDataExplorer, _get_nearest_decoded_most_likely_position_callback
    
    Usage:
        _update_nearest_decoded_most_likely_position_callback(0.0, ipspikesDataExplorer.t[0])
        _conn = ipspikesDataExplorer.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

    """
    def _get_nearest_decoded_most_likely_position_callback(t):
        """ A callback that when passed a visualization timestamp (the current time to render) returns the most likely predicted position provided by the active_two_step_decoder
        Implicitly captures:
            active_one_step_decoder, active_two_step_decoder
        Usage:
            _get_nearest_decoded_most_likely_position_callback(9000.1)
        """
        active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
        active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
        # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
        assert np.shape(active_time_window_variable)[0] == np.shape(active_most_likely_positions)[1], f"timestamps and num positions must be the same but np.shape(active_time_window_variable): {np.shape(active_time_window_variable)} and np.shape(active_most_likely_positions): {np.shape(active_most_likely_positions)}!"
        last_window_index = np.searchsorted(active_time_window_variable, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        # TODO: CORRECTNESS: why is it returning an index that corresponds to a time later than the current time?
        # for current time t=9000.0
        #     last_window_index: 1577
        #     last_window_time: 9000.5023
        # EH: close enough
        last_window_time = active_time_window_variable[last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        displayed_time_offset = t - last_window_time # negative value if the window time being displayed is in the future
        if _debug_print:
            print(f'for current time t={t}\n\tlast_window_index: {last_window_index}\n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}')
        return (last_window_time, *list(np.squeeze(active_most_likely_positions[:, last_window_index]).copy()))

    t = end_t # the t under consideration should always be the end_t. This is written this way just for compatibility with the ipspikesDataExplorer.sigOnUpdateMeshes (float, float) signature
    curr_t, curr_x, curr_y = _get_nearest_decoded_most_likely_position_callback(t)
    curr_debug_point = [curr_x, curr_y, ipspikesDataExplorer.z_fixed[-1]]
    if _debug_print:
        print(f'tcurr_debug_point: {curr_debug_point}') # \n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}
    ipspikesDataExplorer.perform_plot_location_point('debug_point_plot', curr_debug_point, color='r', render=True)
    return curr_debug_point


# ==================================================================================================================== #
# 2022-12-15 Finishing Up Surprise                                                                                     #
# ==================================================================================================================== #




# ==================================================================================================================== #
# 2022-12-14 Batch Surprise Recomputation                                                                              #
# ==================================================================================================================== #

from numpy import inf # for _normalize_flat_relative_entropy_infs
from sklearn.preprocessing import minmax_scale # for _normalize_flat_relative_entropy_infs

def _normalize_flat_relative_entropy_infs(flat_relative_entropy_results):
    """ Replace np.inf with a maximally high value.
    2022-12-14 WIP

    Usage:
        normalized_flat_relative_entropy_results = _normalize_flat_relative_entropy_infs(flat_relative_entropy_results)
    """

    # Replace np.inf with a maximally high value.
    inf_value_mask = np.isinf(flat_relative_entropy_results) # all the infinte values

    normalized_flat_relative_entropy_results = flat_relative_entropy_results.copy()
    normalized_flat_relative_entropy_results[normalized_flat_relative_entropy_results == inf] = 0  # zero out the infinite values for normalization to the feature range (-1, 1)
    normalized_flat_relative_entropy_results = minmax_scale(normalized_flat_relative_entropy_results, feature_range=(-1, 1)) # normalize to the feature_range (-1, 1)

    # Restore the infinite values at the specified value:
    # normalized_flat_relative_entropy_results[inf_value_mask] = 0.0
    return normalized_flat_relative_entropy_results


# ==================================================================================================================== #
# 2022-12-13 Misc                                                                                                      #
# ==================================================================================================================== #

# def process_session_plots(curr_active_pipeline, active_config_name, debug_print=False):
#     """ Unwrap single config 
#     UNUSED AND UNTESTED

#     Usage:

#         from PendingNotebookCode import process_session_plots

#         # active_config_name = 'maze1'
#         # active_config_name = 'maze2'
#         # active_config_name = 'maze'
#         # active_config_name = 'sprinkle'

#         # active_config_name = 'maze_PYR'

#         # active_config_name = 'maze1_rippleOnly'
#         # active_config_name = 'maze2_rippleOnly'

#         # active_config_name = curr_active_pipeline.active_config_names[0] # get the first name by default
#         active_config_name = curr_active_pipeline.active_config_names[-1] # get the last name

#         active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path = process_session_plots(curr_active_pipeline, active_config_name)

#     """

#     def _subfn_build_pdf_export_metadata(session_descriptor_string, filter_name, out_path=None, debug_print=False):
#         """ OLD - Pre 2022-10-04 - Builds the PDF metadata generating function from the passed info
        
#             session_descriptor_string: a string describing the context of the session like 'sess_kdiba_2006-6-07_11-26-53'
#                 Can be obtained from pipleine via `curr_active_pipeline.sess.get_description()`
#             filter_name: a name like 'maze1'
#             out_path: an optional Path to use instead of generating a new one
            
#         Returns:
#             a function that takes one argument, the display function name, and returns the PDF metadata
            
#         History:
#             Refactored from PhoPy3DPositionAnalysis2021.PendingNotebookCode._build_programmatic_display_function_testing_pdf_metadata on 2022-08-17
            
#         Usage:
#             session_descriptor_string = curr_active_pipeline.sess.get_description()
#             ## PDF Output, NOTE this is single plot stuff: uses active_config_name
#             from matplotlib.backends import backend_pdf, backend_pgf, backend_ps
#             from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_export_metadata

#             filter_name = active_config_name
#             _build_pdf_pages_output_info, out_parent_path = build_pdf_export_metadata(session_descriptor_string, filter_name=active_config_name, out_path=None)
#             _build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
#             print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')

            
#             curr_display_function_name = '_display_1d_placefield_validations'
#             built_pdf_metadata, curr_pdf_save_path = _build_pdf_pages_output_info(curr_display_function_name)
#             with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=built_pdf_metadata) as pdf:
#                 # plt.ioff() # disable displaying the plots inline in the Jupyter-lab notebook. NOTE: does not work in Jupyter-Lab, figures still show
#                 plots = curr_active_pipeline.display(curr_display_function_name, active_config_name) # works, but generates a TON of plots!
#                 # plt.ion()
#                 for fig_idx, a_fig in enumerate(plots):
#                     # print(f'saving fig: {fig_idx+1}/{len(plots)}')
#                     pdf.savefig(a_fig)
#                     # pdf.savefig(a_fig, transparent=True)
#                 # When no figure is specified the current figure is saved
#                 # pdf.savefig()

            
#         """
#         if out_path is None:   
#             out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # A string with the day's date like '2022-01-16'
#             out_path = Path(r'EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').joinpath(out_day_date_folder_name).resolve()
#         else:
#             out_path = Path(out_path) # make sure it's a Path
#         out_path.mkdir(exist_ok=True)

        
#         pho_pdf_metadata = {'Creator': 'Spike3D - TestNeuroPyPipeline116', 'Author': 'Pho Hale', 'Title': session_descriptor_string, 'Subject': '', 'Keywords': [session_descriptor_string]}
#         if debug_print:
#             print(f'filter_name: {filter_name}')

#         def _build_pdf_pages_output_info(display_function_name):
#             """ 
#             Implicitly captures:
#                 programmatic_display_fcn_out_path
#                 session_descriptor_string
#                 pho_pdf_metadata
#                 filter_name
#             """
#             built_pdf_metadata = pho_pdf_metadata.copy()
#             context_tuple = [session_descriptor_string, filter_name, display_function_name]
#             built_pdf_metadata['Title'] = '_'.join(context_tuple)
#             built_pdf_metadata['Subject'] = display_function_name
#             built_pdf_metadata['Keywords'] = ' | '.join(context_tuple)
#             curr_pdf_save_path = out_path.joinpath(('_'.join(context_tuple) + '.pdf'))
#             return built_pdf_metadata, curr_pdf_save_path
        
#         return _build_pdf_pages_output_info, out_path





#     # START FUNCTION BODY ________________________________________________________________________________________________ #
#     print(f'active_config_name: {active_config_name}')

#     active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

#     ## Add the filter to the active context
#     # active_identifying_filtered_session_ctx = active_identifying_session_ctx.adding_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
#     active_identifying_filtered_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name] # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

#     # Get relevant variables:
#     # curr_active_pipeline is set above, and usable here
#     sess = curr_active_pipeline.filtered_sessions[active_config_name]

#     active_computation_results = curr_active_pipeline.computation_results[active_config_name]
#     active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
#     active_computation_config = curr_active_pipeline.computation_results[active_config_name].computation_config
#     active_computation_errors = curr_active_pipeline.computation_results[active_config_name].accumulated_errors
#     print(f'active_computed_data.keys(): {list(active_computed_data.keys())}')
#     print(f'active_computation_errors: {active_computation_errors}')
#     active_pf_1D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
#     active_pf_2D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D']
#     active_pf_1D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_dt', None)
#     active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
#     active_firing_rate_trends = curr_active_pipeline.computation_results[active_config_name].computed_data.get('firing_rate_trends', None)
#     active_one_step_decoder_2D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None) # BayesianPlacemapPositionDecoder
#     active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None) 
#     active_one_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_Decoder', None) # BayesianPlacemapPositionDecoder
#     active_two_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_TwoStepDecoder', None)
#     active_extended_stats = curr_active_pipeline.computation_results[active_config_name].computed_data.get('extended_stats', None)
#     active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
#     active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
#     active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
#     active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
#     active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()
#     curr_spikes_df = sess.spikes_df

#     curr_active_config = curr_active_pipeline.active_configs[active_config_name]
#     curr_active_display_config = curr_active_config.plotting_config

#     active_display_output = curr_active_pipeline.display_output[active_identifying_filtered_session_ctx]
#     print(f'active_display_output: {active_display_output}')

#     # Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18)
#     display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_session_ctx, enable_gui=False, debug_print=True) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
#     master_dock_win = display_output['master_dock_win']
#     app = display_output['app']
#     out_items = display_output['out_items']

#     def _get_curr_figure_format_config():
#         """ Aims to fetch the current figure_format_config and context from the figure_format_config widget:    
#         Implicitly captures: `out_items`, `active_config_name`, `active_identifying_filtered_session_ctx` 
#         """
#         ## Get the figure_format_config from the figure_format_config widget:
#         # Fetch the context from the GUI:
#         _curr_gui_session_ctx, _curr_gui_out_display_items = out_items[active_config_name]
#         _curr_gui_figure_format_config_widget = _curr_gui_out_display_items[active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')] # [0] is seemingly not needed to unpack the tuple
#         if _curr_gui_figure_format_config_widget is not None:
#             # has GUI for config
#             figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config
#         else:
#             # has non-GUI provider of figure_format_config
#             figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config

#         if debug_print:
#             print(f'recovered gui figure_format_config: {figure_format_config}')

#         return figure_format_config

#     figure_format_config = _get_curr_figure_format_config()

#     ## PDF Output, NOTE this is single plot stuff: uses active_config_name
#     filter_name = active_config_name
#     _build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = _subfn_build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
#     print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')
    
    
#     ## Test getting figure save paths:
#     _test_fig_path = curr_active_config.plotting_config.get_figure_save_path('test')
#     print(f'_test_fig_path: {_test_fig_path}\n\t exists? {_test_fig_path.exists()}')

#     return active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path
    

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
        active_containing_dockAreaWidget, app = DockAreaWrapper.build_default_dockAreaWindow(title='active_pf_2D_figures', defer_show=False)
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
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.Render3DTimeCurvesBaseGridMixin import BaseGrid3DTimeCurvesHelper


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

