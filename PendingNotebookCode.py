## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from typing import List
from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt

from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

should_force_recompute_placefields = True
should_display_2D_plots = True
_debug_print = False


import sys

# ==================================================================================================================== #
# 2023-04-17 - Factor out interactive diagnostic figure code                                                           #
# ==================================================================================================================== #
## Create a diagnostic plot that plots a stack of the three curves used for computations in the given epoch:

from attrs import define, Factory
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import LeaveOneOutDecodingResult
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult


@define(slots=False, repr=False)
class TimebinnedNeuronActivity:
    """ keeps track of which neurons are active and inactive in each decoded timebin """
    n_timebins: int
    active_IDXs: np.ndarray
    active_aclus: np.ndarray
    inactive_IDXs: np.ndarray
    inactive_aclus: np.ndarray
    
    time_bin_centers: np.ndarray # the timebin center times that each time bin corresponds to

    # derived
    num_timebin_active_aclus: np.ndarray = None # int ndarray, the number of active aclus in each timebin
    is_timebin_valid: np.ndarray = None # bool ndarray, whether there is at least one aclu active in each timebin

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        self.num_timebin_active_aclus = np.array([len(timebin_aclus) for timebin_aclus in self.active_aclus]) # .shape # (2917,)
        self.is_timebin_valid = (self.num_timebin_active_aclus > 0) # NEVERMIND: already is the leave-one-out result, so don't do TWO or more aclus in each timebin constraint due to leave-one-out-requirements

    @classmethod
    def init_from_results_obj(cls, results_obj: SurpriseAnalysisResult):
        n_timebins = np.sum(results_obj.all_included_filter_epochs_decoder_result.nbins)
        # a list of lists where each list contains the aclus that are active during that timebin:
        timebins_active_neuron_IDXs = [np.array(results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in np.logical_not(results_obj.is_non_firing_time_bin).T]
        timebins_active_aclus = [np.array(results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_active_neuron_IDXs]

        timebins_inactive_neuron_IDXs = [np.array(results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in results_obj.is_non_firing_time_bin.T]
        timebins_inactive_aclus = [np.array(results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_inactive_neuron_IDXs]
        # timebins_p_x_given_n = np.hstack(results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)        

        assert np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins)[1] == n_timebins, f"the last dimension of long_results_obj.flat_all_epochs_decoded_epoch_time_bins should be equal to n_timebins but instead np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins): {np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins)} " 
        # long_results_obj.flat_all_epochs_decoded_epoch_time_bins[0].shape
        time_bin_centers = results_obj.flat_all_epochs_decoded_epoch_time_bins[0].copy()
        return cls(n_timebins=n_timebins, active_IDXs=timebins_active_neuron_IDXs, active_aclus=timebins_active_aclus, inactive_IDXs=timebins_inactive_neuron_IDXs, inactive_aclus=timebins_inactive_aclus,
                    time_bin_centers=time_bin_centers)


@define(slots=False, repr=False)
class DiagnosticDistanceMetricFigure:
    """ 2023-04-14 - Metric Figure - Plots a vertical stack of 3 subplots with synchronized x-axes. 
    TOP: At the top is the placefield of the first firing cell in the current timebin.
    MID: The middle shows a placefield of a randomly chosen cell from the set that wasn't firing in this timebin.
    BOTTOM: The bottom shows the current timebin's decoded posterior (p_x_given_n)


    Usage: (for use in Jupyter Notebook)
        ```python
        from PendingNotebookCode import DiagnosticDistanceMetricFigure
        import ipywidgets as widgets
        from IPython.display import display

        def integer_slider(update_func):
            slider = widgets.IntSlider(description='Slider:', min=0, max=100, value=0)
            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    # Call the user-provided update function with the current slider index
                    update_func(change['new'])
            slider.observe(on_slider_change)
            display(slider)


        timebinned_neuron_info = long_results_obj.timebinned_neuron_info
        active_fig_obj, update_function = DiagnosticDistanceMetricFigure.build_interactive_diagnostic_distance_metric_figure(long_results_obj, timebinned_neuron_info, result)
        # Call the integer_slider function with the update function
        integer_slider(update_function)
        ```

    History:
        2023-04-17 - Refactored to class from standalone function `_build_interactive_diagnostic_distance_metric_figure`
    """

    results_obj: SurpriseAnalysisResult
    timebinned_neuron_info: TimebinnedNeuronActivity
    result: LeaveOneOutDecodingResult
    hardcoded_sub_epoch_item_idx: int = 0

    ## derived
    plot_dict: dict = Factory(dict) # holds the pyqtgraph plot objects
    plot_data: dict = Factory(dict)
    is_valid: bool = False
    ## Graphics
    win: pg.GraphicsLayoutWidget = None

    @property
    def n_timebins(self):
        """The total number of timebins."""
        return np.sum(self.results_obj.all_epochs_num_epoch_time_bins)


    # ==================================================================================================================== #
    # Initializer                                                                                                          #
    # ==================================================================================================================== #

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # Perform the primary setup to build the placefield
        self.win = pg.GraphicsLayoutWidget(show=True, title='diagnostic_plot')
        # plot_data = {'curr_cell_pf_curve': curr_cell_pf_curve, 'curr_random_not_firing_cell_pf_curve': curr_random_not_firing_cell_pf_curve, 'curr_timebin_p_x_given_n': curr_timebin_p_x_given_n}
        # plot_data = {'curr_cell_pf_curve': None, 'curr_random_not_firing_cell_pf_curve': None, 'curr_timebin_p_x_given_n': None}

        is_valid = False
        for index in np.arange(self.timebinned_neuron_info.n_timebins):
            # find the first valid index
            if not is_valid:
                self.plot_data, is_valid, (normal_surprise, random_surprise) = self._get_updated_plot_data(index)
                print(f'first valid index: {index}')

        self.plot_dict = self._initialize_plots()


    # Private Methods ____________________________________________________________________________________________________ #
    def _get_updated_plot_data(self, index):
        """ called to actually get the plot data for any given timebin index """
        curr_random_not_firing_cell_pf_curve = self.result.random_noise_curves[index]
        curr_decoded_timebins_p_x_given_n = self.result.decoded_timebins_p_x_given_n[index]
        neuron_IDX, aclu = self.timebinned_neuron_info.active_IDXs[index], self.timebinned_neuron_info.active_aclus[index]
        if len(neuron_IDX) > 0:
            # Get first index
            is_valid = True
            neuron_IDX = neuron_IDX[self.hardcoded_sub_epoch_item_idx]
            aclu = aclu[self.hardcoded_sub_epoch_item_idx]
            # curr_cell_pf_curve = long_results_obj.original_1D_decoder.pf.ratemap.tuning_curves[neuron_IDX]
            curr_cell_pf_curve = self.results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[neuron_IDX]

            if curr_random_not_firing_cell_pf_curve.ndim > 1:
                curr_random_not_firing_cell_pf_curve = curr_random_not_firing_cell_pf_curve[self.hardcoded_sub_epoch_item_idx]

            if curr_decoded_timebins_p_x_given_n.ndim > 1:
                curr_decoded_timebins_p_x_given_n = curr_decoded_timebins_p_x_given_n[self.hardcoded_sub_epoch_item_idx]

            # curr_timebin_p_x_given_n = curr_timebins_p_x_given_n[:, index]
            curr_timebin_p_x_given_n = curr_decoded_timebins_p_x_given_n
            normal_surprise, random_surprise = self.result.one_left_out_posterior_to_pf_surprises[index][self.hardcoded_sub_epoch_item_idx], self.result.one_left_out_posterior_to_scrambled_pf_surprises[index][self.hardcoded_sub_epoch_item_idx]
            updated_plot_data = {'curr_cell_pf_curve': curr_cell_pf_curve, 'curr_random_not_firing_cell_pf_curve': curr_random_not_firing_cell_pf_curve, 'curr_timebin_p_x_given_n': curr_timebin_p_x_given_n}
            
        else:
            # Invalid period
            is_valid = False
            normal_surprise, random_surprise = None, None
            updated_plot_data = {'curr_cell_pf_curve': None, 'curr_random_not_firing_cell_pf_curve': None, 'curr_timebin_p_x_given_n': None}

        return updated_plot_data, is_valid, (normal_surprise, random_surprise)


    @staticmethod 
    def _add_plot(win: pg.GraphicsLayoutWidget, data, name:str):
        plot = win.addPlot() # PlotItem has to be built first?
        curve = plot.plot(data, name=name, label=name)
        plot.setLabel('top', name)
        return plot, curve

    
    def _initialize_plots(self):
        for i, (name, data) in enumerate(self.plot_data.items()):
            plot_item, curve = self._add_plot(self.win, data=data, name=name)
            self.plot_dict[name] = {'plot_item':plot_item,'curve':curve}
            if i == 0:
                first_curve_name = name
            else:
                self.plot_dict[name]['plot_item'].setYLink(first_curve_name)  ## test linking by name
            self.win.nextRow()
        return self.plot_dict


    def _update_plots(self, updated_plot_data):
        """ updates the plots created with `_initialize_plots`"""
        for i, (name, data) in enumerate(updated_plot_data.items()):
            curr_plot = self.plot_dict[name]['plot_item']
            curr_curve = self.plot_dict[name]['curve']
            if data is not None:
                curr_curve.setData(data)
            else:
                curr_curve.setData([])

    # Public Functions ___________________________________________________________________________________________________ #
    def update_function(self, index):
        """ Define an update function that will be called with the current slider index 
        Captures plot_dict, and all data variables
        """
        # print(f'Slider index: {index}')
        hardcoded_sub_epoch_item_idx = 0
        updated_plot_data, is_valid, (normal_surprise, random_surprise) = self._get_updated_plot_data(index)
        self.plot_data = updated_plot_data
        self.is_valid = is_valid

        if is_valid:
            if normal_surprise > random_surprise:
                # Set the pen color to green
                pen = pg.mkPen(color='g')
            else:
                pen = pg.mkPen(color='w')

            self._update_plots(updated_plot_data)
            self.plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"{normal_surprise}")
            self.plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"{random_surprise}")
            curr_curve = self.plot_dict['curr_cell_pf_curve']['curve']
            curr_curve.setPen(pen)

        else:
            # Invalid period
            self.plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            self.plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            self._update_plots(updated_plot_data)


    @classmethod
    def build_interactive_diagnostic_distance_metric_figure(cls, results_obj, timebinned_neuron_info, result):
        out_obj = cls(results_obj, timebinned_neuron_info, result)
        return out_obj, out_obj.update_function
    

    # return win, self.plot_dict, plot_data, update_function



def _build_interactive_diagnostic_distance_metric_figure(results_obj, timebinned_neuron_info, result, debug_print = False):
    """ 2023-04-14 - Metric Figure - Plots a vertical stack of 3 subplots with synchronized x-axes. 
    TOP: At the top is the placefield of the first firing cell in the current timebin.
    MID: The middle shows a placefield of a randomly chosen cell from the set that wasn't firing in this timebin.
    BOTTOM: The bottom shows the current timebin's decoded posterior (p_x_given_n)


    Usage: (for use in Jupyter Notebook)
        ```python
        import ipywidgets as widgets
        from IPython.display import display

        def integer_slider(update_func):
            slider = widgets.IntSlider(description='Slider:', min=0, max=100, value=0)
            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    # Call the user-provided update function with the current slider index
                    update_func(change['new'])
            slider.observe(on_slider_change)
            display(slider)


        timebinned_neuron_info = long_results_obj.timebinned_neuron_info
        win, plot_dict, plot_data, update_function = _build_interactive_diagnostic_distance_metric_figure(timebinned_neuron_info, result)
        # Call the integer_slider function with the update function
        integer_slider(update_function)
        ```
    """
    def _get_updated_plot_data(index):
        """ called to actually get the plot data for any given timebin index """
        hardcoded_sub_epoch_item_idx = 0
        curr_random_not_firing_cell_pf_curve = result.random_noise_curves[index]
        curr_decoded_timebins_p_x_given_n = result.decoded_timebins_p_x_given_n[index]
        neuron_IDX, aclu = timebinned_neuron_info.active_IDXs[index], timebinned_neuron_info.active_aclus[index]
        if len(neuron_IDX) > 0:
            # Get first index
            is_valid = True
            neuron_IDX = neuron_IDX[hardcoded_sub_epoch_item_idx]
            aclu = aclu[hardcoded_sub_epoch_item_idx]
            # curr_cell_pf_curve = long_results_obj.original_1D_decoder.pf.ratemap.tuning_curves[neuron_IDX]
            curr_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[neuron_IDX]

            if curr_random_not_firing_cell_pf_curve.ndim > 1:
                curr_random_not_firing_cell_pf_curve = curr_random_not_firing_cell_pf_curve[hardcoded_sub_epoch_item_idx]

            if curr_decoded_timebins_p_x_given_n.ndim > 1:
                curr_decoded_timebins_p_x_given_n = curr_decoded_timebins_p_x_given_n[hardcoded_sub_epoch_item_idx]

            # curr_timebin_p_x_given_n = curr_timebins_p_x_given_n[:, index]
            curr_timebin_p_x_given_n = curr_decoded_timebins_p_x_given_n
            normal_surprise, random_surprise = result.one_left_out_posterior_to_pf_surprises[index][hardcoded_sub_epoch_item_idx], result.one_left_out_posterior_to_scrambled_pf_surprises[index][hardcoded_sub_epoch_item_idx]
            updated_plot_data = {'curr_cell_pf_curve': curr_cell_pf_curve, 'curr_random_not_firing_cell_pf_curve': curr_random_not_firing_cell_pf_curve, 'curr_timebin_p_x_given_n': curr_timebin_p_x_given_n}
            
        else:
            # Invalid period
            is_valid = False
            normal_surprise, random_surprise = None, None
            updated_plot_data = {'curr_cell_pf_curve': None, 'curr_random_not_firing_cell_pf_curve': None, 'curr_timebin_p_x_given_n': None}

        return updated_plot_data, is_valid, (normal_surprise, random_surprise)


    def _add_plot(win: pg.GraphicsLayoutWidget, data, name:str):
        plot = win.addPlot() # PlotItem has to be built first?
        curve = plot.plot(data, name=name, label=name)
        plot.setLabel('top', name)
        return plot, curve

    win = pg.GraphicsLayoutWidget(show=True, title='diagnostic_plot')
    # plot_data = {'curr_cell_pf_curve': curr_cell_pf_curve, 'curr_random_not_firing_cell_pf_curve': curr_random_not_firing_cell_pf_curve, 'curr_timebin_p_x_given_n': curr_timebin_p_x_given_n}
    # plot_data = {'curr_cell_pf_curve': None, 'curr_random_not_firing_cell_pf_curve': None, 'curr_timebin_p_x_given_n': None}

    is_valid = False

    for index in np.arange(timebinned_neuron_info.n_timebins):
        # find the first valid index
        if not is_valid:
            plot_data, is_valid, (normal_surprise, random_surprise) = _get_updated_plot_data(index)
            print(f'first valid index: {index}')


    plot_dict = {}

    ## Many capture `plot_dict`
    def _initialize_plots(plot_data):
        for i, (name, data) in enumerate(plot_data.items()):
            plot_item, curve = _add_plot(win, data=data, name=name)
            plot_dict[name] = {'plot_item':plot_item,'curve':curve}
            if i == 0:
                first_curve_name = name
            else:
                plot_dict[name]['plot_item'].setYLink(first_curve_name)  ## test linking by name
            win.nextRow()
        return plot_dict


    def _update_plots(plot_dict, updated_plot_data):
        """ updates the plots created with `_initialize_plots`"""
        for i, (name, data) in enumerate(updated_plot_data.items()):
            curr_plot = plot_dict[name]['plot_item']
            curr_curve = plot_dict[name]['curve']
            if data is not None:
                curr_curve.setData(data)
            else:
                # curr_plot.clear() # will this mess up the plot by perminantly removing the curve? 
                # curr_curve.clear()
                curr_curve.setData([])

    def update_function(index):
        """ Define an update function that will be called with the current slider index 
        Captures plot_dict, and all data variables
        """
        # print(f'Slider index: {index}')
        hardcoded_sub_epoch_item_idx = 0
        updated_plot_data, is_valid, (normal_surprise, random_surprise) = _get_updated_plot_data(index)
        if is_valid:
            if normal_surprise > random_surprise:
                # Set the pen color to green
                pen = pg.mkPen(color='g')
            else:
                pen = pg.mkPen(color='w')

            _update_plots(plot_dict, updated_plot_data)
            plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"{normal_surprise}")
            plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"{random_surprise}")
            curr_curve = plot_dict['curr_cell_pf_curve']['curve']
            curr_curve.setPen(pen)

        else:
            # Invalid period
            plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            _update_plots(plot_dict, updated_plot_data)

    plot_dict = _initialize_plots(plot_data=plot_data)

    return win, plot_dict, plot_data, update_function





# ==================================================================================================================== #
# 2023-04-14 - New Surprise Implementation                                                                             #
# ==================================================================================================================== #




# Distance metrics used by `_new_compute_surprise`
from scipy.spatial import distance # for Jensen-Shannon distance in `_subfn_compute_leave_one_out_analysis`
import random # for random.choice(mylist)
# from PendingNotebookCode import _scramble_curve
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr


## 0. Precompute the active neurons in each timebin, and the epoch-timebin-flattened decoded posteriors makes it easier to compute for a given time bin:

@function_attributes(short_name='_new_compute_surprise', tags=[], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-14 00:00')
def _new_compute_surprise(results_obj, active_surprise_metric_fn):
    """ 2023-04-14 - To Finish factoring out
        long_results_obj.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(long_results_obj)
        short_results_obj.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(short_results_obj)
        assert long_results_obj.timebinned_neuron_info.n_timebins == short_results_obj.timebinned_neuron_info.n_timebins


    Usage:

        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.jensenshannon(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.correlation(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.sqeuclidean(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: wasserstein_distance(pf, p_x_given_n) # Figure out the correct function for this, it's in my old notebooks
        active_surprise_metric_fn = lambda pf, p_x_given_n: pearsonr(pf, p_x_given_n)[0] # this returns just the correlation coefficient (R), not the p-value due to the [0]


    """
    # Extract important things from the decoded data, like the time bins which are the same for all:
    n_epochs = results_obj.all_included_filter_epochs_decoder_result.num_filter_epochs
    n_timebins = np.sum(results_obj.all_included_filter_epochs_decoder_result.nbins)
    shared_timebin_containers = results_obj.all_included_filter_epochs_decoder_result.time_bin_containers

    results_obj.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(results_obj)

    # @define(slots=False, repr=False)
    # class PlacefieldPosteriorComputationHelper:

    # 	def compute(self, curr_cell_pf_curve, curr_timebin_p_x_given_n):
    # 		result.one_left_out_posterior_to_pf_surprises[timebin_IDX].append(distance.jensenshannon(curr_cell_pf_curve, curr_timebin_p_x_given_n))
    # 		result.one_left_out_posterior_to_pf_correlations[timebin_IDX].append(distance.correlation(curr_cell_pf_curve, curr_timebin_p_x_given_n))


    if active_surprise_metric_fn is None:
        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.jensenshannon(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.correlation(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.sqeuclidean(pf, p_x_given_n)
        # active_surprise_metric_fn = lambda pf, p_x_given_n: wasserstein_distance(pf, p_x_given_n) # Figure out the correct function for this, it's in my old notebooks
        active_surprise_metric_fn = lambda pf, p_x_given_n: pearsonr(pf, p_x_given_n)[0] # this returns just the correlation coefficient (R), not the p-value due to the [0]


    timebinned_neuron_info = results_obj.timebinned_neuron_info
    result = LeaveOneOutDecodingResult(shuffle_IDXs=None)

    pf_shape = (len(results_obj.original_1D_decoder.pf.ratemap.xbin_centers),) # (59, )
    result.random_noise_curves = {}
    # result.random_noise_curves = np.random.uniform(low=0, high=1, size=(timebinned_neuron_info.n_timebins, *pf_shape))
    # result.random_noise_curves = (result.random_noise_curves.T / np.sum(result.random_noise_curves, axis=1)).T # normalize
    # result.random_noise_curves = (result.random_noise_curves.T / np.max(result.random_noise_curves, axis=1)).T # unit max normalization
    result.decoded_timebins_p_x_given_n = {}

    for index in np.arange(timebinned_neuron_info.n_timebins):
        # iterate through timebins
        

        ## Pre loop: add empty array for accumulation
        if index not in result.one_left_out_posterior_to_pf_surprises:
            result.one_left_out_posterior_to_pf_surprises[index] = []
        if index not in result.one_left_out_posterior_to_scrambled_pf_surprises:
            result.one_left_out_posterior_to_scrambled_pf_surprises[index] = []

        # curr_random_not_firing_cell_pf_curve = np.random.uniform(low=0, high=1, size=curr_cell_pf_curve.shape) # generate one at a time
        # curr_random_not_firing_cell_pf_curve = curr_random_not_firing_cell_pf_curve / np.sum(curr_random_not_firing_cell_pf_curve) # normalize
        # result.random_noise_curves.append(curr_random_not_firing_cell_pf_curve)

        # curr_random_not_firing_cell_pf_curve = result.random_noise_curves[index]

        result.random_noise_curves[index] = [] # list
        result.decoded_timebins_p_x_given_n[index] = []

        for neuron_IDX, aclu in zip(timebinned_neuron_info.active_IDXs[index], timebinned_neuron_info.active_aclus[index]):
            # iterate through only the active cells
            # 1. Get set of cells active in a given time bin, for each compute the surprise of its placefield with the leave-one-out decoded posterior.
            left_out_decoder_result = results_obj.one_left_out_filter_epochs_decoder_result_dict[aclu]
            # curr_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[neuron_IDX] # normalized pdf tuning curve
            # curr_cell_spike_curve = original_1D_decoder.pf.ratemap.spikes_maps[unit_IDX] ## not occupancy weighted... is this the right one to use for computing the expected spike rate? NO... doesn't seem like it
            curr_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[neuron_IDX] # Unit max tuning curve

            _, _, curr_timebins_p_x_given_n = left_out_decoder_result.flatten()
            curr_timebin_p_x_given_n = curr_timebins_p_x_given_n[:, index] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_timebin_p_x_given_n.shape[0] == curr_cell_pf_curve.shape[0], f"{curr_timebin_p_x_given_n.shape = } == {curr_cell_pf_curve.shape = }"
            
            # if aclu not in result.one_left_out_posterior_to_pf_surprises:
            # 	result.one_left_out_posterior_to_pf_surprises[aclu] = []
            # result.one_left_out_posterior_to_pf_surprises[aclu].append(distance.jensenshannon(curr_cell_pf_curve, curr_timebin_p_x_given_n))

            result.one_left_out_posterior_to_pf_surprises[index].append(active_surprise_metric_fn(curr_cell_pf_curve, curr_timebin_p_x_given_n))
            # result.one_left_out_posterior_to_pf_correlations[timebin_IDX].append(distance.correlation(curr_cell_pf_curve, curr_timebin_p_x_given_n))

            # 2. From the remainder of cells (those not active), randomly choose one to grab the placefield of and compute the surprise with that and the same posterior.
            # shuffled_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[shuffle_IDXs[i]]

            # a) Use a random non-firing cell's placefield:
            random_not_firing_neuron_IDX = random.choice(timebinned_neuron_info.inactive_IDXs[index])
            # random_not_firing_neuron_IDX = random.choices(timebinned_neuron_info.inactive_IDXs[index], k=)

            # random_not_firing_aclu = random.choice(timebinned_neuron_info.inactive_aclus[i])
            # curr_random_not_firing_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[random_not_firing_neuron_IDX] # normalized pdf tuning curve
            curr_random_not_firing_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[random_not_firing_neuron_IDX] # Unit max tuning curve

            # b) Use a scrambled version of the real curve:
            # curr_random_not_firing_cell_pf_curve = _scramble_curve(curr_cell_pf_curve)


            ## Save the curve for this neuron
            result.random_noise_curves[index].append(curr_random_not_firing_cell_pf_curve)

            # Save the posteriors for this neuron:
            result.decoded_timebins_p_x_given_n[index].append(curr_timebin_p_x_given_n)

            # if aclu not in result.one_left_out_posterior_to_scrambled_pf_surprises:
            # 	result.one_left_out_posterior_to_scrambled_pf_surprises[aclu] = []
            # # The shuffled cell's placefield and the posterior from leaving a cell out:
            # result.one_left_out_posterior_to_scrambled_pf_surprises[aclu].append(distance.jensenshannon(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))
            
            # The shuffled cell's placefield and the posterior from leaving a cell out:
            result.one_left_out_posterior_to_scrambled_pf_surprises[index].append(active_surprise_metric_fn(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))
            # result.one_left_out_posterior_to_scrambled_pf_correlations[timebin_IDX].append(distance.correlation(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))

        # END Neuron Loop
        ## Post neuron loops: convert lists to np.arrays
        result.one_left_out_posterior_to_pf_surprises[index] = np.array(result.one_left_out_posterior_to_pf_surprises[index])
        result.one_left_out_posterior_to_scrambled_pf_surprises[index] = np.array(result.one_left_out_posterior_to_scrambled_pf_surprises[index])
        if len(result.random_noise_curves[index])>0:
            result.random_noise_curves[index] = np.vstack(result.random_noise_curves[index]) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
        else:
            result.random_noise_curves[index] = np.array(result.random_noise_curves[index]) 

        if len(result.decoded_timebins_p_x_given_n[index])>0:
            result.decoded_timebins_p_x_given_n[index] = np.vstack(result.decoded_timebins_p_x_given_n[index]) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
        else:
            result.decoded_timebins_p_x_given_n[index] = np.array(result.decoded_timebins_p_x_given_n[index]) 

    # End Timebin Loop
    ## Post timebin loops compute mean variables:
    result.one_left_out_posterior_to_pf_surprises_mean = {k:np.mean(v) for k, v in result.one_left_out_posterior_to_pf_surprises.items() if np.size(v) > 0}
    result.one_left_out_posterior_to_scrambled_pf_surprises_mean = {k:np.mean(v) for k, v in result.one_left_out_posterior_to_scrambled_pf_surprises.items() if np.size(v) > 0}
    assert len(result.one_left_out_posterior_to_scrambled_pf_surprises_mean) == len(result.one_left_out_posterior_to_pf_surprises_mean)
    assert list(result.one_left_out_posterior_to_scrambled_pf_surprises_mean.keys()) == list(result.one_left_out_posterior_to_pf_surprises_mean.keys())

    valid_time_bin_indicies = np.array(list(result.one_left_out_posterior_to_pf_surprises_mean.keys()))
    one_left_out_posterior_to_pf_surprises_mean = np.array(list(result.one_left_out_posterior_to_pf_surprises_mean.values()))
    one_left_out_posterior_to_scrambled_pf_surprises_mean = np.array(list(result.one_left_out_posterior_to_scrambled_pf_surprises_mean.values()))

    
    # Build Output Dataframes:
    result_df = pd.DataFrame({'time_bin_indices': valid_time_bin_indicies, 'time_bin_centers': timebinned_neuron_info.time_bin_centers[timebinned_neuron_info.is_timebin_valid], 'epoch_IDX': results_obj.all_epochs_reverse_flat_epoch_indicies_array[valid_time_bin_indicies],
        'posterior_to_pf_mean_surprise': one_left_out_posterior_to_pf_surprises_mean, 'posterior_to_scrambled_pf_mean_surprise': one_left_out_posterior_to_scrambled_pf_surprises_mean})
    result_df['surprise_diff'] = result_df['posterior_to_scrambled_pf_mean_surprise'] - result_df['posterior_to_pf_mean_surprise']
    # 24.9 seconds to compute

    ## Compute Aggregate Dataframe for Epoch means:
    # Group by 'epoch_IDX' and compute means of all columns
    result_df_grouped = result_df.groupby('epoch_IDX').mean()
    return results_obj, result, result_df, result_df_grouped



from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_full_session_leave_one_out_decoding_analysis
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_neurons_color_map # for plot_short_v_long_pf1D_comparison


def _long_short_decoding_analysis_from_decoders(long_one_step_decoder_1D, short_one_step_decoder_1D, long_session, short_session, global_session, decoding_time_bin_size = 0.025, perform_cache_load=True):
    """ Uses existing decoders and other long/short variables to run `perform_full_session_leave_one_out_decoding_analysis` on each. """
    # Get existing long/short decoders from the cell under "# 2023-02-24 Decoders"
    long_decoder, short_decoder = deepcopy(long_one_step_decoder_1D), deepcopy(short_one_step_decoder_1D)
    assert np.all(long_decoder.xbin == short_decoder.xbin)

    ## backup existing replay objects
    # long_session.replay_backup, short_session.replay_backup, global_session.replay_backup = [deepcopy(a_session.replay) for a_session in [long_session, short_session, global_session]]
    # null-out the replay objects
    # long_session.replay, short_session.replay, global_session.replay = [None, None, None]

    # Compute/estimate replays if missing from session:
    if not global_session.has_replays:
        print(f'Replays missing from sessions. Computing replays...')
        long_session.replay, short_session.replay, global_session.replay = [a_session.estimate_replay_epochs(min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3).to_dataframe() for a_session in [long_session, short_session, global_session]]

    # Prune to the shared aclus in both epochs (short/long):
    long_shared_aclus_only_decoder, short_shared_aclus_only_decoder = [BasePositionDecoder.init_from_stateful_decoder(a_decoder) for a_decoder in (long_decoder, short_decoder)]
    shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff = BasePositionDecoder.prune_to_shared_aclus_only(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder)

    # n_neurons = len(shared_aclus)
    # # for plotting purposes, build colors only for the common (present in both, the intersection) neurons:
    # neurons_colors_array = build_neurons_color_map(n_neurons, sortby=None, cmap=None)
    # print(f'{n_neurons = }, {neurons_colors_array.shape =}')

    # with VizTracer(output_file=f"viztracer_{get_now_time_str()}-full_session_LOO_decoding_analysis.json", min_duration=200, tracer_entries=3000000, ignore_frozen=True) as tracer:
    long_results_obj = perform_full_session_leave_one_out_decoding_analysis(global_session, original_1D_decoder=long_shared_aclus_only_decoder, decoding_time_bin_size=decoding_time_bin_size, cache_suffix = '_long', perform_cache_load=perform_cache_load) # , perform_cache_load=False
    short_results_obj = perform_full_session_leave_one_out_decoding_analysis(global_session, original_1D_decoder=short_shared_aclus_only_decoder, decoding_time_bin_size=decoding_time_bin_size, cache_suffix = '_short', perform_cache_load=perform_cache_load) # , perform_cache_load=False

    return long_results_obj, short_results_obj

# ==================================================================================================================== #
# 2023-04-10 - Long short expected surprise                                                                            #
# ==================================================================================================================== #
import pyphoplacecellanalysis.External.pyqtgraph as pg

def _scramble_curve(pf: np.ndarray, roll_num_bins:int = 10, method='circ'):
    """ Circularly rotates the 1D placefield """
    return np.roll(pf, roll_num_bins)

def plot_long_short_expected_vs_observed_firing_rates(long_results_obj, short_results_obj, limit_aclus=None):
    """ 2023-03-28 4:30pm - Expected vs. Observed Firing Rates for each cell and each epoch 
    
    Usage:
        win, plots_tuple, legend = plot_long_short_expected_vs_observed_firing_rates(long_results_obj, short_results_obj, limit_aclus=[20])

    """
    num_cells = long_results_obj.original_1D_decoder.num_neurons
    num_epochs = long_results_obj.active_filter_epochs.n_epochs
    # make a separate symbol_brush color for each cell:
    cell_color_symbol_brush = [pg.intColor(i,hues=9, values=3, alpha=180) for i, aclu in enumerate(long_results_obj.original_1D_decoder.neuron_IDs)] # maxValue=128
    # All properties in common:
    win = pg.plot()
     # win.setWindowTitle('Short v. Long - Leave-one-out Expected vs. Observed Firing Rates')
    win.setWindowTitle('Short v. Long - Leave-one-out Expected vs. Observed Num Spikes')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())
    # restrict the aclus to display to limit_aclus
    if limit_aclus is None:
        limit_aclus = long_results_obj.original_1D_decoder.neuron_IDs
    # check whether the neuron_ID is included:
    is_neuron_ID_active = np.isin(long_results_obj.original_1D_decoder.neuron_IDs, limit_aclus)    
    # restrict to the limit indicies
    active_neuron_IDs = np.array(long_results_obj.original_1D_decoder.neuron_IDs)[is_neuron_ID_active]
    active_neuron_IDXs =  np.array(long_results_obj.original_1D_decoder.neuron_IDXs)[is_neuron_ID_active]

    plots_tuple = tuple([{}, {}])
    label_prefix_list = ['long', 'short']
    long_short_symbol_list = ['t', 't1'] # note: 's' is a square. 'o', 't1': triangle pointing upwards
    
    for long_or_short_idx, a_results_obj in enumerate((long_results_obj, short_results_obj)):
        label_prefix = label_prefix_list[long_or_short_idx]
        # print(F'long_or_short_idx: {long_or_short_idx = }, label_prefix: {label_prefix =}')
        plots = plots_tuple[long_or_short_idx]
        curr_symbol = long_short_symbol_list[long_or_short_idx]
        
        ## add scatter plots on top
        for unit_IDX, aclu in zip(active_neuron_IDXs, active_neuron_IDs):
            # find only the time bins when the cell fires:
            curr_epoch_is_cell_active = np.logical_not(a_results_obj.is_non_firing_time_bin)[unit_IDX, :]
            # Use mean time_bin and surprise for each epoch
            curr_epoch_time_bins = a_results_obj.flat_all_epochs_decoded_epoch_time_bins[unit_IDX, curr_epoch_is_cell_active]
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_firing_rates[unit_IDX, curr_epoch_is_cell_active] # measured firing rates (Hz) 
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num measured spikes 
            curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num spikes diff
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_firing_rates[unit_IDX, :] # firing rate diff
            plots[aclu] = win.plot(x=curr_epoch_time_bins, y=curr_epoch_data, pen=None, symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX], name=f'{label_prefix}[{aclu}]', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128)
            legend.addItem(plots[aclu], f'{label_prefix}[{aclu}]')

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Expected vs. Observed # Spikes')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, plots_tuple, legend

def plot_long_short_any_values(long_results_obj, short_results_obj, x, y, limit_aclus=None):
    """ 2023-03-28 4:31pm - Any values, specified by a lambda function for each cell and each epoch 

        x_fn = lambda a_results_obj: a_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0]
        # y_fn = lambda a_results_obj: a_results_obj.all_epochs_all_cells_one_left_out_posterior_to_scrambled_pf_surprises_mean
        # y_fn = lambda a_results_obj: a_results_obj.all_epochs_all_cells_one_left_out_posterior_to_pf_surprises_mean
        y_fn = lambda a_results_obj: a_results_obj.all_epochs_computed_one_left_out_posterior_to_pf_surprises

        # (time_bins, neurons), (epochs, neurons), (epochs)
        # all_epochs_computed_one_left_out_posterior_to_pf_surprises, all_epochs_computed_cell_one_left_out_posterior_to_pf_surprises_mean, all_epochs_all_cells_one_left_out_posterior_to_pf_surprises_mean
        win, plots_tuple, legend = plot_long_short_any_values(long_results_obj, short_results_obj, x=x_fn, y=y_fn, limit_aclus=[20])

    """
    num_cells = long_results_obj.original_1D_decoder.num_neurons
    num_epochs = long_results_obj.active_filter_epochs.n_epochs
    # make a separate symbol_brush color for each cell:
    cell_color_symbol_brush = [pg.intColor(i,hues=9, values=3, alpha=180) for i, aclu in enumerate(long_results_obj.original_1D_decoder.neuron_IDs)] # maxValue=128
    # All properties in common:
    win = pg.plot()
    win.setWindowTitle('Short v. Long - Leave-one-out Custom Surprise Plot')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())
    # restrict the aclus to display to limit_aclus
    if limit_aclus is None:
        limit_aclus = long_results_obj.original_1D_decoder.neuron_IDs
    # check whether the neuron_ID is included:
    is_neuron_ID_active = np.isin(long_results_obj.original_1D_decoder.neuron_IDs, limit_aclus)    
    # restrict to the limit indicies
    active_neuron_IDs = np.array(long_results_obj.original_1D_decoder.neuron_IDs)[is_neuron_ID_active]
    active_neuron_IDXs =  np.array(long_results_obj.original_1D_decoder.neuron_IDXs)[is_neuron_ID_active]

    plots_tuple = tuple([{}, {}])
    label_prefix_list = ['long', 'short']
    long_short_symbol_list = ['t', 'o'] # note: 's' is a square. 'o', 't1': triangle pointing upwards
    
    for long_or_short_idx, a_results_obj in enumerate((long_results_obj, short_results_obj)):
        label_prefix = label_prefix_list[long_or_short_idx]
        # print(F'long_or_short_idx: {long_or_short_idx = }, label_prefix: {label_prefix =}')
        plots = plots_tuple[long_or_short_idx]
        curr_symbol = long_short_symbol_list[long_or_short_idx]
        
        ## add scatter plots on top
        for unit_IDX, aclu in zip(active_neuron_IDXs, active_neuron_IDs):
            # find only the time bins when the cell fires:
            curr_epoch_is_cell_active = np.logical_not(a_results_obj.is_non_firing_time_bin)[unit_IDX, :]
            # Use mean time_bin and surprise for each epoch
            curr_epoch_time_bins = a_results_obj.flat_all_epochs_decoded_epoch_time_bins[unit_IDX, curr_epoch_is_cell_active]
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_firing_rates[unit_IDX, curr_epoch_is_cell_active] # measured firing rates (Hz) 
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num measured spikes 
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num spikes diff
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_firing_rates[unit_IDX, :] # firing rate diff
            print(f'curr_epoch_time_bins.shape: {np.shape(curr_epoch_time_bins)}')
            curr_epoch_data = y(a_results_obj) # [unit_IDX, curr_epoch_is_cell_active]
            print(f'np.shape(curr_epoch_data): {np.shape(curr_epoch_data)}')
            curr_epoch_data = curr_epoch_data[unit_IDX, curr_epoch_is_cell_active]
            plots[aclu] = win.plot(x=curr_epoch_time_bins, y=curr_epoch_data, pen=None, symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX], name=f'{label_prefix}[{aclu}]', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128)
            legend.addItem(plots[aclu], f'{label_prefix}[{aclu}]')

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Surprise (Custom)')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, plots_tuple, legend

def plot_long_short(long_results_obj, short_results_obj):
    win = pg.plot()
    win.setWindowTitle('Short v. Long - Leave-one-out All Cell Average Surprise Outputs')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())

    ax_long = win.plot(x=long_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0], y=long_results_obj.all_epochs_all_cells_computed_surprises_mean, pen=None, symbol='o', symbolBrush=pg.intColor(0,6,maxValue=128), name=f'long') #  symbolBrush=pg.intColor(i,6,maxValue=128)
    legend.addItem(ax_long, f'long')
    ax_short = win.plot(x=short_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0], y=short_results_obj.all_epochs_all_cells_computed_surprises_mean, pen=None, symbol='o', symbolBrush=pg.intColor(1,6,maxValue=128), name=f'short') #  symbolBrush=pg.intColor(i,6,maxValue=128)
    legend.addItem(ax_short, f'short')

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Leave-one-out All Cell Average Surprise')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, (ax_long, ax_short), legend





# ==================================================================================================================== #
# 2023-04-07 - `constrain_to_laps`                                                                                     #
#   Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.
# ==================================================================================================================== #
## 2023-04-07 - Builds the laps using estimation_session_laps(...) if needed for each epoch, and then sets the decoder's .epochs property to the laps object so the occupancy is correct.

from neuropy.analyses.placefields import PfND
from neuropy.analyses.laps import estimate_session_laps # used for `constrain_to_laps`

def constrain_to_laps(curr_active_pipeline):
    """ 2023-04-07 - Constrains the placefields to just the laps, computing the laps if needed.
    Other laps-related things?
        # ??? pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
        # DataSession.compute_position_laps(self)
        # DataSession.compute_laps_position_df(position_df, laps_df)

    Usage:
        from PendingNotebookCode import constrain_to_laps
        curr_active_pipeline = constrain_to_laps(curr_active_pipeline)

    """
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

    for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):
        a_sess = estimate_session_laps(a_sess, should_plot_laps_2d=True)
        curr_laps_obj = a_sess.laps.as_epoch_obj() # set this to the laps object
        curr_laps_obj = curr_laps_obj.get_non_overlapping()
        curr_laps_obj = curr_laps_obj.filtered_by_duration(1.0, 10.0) # the lap must be at least 1 second long and at most 10 seconds long
        # curr_laps_obj = a_sess.estimate_laps().as_epoch_obj()

        ## Check if already the same:
        if curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs == curr_laps_obj:
            print(f'WARNING: constrain_to_laps(...): already had the computations ran with this laps object, so no recomputations are needed.')
            pass
        else:
            # Must recompute since the computation_epochs changed
            curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs = curr_laps_obj
            curr_pf1D, curr_pf2D = a_result.pf1D, a_result.pf2D

            lap_filtered_curr_pf1D = deepcopy(curr_pf1D)
            lap_filtered_curr_pf1D = PfND(spikes_df=lap_filtered_curr_pf1D.spikes_df, position=lap_filtered_curr_pf1D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf1D.config, compute_on_init=True)
            lap_filtered_curr_pf2D = deepcopy(curr_pf2D)
            lap_filtered_curr_pf2D = PfND(spikes_df=lap_filtered_curr_pf2D.spikes_df, position=lap_filtered_curr_pf2D.position, epochs=deepcopy(curr_laps_obj), config=lap_filtered_curr_pf2D.config, compute_on_init=True)
            a_result.pf1D = lap_filtered_curr_pf1D
            a_result.pf2D = lap_filtered_curr_pf2D

        return curr_active_pipeline

def compute_short_long_constrained_decoders(curr_active_pipeline, enable_two_step_decoders:bool = False, recalculate_anyway:bool=True):
    """ 2023-04-14 - Computes both 1D & 2D Decoders constrained to each other's position bins 
    Usage:

        (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline)

        With Two-step Decoders:
        (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, enable_two_step_decoders=True)

    """
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    # 1D Decoders constrained to each other
    def compute_short_long_constrained_decoders_1D(curr_active_pipeline, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 1D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]

        # long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
        long_one_step_decoder_1D, short_one_step_decoder_1D  = [deepcopy(results_data.get('pf1D_Decoder', None)) for results_data in (long_results, short_results)]
        # ds and Decoders conform between the long and the short epochs:
        short_one_step_decoder_1D, did_recompute = short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D, force_recompute=True)

        # ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_1D is None) or (short_two_step_decoder_1D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=1)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
                long_two_step_decoder_1D, short_two_step_decoder_1D  = [results_data.get('pf1D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
                assert (long_two_step_decoder_1D is not None and short_two_step_decoder_1D is not None)
        else:
            long_two_step_decoder_1D, short_two_step_decoder_1D = None, None

        return long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D

    def compute_short_long_constrained_decoders_2D(curr_active_pipeline, enable_two_step_decoders:bool = False):
        """ 2023-04-14 - 2D Decoders constrained to each other, captures: recalculate_anyway, long_epoch_name, short_epoch_name """
        curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
        long_results, short_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name]]
        # Make the 2D Placefields and Decoders conform between the long and the short epochs:
        long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
        short_one_step_decoder_2D, did_recompute = short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        ## Build or get the two-step decoders for both the long and short:
        if enable_two_step_decoders:
            long_two_step_decoder_2D, short_two_step_decoder_2D  = [results_data.get('pf2D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
            if recalculate_anyway or did_recompute or (long_two_step_decoder_2D is None) or (short_two_step_decoder_2D is None):
                curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_two_step_position_decoding_computation'], computation_kwargs_list=[dict(ndim=2)], enabled_filter_names=[long_epoch_name, short_epoch_name], fail_on_exception=True, debug_print=True)
                long_two_step_decoder_2D, short_two_step_decoder_2D  = [results_data.get('pf2D_TwoStepDecoder', None) for results_data in (long_results, short_results)]
                assert (long_two_step_decoder_2D is not None and short_two_step_decoder_2D is not None)
        else:
            long_two_step_decoder_2D, short_two_step_decoder_2D = None, None
        # Sums are similar:
        # print(f'{np.sum(long_one_step_decoder_2D.marginal.x.p_x_given_n) =},\t {np.sum(long_one_step_decoder_1D.p_x_given_n) = }') # 31181.999999999996 vs 31181.99999999999

        ## Validate:
        assert long_one_step_decoder_2D.marginal.x.p_x_given_n.shape == long_one_step_decoder_1D.p_x_given_n.shape, f"Must equal but: {long_one_step_decoder_2D.marginal.x.p_x_given_n.shape =} and {long_one_step_decoder_1D.p_x_given_n.shape =}"
        assert long_one_step_decoder_2D.marginal.x.most_likely_positions_1D.shape == long_one_step_decoder_1D.most_likely_positions.shape, f"Must equal but: {long_one_step_decoder_2D.marginal.x.most_likely_positions_1D.shape =} and {long_one_step_decoder_1D.most_likely_positions.shape =}"

        ## validate values:
        # assert np.allclose(long_one_step_decoder_2D.marginal.x.p_x_given_n, long_one_step_decoder_1D.p_x_given_n), f"1D Decoder should have an x-posterior equal to its own posterior"
        # assert np.allclose(curr_epoch_result['marginal_x']['most_likely_positions_1D'], curr_epoch_result['most_likely_positions']), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"
        return long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D

    ## BEGIN MAIN FUNCTION BODY:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D = compute_short_long_constrained_decoders_1D(curr_active_pipeline, enable_two_step_decoders=enable_two_step_decoders)
    long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D = compute_short_long_constrained_decoders_2D(curr_active_pipeline, enable_two_step_decoders=enable_two_step_decoders)

    if enable_two_step_decoders:
        return (long_one_step_decoder_1D, short_one_step_decoder_1D, long_two_step_decoder_1D, short_two_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D, long_two_step_decoder_2D, short_two_step_decoder_2D)
    else:
        # Only return the one_step decoders
        return (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D)


# ==================================================================================================================== #
# 2023-03-09 - Parameter Sweeping                                                                                      #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND

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


def compute_rankordered_spikes_during_epochs(active_spikes_df, active_epochs):
    """ 
    Usage:
        from neuropy.utils.efficient_interval_search import filter_epochs_by_num_active_units

        active_sess = curr_active_pipeline.filtered_sessions['maze']
        active_epochs = active_sess.perform_compute_estimated_replay_epochs(min_epoch_included_duration=None, max_epoch_included_duration=None, maximum_speed_thresh=None) # filter on nothing basically
        active_spikes_df = active_sess.spikes_df.spikes.sliced_by_neuron_type('pyr') # only look at pyramidal cells

        spike_trimmed_active_epochs, epoch_split_spike_dfs, all_aclus, dense_epoch_split_frs_mat, is_cell_active_in_epoch_mat = filter_epochs_by_num_active_units(active_spikes_df, active_epochs, min_inclusion_fr_active_thresh=2.0, min_num_unique_aclu_inclusions=1)
        epoch_ranked_aclus_dict, active_spikes_df, all_probe_epoch_ids, all_aclus = compute_rankordered_spikes_during_epochs(active_spikes_df, active_epochs)
"""
    from neuropy.utils.mixins.time_slicing import add_epochs_id_identity
    
    # add the active_epoch's id to each spike in active_spikes_df to make filtering and grouping easier and more efficient:
    active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=-1) # uses new add_epochs_id_identity

    # Get all aclus and epoch_idxs used throughout the entire spikes_df:
    all_aclus = active_spikes_df['aclu'].unique()
    all_probe_epoch_ids = active_spikes_df['Probe_Epoch_id'].unique()

    # first_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name].first() # first spikes
    first_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name].median() # median spikes
    # rank the aclu values by their first t value in each Probe_Epoch_id
    ranked_aclus = first_spikes.groupby('Probe_Epoch_id').rank(method='dense') # resolve ties in ranking by assigning the same rank to each and then incrimenting for the next item
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

def process_session_plots(curr_active_pipeline, active_config_name, debug_print=False):
    """ Unwrap single config 
    UNUSED AND UNTESTED

    Usage:

        from PendingNotebookCode import process_session_plots

        # active_config_name = 'maze1'
        # active_config_name = 'maze2'
        # active_config_name = 'maze'
        # active_config_name = 'sprinkle'

        # active_config_name = 'maze_PYR'

        # active_config_name = 'maze1_rippleOnly'
        # active_config_name = 'maze2_rippleOnly'

        # active_config_name = curr_active_pipeline.active_config_names[0] # get the first name by default
        active_config_name = curr_active_pipeline.active_config_names[-1] # get the last name

        active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path = process_session_plots(curr_active_pipeline, active_config_name)

    """
    print(f'active_config_name: {active_config_name}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

    ## Add the filter to the active context
    # active_identifying_filtered_session_ctx = active_identifying_session_ctx.adding_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
    active_identifying_filtered_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name] # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

    # Get relevant variables:
    # curr_active_pipeline is set above, and usable here
    sess = curr_active_pipeline.filtered_sessions[active_config_name]

    active_computation_results = curr_active_pipeline.computation_results[active_config_name]
    active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
    active_computation_config = curr_active_pipeline.computation_results[active_config_name].computation_config
    active_computation_errors = curr_active_pipeline.computation_results[active_config_name].accumulated_errors
    print(f'active_computed_data.keys(): {list(active_computed_data.keys())}')
    print(f'active_computation_errors: {active_computation_errors}')
    active_pf_1D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
    active_pf_2D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D']
    active_pf_1D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_dt', None)
    active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
    active_firing_rate_trends = curr_active_pipeline.computation_results[active_config_name].computed_data.get('firing_rate_trends', None)
    active_one_step_decoder_2D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None) # BayesianPlacemapPositionDecoder
    active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None) 
    active_one_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_Decoder', None) # BayesianPlacemapPositionDecoder
    active_two_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_TwoStepDecoder', None)
    active_extended_stats = curr_active_pipeline.computation_results[active_config_name].computed_data.get('extended_stats', None)
    active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
    active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
    active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
    active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
    active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()
    curr_spikes_df = sess.spikes_df

    curr_active_config = curr_active_pipeline.active_configs[active_config_name]
    curr_active_display_config = curr_active_config.plotting_config

    active_display_output = curr_active_pipeline.display_output[active_identifying_filtered_session_ctx]
    print(f'active_display_output: {active_display_output}')

    # Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18)
    display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_session_ctx, enable_gui=False, debug_print=True) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
    master_dock_win = display_output['master_dock_win']
    app = display_output['app']
    out_items = display_output['out_items']

    def _get_curr_figure_format_config():
        """ Aims to fetch the current figure_format_config and context from the figure_format_config widget:    
        Implicitly captures: `out_items`, `active_config_name`, `active_identifying_filtered_session_ctx` 
        """
        ## Get the figure_format_config from the figure_format_config widget:
        # Fetch the context from the GUI:
        _curr_gui_session_ctx, _curr_gui_out_display_items = out_items[active_config_name]
        _curr_gui_figure_format_config_widget = _curr_gui_out_display_items[active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')] # [0] is seemingly not needed to unpack the tuple
        if _curr_gui_figure_format_config_widget is not None:
            # has GUI for config
            figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config
        else:
            # has non-GUI provider of figure_format_config
            figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config

        if debug_print:
            print(f'recovered gui figure_format_config: {figure_format_config}')

        return figure_format_config

    figure_format_config = _get_curr_figure_format_config()

    ## PDF Output, NOTE this is single plot stuff: uses active_config_name
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_export_metadata

    filter_name = active_config_name
    _build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
    print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')
    
    
    ## Test getting figure save paths:
    _test_fig_path = curr_active_config.plotting_config.get_figure_save_path('test')
    print(f'_test_fig_path: {_test_fig_path}\n\t exists? {_test_fig_path.exists()}')

    return active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path
    

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

