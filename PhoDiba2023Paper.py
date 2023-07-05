import sys
from copy import deepcopy
from typing import Any, Callable, List
from attrs import define, field, Factory
import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # used for plot_epoch_track_assignments


from enum import Enum
from neuropy.utils.mixins.enum_helpers import ExtendedEnum # used in TrackAssignmentState

from pyphocorehelpers.mixins.key_value_hashable import KeyValueHashableObject
from pyphocorehelpers.indexing_helpers import partition # needed by `AssigningEpochs` to partition the dataframe by aclus

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

## Laps Stuff:
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.result_context import overwriting_display_context
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import determine_long_short_pf1D_indicies_sort_by_peak
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot


from typing import Any, List

from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
import pyphoplacecellanalysis.External.pyqtgraph as pg # pyqtgraph

import matplotlib.pyplot as plt


# Testing:
from pyphocorehelpers.plotting.figure_management import capture_new_figures_decorator


_bak_rcParams = mpl.rcParams.copy()
# mpl.rcParams['toolbar'] = 'None' # disable toolbars
# %matplotlib qt


# ==================================================================================================================== #
# 2023-06-29 - Build Properly sorted ratemaps with potentially missing entries                                         #
# ==================================================================================================================== #

from neuropy.plotting.ratemaps import _help_plot_ratemap_neuronIDs # required for build_shared_sorted_neuronIDs

@function_attributes(short_name=None, tags=['active', 'sort', 'neuron_ids'], input_requires=[], output_provides=[], uses=['_help_plot_ratemap_neuronIDs'], used_by=[], creation_date='2023-06-29 07:46', related_items=[])
def build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind):
    """ 
    
    `sort_ind` should be the indicies to sort `included_unit_neuron_IDs`.
        `included_unit_neuron_IDs` may include entries not present in `ratemap`.
    
    Usage:
        ratemap = long_pf1D.ratemap
        included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
        rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())
    """
    if not isinstance(sort_ind, np.ndarray):
        sort_ind = np.array(sort_ind)
    assert np.size(included_unit_neuron_IDs) == np.size(sort_ind), f"`sort_ind` should be the indicies to sort `included_unit_neuron_IDs`."

    #TODO 2023-06-29 06:50: - [ ] SOOO CLOSE. This is the right way to do it... the way that's done in `neuropy.plotting.ratemaps.plot_ratemap_1D`, but because i'm trying to plot the ratemaps as a heatmap I need to fill the missing entries with appropriately sized np.nans or something.
    active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_neuron_IDs=included_unit_neuron_IDs, debug_print=True)
    n_neurons = len(included_unit_indicies) # n_neurons includes Non-active neurons without a placefield if they're provided in included_unit_indicies.
    if not isinstance(included_unit_indicies, np.ndarray):
        included_unit_indicies = np.array(included_unit_indicies)
    included_unit_indicies

    needed_empty_map_shape = np.shape(active_maps)[1:]

    sorted_included_unit_indicies = included_unit_indicies[sort_ind]
    rediculous_final_sorted_all_included_pfmap = []
    rediculous_final_sorted_all_included_neuron_ID = []

    for i, curr_included_unit_index in enumerate(sorted_included_unit_indicies):
        # `curr_included_unit_index` is either an index into the `included_unit_neuron_IDs` array or None
        ### Three things must be considered for each "row" of the plot: 1. the pfmap curve values, 2. the cell id label displayed to the left of the row, 3. the color which is used for the row.
        if curr_included_unit_index is not None:
            # valid neuron ID, access like normal
            pfmap = active_maps[curr_included_unit_index]
            # normal (non-shared mode)
            curr_ratemap_relative_neuron_IDX = curr_included_unit_index
            curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
            
        else:
            # invalid neuron ID, generate blank entry
            curr_ratemap_relative_neuron_IDX = None # This neuron_ID doesn't correspond to a neuron_IDX in the current ratemap, so we'll mark this value as None
            assert included_unit_neuron_IDs is not None
            curr_neuron_ID = included_unit_neuron_IDs[sort_ind[i]]

            # pfmap = np.zeros((np.shape(active_maps)[1],)) # fully allocated new array of zeros
            pfmap = np.zeros(needed_empty_map_shape) # fully allocated new array of zeros
            
        rediculous_final_sorted_all_included_pfmap.append(pfmap)
        rediculous_final_sorted_all_included_neuron_ID.append(curr_neuron_ID)

    rediculous_final_sorted_all_included_neuron_ID = np.array(rediculous_final_sorted_all_included_neuron_ID)
    
    rediculous_final_sorted_all_included_pfmap = np.vstack(rediculous_final_sorted_all_included_pfmap)
    # rediculous_final_sorted_all_included_pfmap.shape # (68, 117)
    return rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap

# ==================================================================================================================== #
# 2023-06-27 - Paper Figure 1 Code                                                                                     #
# ==================================================================================================================== #

# ==================================================================================================================== #
# 2023-06-21 User Annotations and Paper Figure 1                                                                       #
# ==================================================================================================================== #

@define(slots=False)
class UserAnnotationsManager:
    """ class for holding User Annotations of the data. Performed interactive by the user, and then saved to disk for later use. An example are the selected replays to be used as examples. """
    @staticmethod
    def get_user_annotations():
        """ hardcoded user annotations
        
        Usage:
            user_anootations = get_user_annotations()
            user_anootations

        """
        user_annotations = {}

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([ 13,  14,  15,  25,  27,  28,  31,  37,  42,  45,  48,  57,  61,  62,  63,  76,  79,  82,  89,  90, 111, 112, 113, 115])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([  9,  11,  13,  14,  15,  20,  22,  25,  37,  40,  45,  48,  61, 62,  76,  79,  84,  89,  90,  93,  94, 111, 112, 113, 115, 121])

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131])

        return user_annotations

    @staticmethod
    def update_selections_from_annotations(saved_selection, user_anootations, debug_print=True):
        """ 
        
        saved_selection_L.is_selected
        
        
        saved_selection_L = update_selections_from_annotations(saved_selection_L, user_anootations)
        saved_selection_S = update_selections_from_annotations(saved_selection_S, user_anootations)
        ## re-apply the selections:
        pagination_controller_L.restore_selections(saved_selection_L)
        pagination_controller_S.restore_selections(saved_selection_S)

        """
        final_figure_context = saved_selection.figure_ctx
        was_annotation_found = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_anootations.items():
            an_item_diff = a_ctx.diff(final_figure_context)
            if debug_print:
                print(an_item_diff)
                print(f'\t{len(an_item_diff)}')
            if an_item_diff == {('user_annotation', 'selections')}:
                print(f'item found: {a_ctx}\nselections_array: {selections_array}')
                was_annotation_found = True
                saved_selection.is_selected = np.isin(saved_selection.flat_all_data_indicies, selections_array) # update the is_selected
                break # done looking
            
            # print(IdentifyingContext.subtract(a_ctx, final_context_L))
        if not was_annotation_found:
            print(f'WARNING: no matching context found in {len(user_anootations)} annotations. `saved_selection` will be returned unaltered.')
        return saved_selection



class TrackAssignmentState(ExtendedEnum):
    """Docstring for TrackAssignmentState."""
    UNASSIGNED = "unassigned"
    LONG_TRACK = "long_track"
    SHORT_TRACK = "short_track"
    NEITHER = "neither"
    

    @property
    def displayColor(self):
        return self.displayColorList()[self]
    
    @classmethod
    def displayColorList(cls):
        return cls.build_member_value_dict(['grey', 'blue', 'red', 'black'])
    

    # track_assignment = epoch['track_assignment'].decision.name
    # # Set the color of the epoch based on the track assignment
    # if track_assignment == 'UNASSIGNED':
    #     color = 'grey'
    # elif track_assignment == 'LONG_TRACK':
    #     color = 'blue'
    # elif track_assignment == 'SHORT_TRACK':
    #     color = 'red'
    # elif track_assignment == 'NEITHER':
    #     color = 'black'


@define(slots=False, frozen=True)
class TrackAssignmentDecision(KeyValueHashableObject):
    """ represents the decision made to assign a particular epoch to a track (long or short) or to neither track"""
    decision: TrackAssignmentState
    confidence: float # a value between 0 and 1

@define(slots=False)
class AssigningEpochs:
    """ class responsible for iteratively applying various criteria to assign epochs to a particular track starting with the unassigned set of all epochs (in this analysis replay events) """
    filter_epochs_df: pd.DataFrame
    

    @property
    def unassigned_epochs_df(self):
        """A convenience accessor for only the unassigned epochs remaining in the self.filter_epochs_df."""
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        return self.filter_epochs_df[is_unassigned]
    @unassigned_epochs_df.setter
    def unassigned_epochs_df(self, value):
        self.filter_epochs_df[np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])] = value
        

    def _subfn_find_example_epochs(self, spikes_df: pd.DataFrame, exclusive_aclus, included_neuron_ids=None):
        """ aims to find epochs for use as examples in Figure 1:
            1. Contain spikes from aclus that are exclusive to one of the two tracks (either short or long)
            2. Are labeled as "good replays" by my user annotations
            
            Adds: 'contains_one_exclusive_aclu' column to filter_epochs_df
            
            Usage:
                from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _find_example_epochs
                # included_neuron_ids = deepcopy(exclusive_aclus)
                # included_neuron_ids = None
                included_neuron_ids = EITHER_subset.track_exclusive_aclus
                spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
                # filter_epochs = deepcopy(curr_active_pipeline.sess.replay)
                # spikes_df = deepcopy(long_results_obj.spikes_df) # LeaveOneOutDecodingAnalysisResult
                # spikes_df[np.isin(spikes_df.aclu, included_neuron_ids)]
                filter_epochs = deepcopy(long_results_obj.active_filter_epochs)

                filter_epoch_spikes_df, filter_epochs_df = _find_example_epochs(spikes_df, filter_epochs, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids)

        """

        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids
            
        filter_epoch_spikes_df = deepcopy(spikes_df)
        filter_epoch_spikes_df = _prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=self.filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name='replay_epoch_id', debug_print=False) # replay_epoch_id

        unique_replay_epoch_id_values, epoch_split_spikes_df_list = partition(filter_epoch_spikes_df, 'replay_epoch_id')
        # filter_epoch_spikes_df.groupby('replay_epoch_id')
        # unique_replay_epoch_id_values # (198,)
        epoch_spikes_unique_aclus_list = [] # a list of each aclu that fires at least once in the epoch
        epoch_contains_any_exclusive_aclus = []

        for an_epoch_id, epoch_specific_spikes_df in zip(unique_replay_epoch_id_values, epoch_split_spikes_df_list):
            # Loop through the epochs
            # find epochs containing at least one in `exclusive_aclus`:
            epoch_spikes_unique_aclus = epoch_specific_spikes_df.aclu.unique()
            epoch_spikes_unique_aclus_list.append(epoch_spikes_unique_aclus)
            # epoch_spikes_unique_aclus # finds lists of all the active aclus in the given epoch
            if len(epoch_spikes_unique_aclus) > 0:	
                epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
            else:
                epoch_contains_any_exclusive_aclus.append(False)
            # epoch_contains_any_exclusive_aclus # returns True if the epoch contains one ore more spikes from the search list `exclusive_aclus`

        assert len(epoch_contains_any_exclusive_aclus) == np.shape(self.filter_epochs_df)[0]
        self.filter_epochs_df['contains_one_exclusive_aclu'] = epoch_contains_any_exclusive_aclus # adds the appropriate column to the filter_epochs_df
        self.filter_epochs_df['active_unique_aclus'] = epoch_spikes_unique_aclus_list
        return filter_epoch_spikes_df, self.filter_epochs_df


    def determine_if_contains_active_set_exclusive_cells(self, spikes_df: pd.DataFrame, exclusive_aclus, short_exclusive, long_exclusive, included_neuron_ids=None):
        """ NOTE: also updates `filter_epoch_spikes_df` which is returned
    
        adds ['contains_one_exclusive_aclu', 'active_unique_aclus', 'has_SHORT_exclusive_aclu', 'has_LONG_exclusive_aclu'] columns to `filter_epochs_df`
        """
        filter_epoch_spikes_df, self.filter_epochs_df = self._subfn_find_example_epochs(spikes_df, exclusive_aclus=exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
        # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
        self.filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        self.filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        return filter_epoch_spikes_df, self.filter_epochs_df


    def filter_by_user_selections(self, curr_active_pipeline):
        """Get the manual user annotations to determine the good replays for both long/short decoding
        
        Adds the ['long_is_user_included', 'short_is_user_included'] columns to the `filter_epochs_df` DataFrame
        """
        user_annotations = UserAnnotationsManager.get_user_annotations()

        final_context_L = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        final_context_S = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='short_results_obj')
        # _out_pagination_controller.params.active_identifying_figure_ctx.adding_context(None,  user_annotation="selections")
        selections_context_L = final_context_L.adding_context(None,  user_annotation="selections")
        selections_context_S = final_context_S.adding_context(None,  user_annotation="selections")
        
        ## try to get the user annotations for this session:
        selection_idxs_L = user_annotations[selections_context_L]
        selection_idxs_S = user_annotations[selections_context_S]
        
        # for updating the filter_epochs_df (`filter_epochs_df`) from the selections:
        self.filter_epochs_df['long_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_L)
        self.filter_epochs_df['short_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_S)


    def _debug_print_assignment_statuses(self, debug_print=True):
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        is_disregarded = np.array([v.decision == TrackAssignmentState.NEITHER for v in self.filter_epochs_df.track_assignment])
        num_unassigned = np.sum(is_unassigned)
        num_disregarded = np.sum(is_disregarded)
        num_assigned = len(self.filter_epochs_df) - (num_unassigned + num_disregarded)
        if debug_print:
            print(f'num_unassigned: {num_unassigned}, num_disregarded: {num_disregarded}, num_assigned: {num_assigned}')
        return num_unassigned, num_disregarded, num_assigned
    

    

    def _subfn_plot_epoch_track_assignments(self, axis_idx:int, fig=None, axs=None, defer_render=False):
        """ Plots a figure that represents each epoch as a little box that can be colored in based on the track assignment: grey for Unassigned, blue for Long, red for Short, black for Neither.

        Args:
        - assigning_epochs_obj: an object of type AssigningEpochs

        Returns:
        - None
        """
        def _subfn_setup_each_individual_row_axis(ax, is_bottom_row=False):
            """ captures self 
            """
            # Set the x and y limits of the axis
            ax.set_xlim([0, len(self.filter_epochs_df)])
            ax.set_ylim([0, 1])
            
            # Add vertical grid lines
            for i in range(len(self.filter_epochs_df)):
                ax.axvline(x=i, color='white', linewidth=0.5)

            # Set the title and axis labels
            if is_bottom_row:
                ax.set_title('Replay Track Assignments')
                ax.set_xlabel('Replay Epoch Index')
                
            ax.set_ylabel('') # Track Assignment

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            return ax            

        # Create a figure and axis object if not provided
        if fig is None or axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=5, sharex=True, sharey=True, figsize=(20, 10))
            
            # Set up each individual row axis
            for ax in axs:
                ax = _subfn_setup_each_individual_row_axis(ax, is_bottom_row=(ax == axs[-1]))
            
            # Add legend (only to the bottom axis)
            grey_patch = mpatches.Patch(color='grey', label='Unassigned')
            blue_patch = mpatches.Patch(color='blue', label='Long')
            red_patch = mpatches.Patch(color='red', label='Short')
            black_patch = mpatches.Patch(color='black', label='Neither')
            plt.legend(handles=[grey_patch, blue_patch, red_patch, black_patch], loc='lower center', ncol=4)


        # Get the axis object
        ax = axs[axis_idx]

        # Iterate over each epoch in the filter_epochs_df
        for i, epoch in self.filter_epochs_df.iterrows():
            # Get the track assignment of the epoch
            # track_assignment = epoch['track_assignment'].decision.name
            
            track_assignment = epoch['track_assignment'].decision
            track_assignment_color = track_assignment.displayColor

            # # Set the color of the epoch based on the track assignment
            # if track_assignment == 'UNASSIGNED':
            #     color = 'grey'
            # elif track_assignment == 'LONG_TRACK':
            #     color = 'blue'
            # elif track_assignment == 'SHORT_TRACK':
            #     color = 'red'
            # elif track_assignment == 'NEITHER':
            #     color = 'black'

            # Draw a rectangle for the epoch
            ax.add_patch(plt.Rectangle((i, len(ax.get_yticks())), 1, 1, color=track_assignment_color))

        # Show the plot
        if not defer_render:
            plt.show()
            
        return fig, axs
    

    @classmethod
    def main_plot_iterative_epoch_track_assignments(cls, curr_active_pipeline, defer_render=False):
        """ 2023-06-27 - Test how many can actually be sorted by active_set criteria:
        
        """
        ## First filter by only those user-included replays (in the future this won't be done)

        # from PhoDiba2023Paper import TrackAssignmentDecision, TrackAssignmentState, AssigningEpochs
        (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)

        ## Initialize to unassigned
        axis_idx:int = 0
        assigning_epochs_obj.filter_epochs_df['track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.UNASSIGNED, 0.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, defer_render=True)

        # Partition based on whether the user included the epoch in the long or short track in the user-included epochs:
        is_user_exclusive_L = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']))
        is_user_exclusive_S = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included']))
        is_user_unassigned_in_both_epochs = np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']), np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'])) # the user said it was bad in both epochs, so assign it to neither with high confidence

        # NOTE: be sure to assign to the filter_epochs_df, not the unassigned_epochs_df, because the unassigned_epochs_df is a subset of the filter_epochs_df, and we want to assign to the filter_epochs_df so that the unassigned_epochs_df will be a subset of the filter_epochs_df:
        # assign the user_exclusive_L to long_track:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_L, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        # assign the user_exclusive_S to short_track:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_S, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        # assign the user_unassigned_in_both_epochs to neither:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_unassigned_in_both_epochs, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.NEITHER, 1.0)

        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_S]
        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_L]
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)


        # Filter based on the active_set cells (LxC, SxC):
        assigning_epochs_obj.unassigned_epochs_df.loc[np.logical_and(assigning_epochs_obj.unassigned_epochs_df['has_LONG_exclusive_aclu'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['has_SHORT_exclusive_aclu'])), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        # assign the user_exclusive_S to short_track:
        assigning_epochs_obj.unassigned_epochs_df.loc[np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['has_LONG_exclusive_aclu']), assigning_epochs_obj.unassigned_epochs_df['has_SHORT_exclusive_aclu']), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)
        
        if not defer_render:
            fig.show() 
            
        return fig, axs


@function_attributes(short_name=None, tags=['FIGURE1', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 21:40', related_items=[])
def PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline, debug_print=True):
    """ 
    
    # general approach copied from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations._epoch_unit_avg_firing_rates`
    
    # requires: selection_idxs_L, selection_idxs_S
        - Uses the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
        - Uses the `JonathanFiringRateAnalysisResult` global result to get info about the long/short placefields:
    
    long_results_obj, short_results_obj 
    """
   
    

    ## Use the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    ## Use the `JonathanFiringRateAnalysisResult` to get info about the long/short placefields:
    jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
    neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions()

    assigning_epochs_obj = AssigningEpochs(filter_epochs_df=deepcopy(long_results_obj.active_filter_epochs.to_dataframe()))


    # included_neuron_ids = deepcopy(exclusive_aclus)
    # included_neuron_ids = None
    included_neuron_ids = EITHER_subset.track_exclusive_aclus
    spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
    # filter_epochs_df = deepcopy(long_results_obj.active_filter_epochs.to_dataframe())
    filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj.determine_if_contains_active_set_exclusive_cells(spikes_df, exclusive_aclus=EITHER_subset.track_exclusive_aclus, short_exclusive=short_exclusive, long_exclusive=long_exclusive, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'    
    # filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj._find_example_epochs(spikes_df, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
    # # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
    # filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]
    # filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]

    # Get the manual user annotations to determine the good replays for both long/short decoding:
    assigning_epochs_obj.filter_by_user_selections(curr_active_pipeline=curr_active_pipeline)
    

    #### Finally, get only the epochs that meet the criteria:

    # # only include those with one or more exclusive aclu:
    # considered_filter_epochs_df = filter_epochs_df[filter_epochs_df.contains_one_exclusive_aclu].copy()
    # # require inclusion for long or short:
    # considered_filter_epochs_df = considered_filter_epochs_df[np.logical_xor(filter_epochs_df['long_is_user_included'], filter_epochs_df['short_is_user_included'])]

    # Get separate long-side/short-side canidate replays:
    epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
    epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.

    # Common for all rasters:
    new_all_aclus_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline=curr_active_pipeline, curr_any_context_neurons=EITHER_subset.track_exclusive_aclus)

    # Build one spikes_df for Long and Short:
    filter_epoch_spikes_df_L, filter_epoch_spikes_df_S = [_prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=an_epochs_df, included_neuron_ids=EITHER_subset.track_exclusive_aclus, epoch_id_key_name='replay_epoch_id', debug_print=False) for an_epochs_df in (epochs_df_L, epochs_df_S)]    

    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # get the good epoch indicies from epoch_df_L.Index
    good_example_epoch_indicies_L = epochs_df_L.index.to_numpy()
    # good_epoch_indicies_L = np.array([15, 49])
    good_example_epoch_indicies_S = epochs_df_S.index.to_numpy()
    # good_epoch_indicies_S = np.array([ 31,  49,  55,  68,  70,  71,  73,  77,  78,  89,  94, 100, 104, 105, 111, 114, 117, 118, 122, 123, 131])
    if debug_print:
        print(f'good_example_epoch_indicies_L: {good_example_epoch_indicies_L}')
        print(f'good_example_epoch_indicies_S: {good_example_epoch_indicies_S}')


    return (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj


@function_attributes(short_name=None, tags=['FINAL', 'publication', 'figure', 'combined'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 14:33', related_items=[])
def PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=False, save_figure=True):
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
    # curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    # expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
    # rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
    # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result['Flat_epoch_time_bins_mean'], expected_v_observed_result['Flat_decoder_time_bin_centers'], expected_v_observed_result['num_neurons'], expected_v_observed_result['num_timebins_in_epoch'], expected_v_observed_result['num_total_flat_timebins'], expected_v_observed_result['is_short_track_epoch'], expected_v_observed_result['is_long_track_epoch'], expected_v_observed_result['short_short_diff'], expected_v_observed_result['long_long_diff']

    # ==================================================================================================================== #
    # Show 1D Placefields for both Short and Long (top half of the figure)                                                 #
    # ==================================================================================================================== #
    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

    long_single_cell_pfmap_processing_fn = None
    short_single_cell_pfmap_processing_fn = None
    # sort_idx = None
    # sort_idx = sort_ind
    # sort_idx = sort_indicies
    # sort_idx = new_all_aclus_sort_indicies
    sort_idx = 'peak_long'
    # sort_idx = 'bad'

    # out = curr_active_pipeline.display('_display_short_long_pf1D_comparison', active_identifying_session_ctx, single_figure=False, debug_print=True, fignum='Short v Long pf1D Comparison',
    #                                    long_kwargs={'included_unit_neuron_IDs': curr_any_context_neurons, 'included_unit_indicies': None, 'sortby': sort_idx, 'single_cell_pfmap_processing_fn': long_single_cell_pfmap_processing_fn},
    #                                    short_kwargs={'included_unit_neuron_IDs': curr_any_context_neurons, 'included_unit_indicies': None,'sortby': sort_idx, 'single_cell_pfmap_processing_fn': short_single_cell_pfmap_processing_fn, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}},
    #                                   )

    pf1d_compare_graphics = curr_active_pipeline.display('_display_short_long_pf1D_comparison', active_identifying_session_ctx, single_figure=False, debug_print=False, fignum='Short v Long pf1D Comparison',
                                    long_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': long_single_cell_pfmap_processing_fn},
                                    short_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': short_single_cell_pfmap_processing_fn, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}},
                                    save_figure=save_figure,
                                    defer_render=defer_show
                                    )
                                    

    # ax = out.axes[0]


    # ==================================================================================================================== #
    # Show Example Replay Epochs containing the long or short only cells                                                                  #
    # ==================================================================================================================== #
    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)

    # unit_colors_list = None # default rainbow of colors for the raster plots
    neuron_qcolors_list = [pg.mkColor('black') for aclu in EITHER_subset.track_exclusive_aclus] # solid green for all
    unit_colors_list = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)

    # Copy and modify the colors for the cells that are long/short exclusive:
    unit_colors_list_L = deepcopy(unit_colors_list)
    is_L_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, long_exclusive.track_exclusive_aclus) # get long exclusive
    unit_colors_list_L[0, is_L_exclusive] = 255 # [1.0, 0.0, 0.0, 1.0]
    unit_colors_list_L[1, is_L_exclusive] = 0.0
    unit_colors_list_L[2, is_L_exclusive] = 0.0
    
    unit_colors_list_S = deepcopy(unit_colors_list)
    is_S_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, short_exclusive.track_exclusive_aclus) # get short exclusive
    unit_colors_list_S[0, is_S_exclusive] = 0.0 # [1.0, 0.0, 0.0, 1.0]
    unit_colors_list_S[1, is_S_exclusive] = 0.0
    unit_colors_list_S[2, is_S_exclusive] = 255.0
    
    
    ## Build scatterplot args:    
    # Common Tick Label
    vtick = pg.QtGui.QPainterPath()

    # Thicker Tick Label:
    tick_width = 0.25
    # tick_width = 10.0
    half_tick_width = 0.5 * tick_width
    vtick.moveTo(-half_tick_width, -0.5)
    vtick.addRect(-half_tick_width, -0.5, tick_width, 1.0) # x, y, width, height
    pen = {'color': 'white', 'width': 1}
    override_scatter_plot_kwargs = dict(pxMode=True, symbol=vtick, size=6, pen=pen)

    # Not sure if needed:
    # filter_epoch_spikes_df_L.spikes.rebuild_fragile_linear_neuron_IDXs()

    example_epoch_rasters_L = plot_multiple_raster_plot(epochs_df_L, filter_epoch_spikes_df_L, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_L, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                        epoch_id_key_name='replay_epoch_id', scatter_app_name="Long Decoded Example Replays", defer_show=defer_show)
    app_L, win_L, plots_L, plots_data_L = example_epoch_rasters_L
    plots_data_L.active_context = curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='long', epoch='example_replays')
    if save_figure:
        curr_active_pipeline.output_figure(plots_data_L.active_context, win_L) 

    example_epoch_rasters_S = plot_multiple_raster_plot(epochs_df_S, filter_epoch_spikes_df_S, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_S, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                                                    epoch_id_key_name='replay_epoch_id', scatter_app_name="Short Decoded Example Replays", defer_show=defer_show)
    app_S, win_S, plots_S, plots_data_S = example_epoch_rasters_S
    plots_data_S.active_context = curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='short', epoch='example_replays')
    if save_figure:
        curr_active_pipeline.output_figure(plots_data_S.active_context, win_S)

    ## Stacked Epoch Plot
    example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=defer_show, save_figure=save_figure)


    return pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics

# ==================================================================================================================== #
# 2023-06-26 - Paper Figure 2 Code                                                                                     #
# ==================================================================================================================== #
# Instantaneous versions:
# @overwriting_display_context(
@metadata_attributes(short_name=None, tags=['figure_2', 'paper', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-26 21:36', related_items=[])
@define(slots=False, repr=False)
class PaperFigureTwo:
    """ full instantaneous computations for both Long and Short epochs
    
    Usage:
        _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
        _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
        _out_fig_2.display()

    """
    instantaneous_time_bin_size_seconds: float = 0.01 # 20ms
    active_identifying_session_ctx: IdentifyingContext = field(init=False)
    Fig2_Replay_FR: list[tuple[Any, Any]] = field(init=False)
    Fig2_Laps_FR: list[tuple[Any, Any]] = field(init=False)

    _pipeline_file_callback_fn: Callable = field(init=False, repr=False, default=None)


    def compute(self, curr_active_pipeline, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:
        
        """
        sess = curr_active_pipeline.sess 
         # Get the provided context or use the session context:
        active_context = kwargs.get('active_context', sess.get_context()) 
        
        # curr_fig_num = kwargs.pop('fignum', None)
        # if curr_fig_num is None:
        #     ## Set the fig_num, if not already set:
        #     curr_fig_num = f'long|short fr indicies_{active_context.get_description(separator="/")}'
        # kwargs['fignum'] = curr_fig_num

        # graphics_output_dict: MatplotlibRenderPlots = _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pf1D_all, aclu_to_idx, rdf, irdf,
        #     show_inter_replay_frs=show_inter_replay_frs, n_max_plot_rows=n_max_plot_rows, included_unit_neuron_IDs=included_unit_neuron_IDs, cell_spikes_dfs_dict=cell_spikes_dfs_dict, time_variable_name=time_variable_name, defer_render=defer_render, **kwargs)

        self.active_identifying_session_ctx = active_context
        self._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session
        (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)
        
        long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
        long_laps, long_replays, short_laps, short_replays, global_laps, global_replays = [long_short_fr_indicies_analysis_results[k] for k in ['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays']]

        # Replays: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_replays`, `short_replays`
        # LxC: `long_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        LxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ReplayDeltaPlus: `short_replays`
        LxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        SxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ReplayDeltaPlus: `short_replays`
        SxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # Note that in general LxC and SxC might have differing numbers of cells.
        self.Fig2_Replay_FR: list[tuple[Any, Any]] = [(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std()) for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus)]
        

        # Laps/Theta: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_laps`, `short_laps`
        # LxC: `long_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        LxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ThetaDeltaPlus: `short_laps`
        LxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        SxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ThetaDeltaPlus: `short_laps`
        SxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # Note that in general LxC and SxC might have differing numbers of cells.
        self.Fig2_Laps_FR: list[tuple[Any, Any]] = [(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std()) for v in (LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus)]


    @staticmethod
    def fig_2_Theta_FR_pyqtgraph(Fig2_Laps_FR):
        """ Plots the bar graph that displays the Long/Short eXclusive cells during the laps (theta).
        Usage:
            _fig_2_theta_out = fig_2_Theta_FR(Fig2_Laps_FR)
        """
        x_labels = ['LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus']
        
        mean_values = np.array([v[0] for v in Fig2_Laps_FR])
        std_values = np.array([v[1] for v in Fig2_Laps_FR])

        app = pg.mkQApp("new")
        win = pg.GraphicsWindow(title="Laps")
        plot = win.addPlot()

        x_axis = pg.AxisItem(orientation='bottom')
        x_axis.setTicks([list(enumerate(x_labels))])
        plot.setAxisItems({'bottom': x_axis})
        plot.setLabel('left', 'Laps Firing Rates (Hz)')
        plot.setTitle('Lap (Theta) Firing Rates for Long/Short eXclusive Cells on each track')

        bars = pg.BarGraphItem(x=np.arange(len(x_labels)), height=mean_values, width=0.3, brush='b')
        plot.addItem(bars)

        error_bars = pg.ErrorBarItem(x=np.arange(len(x_labels)), y=mean_values, height=std_values, beam=0.2)
        plot.addItem(error_bars)

        return app, win, plot, (bars, error_bars)

    @staticmethod
    def fig_2_Replay_FR_pyqtgraph(Fig2_Replay_FR):
        """ Plots the bar graph that displays the Long/Short eXclusive cells during the replays.
        
        Usage:
            _fig_2_replay_out = fig_2_Replay_FR(Fig2_Replay_FR)
        
        """
        x_labels = ['LxC_RDeltaMinus', 'LxC_RDeltaPlus', 'SxC_RDeltaMinus', 'SxC_RDeltaPlus']
        mean_values = np.array([v[0] for v in Fig2_Replay_FR])
        std_values = np.array([v[1] for v in Fig2_Replay_FR])

        app = pg.mkQApp("new")
        win = pg.GraphicsWindow(title="Replay")
        plot = win.addPlot()

        x_axis = pg.AxisItem(orientation='bottom')
        x_axis.setTicks([list(enumerate(x_labels))])
        plot.setAxisItems({'bottom': x_axis})
        plot.setLabel('left', 'Replay Firing Rates (Hz)')
        plot.setTitle('Replay Firing Rates for Long/Short eXclusive Cells on each track')

        bars = pg.BarGraphItem(x=np.arange(len(x_labels)), height=mean_values, width=0.3, brush='b')
        plot.addItem(bars)

        error_bars = pg.ErrorBarItem(x=np.arange(len(x_labels)), y=mean_values, height=std_values, beam=0.2)
        plot.addItem(error_bars)

        return app, win, plot, (bars, error_bars)
        
    @staticmethod
    @overwriting_display_context(fig='2', frs='Laps')
    def fig_2_Theta_FR_matplotlib(Fig2_Laps_FR, defer_show=False, **kwargs) -> MatplotlibRenderPlots:
            active_context = kwargs.get('active_context', None)
            assert active_context is not None
            print(f'\t{active_context}')
        
            # x_labels = ['LxC_ThetaDeltaMinus', 'LxC_ThetaDeltaPlus', 'SxC_ThetaDeltaMinus', 'SxC_ThetaDeltaPlus']
            x_labels = ['$L_x C$\t$\\theta_{\\Delta -}$', '$L_x C$\t$\\theta_{\\Delta +}$', '$S_x C$\t$\\theta_{\\Delta -}$', '$S_x C$\t$\\theta_{\\Delta +}$']

            mean_values = np.array([v[0] for v in Fig2_Laps_FR])
            std_values = np.array([v[1] for v in Fig2_Laps_FR])

            fig, ax = plt.subplots()
            x = np.arange(len(x_labels))
            width = 0.3

            bars = ax.bar(x, mean_values, width, yerr=std_values, capsize=5)
            
            ax.set_xlabel('Groups')
            ax.set_ylabel('Laps Firing Rates (Hz)')
            ax.set_title('Lap ($\\theta$) Firing Rates for Long/Short eXclusive Cells on each track')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            
            # # Add grouping annotations
            # ax.text((x[0] + x[1]) / 2, -0.25, '$L_x C$', ha='center', va='center')
            # ax.text((x[2] + x[3]) / 2, -0.25, '$S_x C$', ha='center', va='center')
            
            if not defer_show:
                plt.show()
                        
            return MatplotlibRenderPlots(name="fig_2_Theta_FR_matplotlib", figures=[fig], axes=[ax], context=active_context)
            # return fig, ax, bars
        
    @staticmethod
    @overwriting_display_context(fig='2', frs='Replay')
    def fig_2_Replay_FR_matplotlib(Fig2_Replay_FR, defer_show=False, **kwargs) -> MatplotlibRenderPlots:
        
        active_context = kwargs.get('active_context', None)
        assert active_context is not None
        print(f'\t{active_context}')
        
        # x_labels = ['LxC_RDeltaMinus', 'LxC_RDeltaPlus', 'SxC_RDeltaMinus', 'SxC_RDeltaPlus']
        x_labels = ['$L_x C$\t$R_{\\Delta -}$', '$L_x C$\t$R_{\\Delta +}$', '$S_x C$\t$R_{\\Delta -}$', '$S_x C$\t$R_{\\Delta +}$']
        
        mean_values = np.array([v[0] for v in Fig2_Replay_FR])
        std_values = np.array([v[1] for v in Fig2_Replay_FR])

        fig, ax = plt.subplots()
        x = np.arange(len(x_labels))
        width = 0.3

        bars = ax.bar(x, mean_values, width, yerr=std_values, capsize=5)
        
        ax.set_xlabel('Groups')
        ax.set_ylabel('Replay Firing Rates (Hz)')
        ax.set_title('Replay Firing Rates for Long/Short eXclusive Cells on each track')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        if not defer_show:
            plt.show()
                    
        # return fig, ax, bars
        return MatplotlibRenderPlots(name="fig_2_Replay_FR_matplotlib", figures=[fig], axes=[ax], context=active_context)


    def perform_save(self, *args, **kwargs):
        """ used to save the figure without needing a hard reference to curr_active_pipeline """
        assert self._pipeline_file_callback_fn is not None
        return self._pipeline_file_callback_fn(*args, **kwargs) # call the saved callback


    @overwriting_display_context(fig='2')
    def display(self, defer_show=False, save_figure=True, **kwargs):
        # Get the provided context or use the session context:
        active_context = kwargs.get('active_context', self.active_identifying_session_ctx)

        active_context = active_context.adding_context_if_missing(display_fn_name='fig_2_Lap vs Replay FR')
        # curr_fig_num = kwargs.pop('fignum', None)
        # if curr_fig_num is None:
        #     ## Set the fig_num, if not already set:
        #     curr_fig_num = f'long|short fr indicies_{active_context.get_description(separator="/")}'
        # kwargs['fignum'] = curr_fig_num


        # Matplotlib Mode:
        # matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
    
        fig_2_Theta_FR, fig_2_Replay_FR = self.fig_2_Theta_FR_matplotlib, self.fig_2_Replay_FR_matplotlib
        
        # # PyQtGraph Mode:
        # import pyphoplacecellanalysis.External.pyqtgraph as pg
        # fig_2_Theta_FR, fig_2_Replay_FR = self.fig_2_Theta_FR_pyqtgraph, self.fig_2_Replay_FR_pyqtgraph

        # Any Mode:
        _fig_2_theta_out = fig_2_Theta_FR(self.Fig2_Laps_FR, defer_show=defer_show, active_context=active_context)
        _fig_2_replay_out = fig_2_Replay_FR(self.Fig2_Replay_FR, defer_show=defer_show, active_context=active_context)


        def _perform_write_to_file_callback():
            ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            return (self.perform_save(_fig_2_theta_out.context, _fig_2_theta_out.figures[0]), 
                    self.perform_save(_fig_2_replay_out.context, _fig_2_replay_out.figures[0]))

        if save_figure:
             _fig_2_theta_out['saved_figures'],  _fig_2_replay_out['saved_figures'] = _perform_write_to_file_callback()
        else:
            _fig_2_theta_out['saved_figures'],  _fig_2_replay_out['saved_figures'] = [], []

        # Merge the two (_fig_2_theta_out | _fig_2_replay_out)
        return (_fig_2_theta_out, _fig_2_replay_out)
            

# ==================================================================================================================== #
# 2023-06-26 Figure 3                                                                                                  #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['figure_3', 'paper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-27 03:03', related_items=[])
def PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True):
    """ Firing rate index, Long|Short firing rate figures 

    from PendingNotebookCode import PAPER_FIGURE_figure_3
    _out, _out2 = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True)
    
    """
    _out = curr_active_pipeline.display('_display_short_long_firing_rate_index_comparison', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)
    _out2 = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)

    return (_out, _out2)





# ==================================================================================================================== #
# MAIN RUN FUNCTION TO GENERATE ALL FIGURES                                                                            #
# ==================================================================================================================== #
def main_complete_figure_generations(curr_active_pipeline, defer_show=True, save_figure=True):
    """ main run function to generate all figures
    
        from PhoDiba2023Paper import main_complete_figure_generations
        main_complete_figure_generations(curr_active_pipeline)
        
    
    """
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
    pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = PAPER_FIGURE_figure_1_full(curr_active_pipeline) # did not display the pf1





    # Critical new code: Not used anyhwere
    ratemap = long_pf1D.ratemap
    included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
    rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())


    # ==================================================================================================================== #
    # Figure 2) Firing Rate Bar Graphs                                                                                     #
    # ==================================================================================================================== #


    # Instantaneous versions:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends

    _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
    _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
    _out_fig_2.display(defer_show=defer_show, save_figure=save_figure)


    # ==================================================================================================================== #
    # Figure 3) `PAPER_FIGURE_figure_3`: Firing Rate Index and Long/Short Firing Rate Replays v. Laps                      #
    # ==================================================================================================================== #
    _out_fig_3_a, _out_fig_3_b = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=defer_show, save_figure=save_figure)

    # # ==================================================================================================================== #
    # # HELPERS: Interactive Components                                                                                      #
    # # ==================================================================================================================== #
    # from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
    # # fig, ax, rect_selector, set_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)

    # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True, should_apply_updates_to_pipeline=False)
    # print(f'grid_bin_bounds: {grid_bin_bounds}')
    # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")

    # # ==================================================================================================================== #
    # # DEBUGGING:                                                                                                           #
    # # ==================================================================================================================== #
    # ### Testing `plot_kourosh_activity_style_figure` for debugging:
    # from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import plot_kourosh_activity_style_figure

    # # plot_aclus = EITHER_subset.track_exclusive_aclus.copy()
    # plot_aclus = EITHER_subset.track_exclusive_aclus[new_all_aclus_sort_indicies].copy()
    # _out_A = plot_kourosh_activity_style_figure(long_results_obj, long_session, plot_aclus, unit_sort_order=new_all_aclus_sort_indicies, epoch_idx=13, callout_epoch_IDXs=None, skip_rendering_callouts=False)
    # app, win, plots, plots_data = _out_A

    # plots


