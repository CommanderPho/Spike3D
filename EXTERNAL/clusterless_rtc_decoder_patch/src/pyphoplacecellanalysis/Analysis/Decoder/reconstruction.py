 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind


from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import pyphoplacecellanalysis.General.type_aliases as types
from attrs import define, field, Factory
from datetime import datetime
from neuropy.analyses import Epoch
from neuropy.core import Ratemap # for BasePositionDecoder
from neuropy.core.epoch import ensure_dataframe
import nptyping as ND
from nptyping import NDArray # for DecodedFilterEpochsResult
# import pathlib

import numpy as np
import pandas as pd
# from scipy.stats import multivariate_normal
from scipy.special import factorial, logsumexp

# import neuropy
from neuropy.utils.dynamic_container import DynamicContainer # for decode_specific_epochs
from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # for decode_specific_epochs
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol # allows placefields to be sliced by neuron ids
from neuropy.analyses.decoders import epochs_spkcount # for decode_specific_epochs
from neuropy.utils.mixins.binning_helpers import BinningContainer # for epochs_spkcount getting the correct time bins
from neuropy.analyses.placefields import PfND # for BasePositionDecoder
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr


from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.indexing_helpers import np_ffill_1D # for compute_corrected_positions(...)
from neuropy.utils.mixins.binning_helpers import compute_spanning_bins, build_spanning_grid_matrix # for compute_corrected_positions(...)
from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.indexing_helpers import safe_get_variable_shape
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results # for finding common neurons in `prune_to_shared_aclus_only`
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin
    

# cut_bins = np.linspace(59200, 60800, 9)
# pd.cut(df['column_name'], bins=cut_bins)

# # just want counts of number of occurences of each?
# df['column_name'].value_counts(bins=8, sort=False)

# methods of reconstruction/decoding:

""" 
occupancy gives the P(x) in general.
n_i: the number of spikes fired by each cell during the time window of consideration


np.savez_compressed('test_parameters-neuropy_bayesian_prob.npz', **{'tau':time_bin_size, 'P_x':self.P_x, 'F':self.F, 'n':unit_specific_time_binned_spike_counts})


"""

class ZhangReconstructionImplementation:

    # Shared:    
    @staticmethod
    def compute_time_bins(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
        """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

        # importantly adds 'binned_time' column to spikes_df


        Args:
            spikes_df ([type]): [description]
            time_bin_size ([type]): [description]
            debug_print (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
            
            
        Added Columns to spikes_df:
            'binned_time': the binned time index
        
        Usage:
            time_bin_size=0.02

            curr_result_label = 'maze1'
            sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
            pf = curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf1D']

            spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = compute_time_binned_spiking_activity(spikes_df, time_bin_size)

        """
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'
        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(spikes_df[time_variable_name].to_numpy(), bin_size=max_time_bin_size) # np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]
        spikes_df = spikes_df.spikes.add_binned_time_column(time_window_edges, time_window_edges_binning_info, debug_print=debug_print)
        return time_window_edges, time_window_edges_binning_info, spikes_df
        
    
    @staticmethod
    def compute_unit_specific_bin_specific_spike_counts(spikes_df, time_bin_indicies, debug_print=False):
        """ 
        spikes_df: a dataframe with at least the ['aclu','binned_time'] columns
        time_bin_indicies: np.ndarray of indicies that will be used for the produced output dataframe of the binned spike counts for each unit. (e.g. time_bin_indicies = time_window_edges_binning_info.bin_indicies[:-1]).
        
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'
        assert 'binned_time' in spikes_df.columns
        unit_specific_bin_specific_spike_counts = spikes_df.groupby(['aclu','binned_time'])[time_variable_name].agg('count')
        active_aclu_binned_time_multiindex = unit_specific_bin_specific_spike_counts.index
        active_unique_aclu_values = np.unique(active_aclu_binned_time_multiindex.get_level_values('aclu'))
        unit_specific_binned_spike_counts = np.array([unit_specific_bin_specific_spike_counts[aclu].values for aclu in active_unique_aclu_values]).T # (85841, 40)
        if debug_print:
            print(f'np.shape(unit_specific_spike_counts): {np.shape(unit_specific_binned_spike_counts)}') # np.shape(unit_specific_spike_counts): (40, 85841)
        unit_specific_binned_spike_counts = pd.DataFrame(unit_specific_binned_spike_counts, columns=active_unique_aclu_values, index=time_bin_indicies)
        return unit_specific_binned_spike_counts
    

    @staticmethod
    def compute_time_binned_spiking_activity(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
        """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

        Args:
            spikes_df ([type]): [description]
            time_bin_size ([type]): [description]
            debug_print (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
            
            
        Added Columns to spikes_df:
            'binned_time': the binned time index
        
        Usage:
            time_bin_size=0.02

            curr_result_label = 'maze1'
            sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
            pf = curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf1D']

            spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = compute_time_binned_spiking_activity(spikes_df, time_bin_size)

        """
        time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=max_time_bin_size, debug_print=debug_print) # importantly adds 'binned_time' column to spikes_df
        active_indicies = time_window_edges_binning_info.bin_indicies[1:] # pre-2025-01-13 Old way that led to losing a time bin
        # active_indicies = time_window_edges_binning_info.bin_indicies[:-1] # ABORT - 2025-01-14 New way that supposedly uses correct indexing - ACTUALLY Mismatch: The binned_time column contains labels [1, 2, ..., N] but your DataFrame uses index [0, 1, ..., N-1] when using [:-1]
        unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, active_indicies, debug_print=debug_print) # 2025-01-13 16:14 replaced [1:] with [:-1]
        return unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info

    @classmethod
    def _validate_time_binned_spike_rate_df(cls, time_bins, unit_specific_binned_spike_counts_df):
        """ Validates the outputs of `ZhangReconstructionImplementation.compute_time_binned_spiking_activity(...)`
        
        Usage:
            active_session_spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_count_df, sess_time_window_edges, sess_time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(active_session_spikes_df.copy(), max_time_bin_size=time_bin_size_seconds, debug_print=False) # np.shape(unit_specific_spike_counts): (4188, 108)
            sess_time_binning_container = BinningContainer(edges=sess_time_window_edges, edge_info=sess_time_window_edges_binning_info)
            _validate_time_binned_spike_rate_df(sess_time_binning_container.centers, unit_specific_binned_spike_count_df)

        """
        assert isinstance(unit_specific_binned_spike_counts_df, pd.DataFrame)
        assert unit_specific_binned_spike_counts_df.shape[0] == time_bins.shape[0], f"unit_specific_binned_spike_counts_df.shape[0]: {unit_specific_binned_spike_counts_df.shape[0]} and time_bins.shape[0] {time_bins.shape[0]}"
        print(f'unit_specific_binned_spike_counts_df.shape: {unit_specific_binned_spike_counts_df.shape}')
        nCells = unit_specific_binned_spike_counts_df.shape[1]
        nTimeBins = unit_specific_binned_spike_counts_df.shape[0] # many time_bins
        print(f'nCells: {nCells}, nTimeBins: {nTimeBins}')


    @staticmethod
    def build_concatenated_F(pf, debug_print=False):
        """ returns flattened versions of the occupancy (P_x), and the tuning_curves (F) """
        neuron_IDs = pf.ratemap.neuron_ids
        neuron_IDXs = np.arange(len(neuron_IDs))
        ## 2022-09-19 - should this be the non-normalized tuning curves instead of the normalized ones?
        #   2023-04-07 - CONFIRMED: yes, this should be the non-normalized tuning curves instead of the normalized ones
        maps = pf.ratemap.tuning_curves  # (40, 48) for 1D, (40, 48, 10) for 2D
        if debug_print:
            print(f'maps: {np.shape(maps)}') # maps: (40, 48, 10)

        try:
            f_i = [np.squeeze(maps[i,:,:]) for i in neuron_IDXs] # produces a list of (48 x 10) maps
        except IndexError as e:
            # Happens when called on a 1D decoder
            assert np.ndim(maps) == 2, f"Currently only handles special 1D decoder case but np.shape(maps): {np.shape(maps)} and np.ndim(maps): {np.ndim(maps)} != 2"
            f_i = [np.squeeze(maps[i,:]) for i in neuron_IDXs] # produces a list of (48, ) maps
        except Exception as e:
            raise e        
        if debug_print:
            print(f'np.shape(f_i[i]): {np.shape(f_i[0])}') # (48, 6)
        F_i = [np.reshape(f_i[i], (-1, 1)) for i in neuron_IDXs] # Convert each function to a column vector
        if debug_print:
            print(f'np.shape(F_i[i]): {np.shape(F_i[0])}') # (288, 1)
        # Concatenate each individual F_i to produce F
        F = np.hstack(F_i) #@IgnoreException  
        if debug_print:
            print(f'np.shape(F): {np.shape(F)}') # (288, 40)
        P_x = np.reshape(pf.occupancy, (-1, 1)) # occupancy gives the P(x) in general.
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}') # np.shape(P_x): (48, 6)

        return neuron_IDXs, neuron_IDs, f_i, F_i, F, P_x


    @staticmethod
    def time_bin_spike_counts_N_i(spikes_df, time_bin_size, time_window_edges=None, time_window_edges_binning_info=None, debug_print=False):
        """ Returns the number of spikes that occured for each neuron in each time bin.
        Example:
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
            
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        if (time_window_edges is None) or (time_window_edges_binning_info is None):
            # build time_window_edges/time_window_edges_binning_info AND adds 'binned_time' column to spikes_df. # NOTE: the added 'binned_time' column is 1-indexed, which may explain why we use `time_window_edges_binning_info.bin_indicies[1:]` down below
            time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=time_bin_size, debug_print=debug_print)
        else:
            # already have time bins (time_window_edges/time_window_edges_binning_info) so just add 'binned_time' column to spikes_df if needed:
            assert (time_window_edges_binning_info.num_bins == len(time_window_edges)), f"time_window_edges_binning_info.num_bins: {time_window_edges_binning_info.num_bins}, len(time_window_edges): {len(time_window_edges)}"
            if 'binned_time' not in spikes_df.columns:
                # we must have the 'binned_time' column in spikes_df, so add it if needed
                spikes_df = spikes_df.spikes.add_binned_time_column(time_window_edges, time_window_edges_binning_info, debug_print=debug_print)
        
        # either way to compute the unit_specific_binned_spike_counts:
        active_indicies = time_window_edges_binning_info.bin_indicies[1:] # pre-2025-01-13 Old way that led to losing a time bin
        # active_indicies = time_window_edges_binning_info.bin_indicies[:-1] # 2025-01-14 New way that supposedly uses correct indexing
        assert np.nanmax(spikes_df['binned_time'].to_numpy()) <= np.nanmax(active_indicies), f"np.nanmax(spikes_df['binned_time'].to_numpy()): {np.nanmax(spikes_df['binned_time'].to_numpy())}, np.nanmax(active_indicies): {np.nanmax(active_indicies)}"
        # unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print) # requires 'binned_time' in spikes_df. TODO 2023-05-16 - ERROR: isn't getting centers. time_window_edges_binning_info.bin_indicies[1:]
        unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, active_indicies, debug_print=debug_print) ## removed `time_window_edges_binning_info.bin_indicies[1:]` on 2025-01-13 16:05 
        # unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies, debug_print=debug_print) # ValueError: Shape of passed values is (11816, 80), indices imply (11817, 80)
        
        unit_specific_binned_spike_counts = unit_specific_binned_spike_counts.T # Want the outputs to have each time window as a column, with a single time window giving a column vector for each neuron
        
        assert len(active_indicies) == np.shape(unit_specific_binned_spike_counts.to_numpy())[1], f"len(active_indicies): {len(active_indicies)}, np.shape(unit_specific_binned_spike_counts.to_numpy())[1]: {np.shape(unit_specific_binned_spike_counts.to_numpy())[1]}"
        
        if debug_print:
            print(f'unit_specific_binned_spike_counts.to_numpy(): {np.shape(unit_specific_binned_spike_counts.to_numpy())}') # (85841, 40)
        return unit_specific_binned_spike_counts.to_numpy(), time_window_edges, time_window_edges_binning_info



    @classmethod
    def _validate_time_binned_spike_counts(cls, time_binning_container, unit_specific_binned_spike_counts):
        """ Validates the outputs of `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)`
        
        Usage:
            active_session_spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(active_session_spikes_df.copy(), time_bin_size=time_bin_size_seconds, debug_print=False)  # np.shape(unit_specific_spike_counts): (4188, 108)
            time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
            _validate_time_binned_spike_counts(time_binning_container, unit_specific_binned_spike_counts)

        """
        assert isinstance(unit_specific_binned_spike_counts, np.ndarray)
        assert unit_specific_binned_spike_counts.shape[1] == time_binning_container.center_info.num_bins, f"unit_specific_binned_spike_counts.shape[1]: {unit_specific_binned_spike_counts.shape[1]} and time_binning_container.center_info.num_bins {time_binning_container.center_info.num_bins}"
        print(f'unit_specific_binned_spike_counts.shape: {unit_specific_binned_spike_counts.shape}')
        nCells = unit_specific_binned_spike_counts.shape[0]
        nTimeBins = unit_specific_binned_spike_counts.shape[1] # many time_bins
        print(f'nCells: {nCells}, nTimeBins: {nTimeBins}')


    # Optimal Functions:
    @staticmethod
    def compute_optimal_functions_G(F):
        G = np.linalg.pinv(F).T # Perform the Moore-Penrose pseudoinverse on F to compute G.
        return G
    
    @staticmethod
    def test_identity_deviation(F):
        """ Tests how close the computed F is to the identity matrix, which indicates independence of the functions. """
        identity_approx = np.matmul(F.T, F)
        print(f'np.shape(identity_approx): {np.shape(identity_approx)}') # np.shape(identity_approx): (40, 40)
        identity_deviation = identity_approx - np.identity(np.shape(identity_approx)[0])
        return identity_deviation

    
    @staticmethod
    def neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True, debug_intermediates_mode=False, debug_print=False):
        """ 
            n_i: the number of spikes fired by each cell during the time window of consideration
            use_flat_computation_mode: bool - if True, a more memory efficient accumulating computation is performed that avoids `MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64` caused by allocating the full `cell_prob` matrix
            debug_intermediates_mode: bool - if True, the intermediate computations are stored and returned for debugging purposes. MUCH slower.

        NOTES: Flat vs. Full computation modes:
        Originally 
            cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) 
        This was updated throughout the loop, and then after the loop completed np.prod(cell_prob, axis=2) was used to collapse along axis=2 (nCells), leaving the output posterior with dimensions (nFlatPositionBins, nTimeBins)

        To get around this, I introduced a version that accumulates the multilications over the course of the loop.
            cell_prob = np.ones((nFlatPositionBins, nTimeBins))

        Note: This means that the "Flat" implementation may be more susceptible to numerical underflow, as the intermediate products can become very small, whereas the "Full" implementation does not have this issue. However, the "Flat" implementation can be more efficient in terms of memory usage and computation time, as it avoids creating a large number of intermediate arrays.

        2023-04-19 - I'm confused by the lack of use of P_x in the calculation. According to my written formula P_x was used as the occupancy and multiplied in the output equation. 
            After talking to Kourosh and the lab, it seems that the typical modern approach is to use a uniform `P_x` as to not bias the decoded position.
            But when I went to change my code to try a uniform P_x, it seems that P_x isn't used in the computation at all, and instead only its size is used?
            TODO 2023-04-19 - Check agreement with written equations.
        """
        assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'        
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}, np.shape(F): {np.shape(F)}, np.shape(n): {np.shape(n)}')
        # np.shape(P_x): (1066, 1), np.shape(F): (1066, 66), np.shape(n): (66, 3530)

        nCells = n.shape[0]
        nTimeBins = n.shape[1] # many time_bins
        nFlatPositionBins = np.shape(P_x)[0]

        F = F.T # Transpose F so it's of the right form

        if debug_intermediates_mode:
            assert (not use_flat_computation_mode), "debug_intermediates_mode is only supported when use_flat_computation_mode is False"
            print(f'neuropy_bayesian_prob is running in debug_intermediates_mode. This will be much slower than normal.')
            # To save test parameters:
            out_save_path = 'data/test_parameters-neuropy_bayesian_prob_2023-04-07.npz'
            np.savez_compressed(out_save_path, **{'tau':tau, 'P_x':P_x, 'F':F, 'n':n})
            print(f'saved test parameters to {out_save_path}')

        if use_flat_computation_mode:
            ## Single-cell flat version which updates each iteration:
            cell_prob = np.ones((nFlatPositionBins, nTimeBins)) # Must start with ONES (not Zeros) since we're accumulating multiplications

        else:
            # Full Version which leads to MemoryError when nCells is too large:
            cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) ## MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64

        for cell in range(nCells):
            """ Comparing to the Zhang paper: the output posterior is P_n_given_x (Eqn 35)
                cell_ratemap: [f_{i}(x) for i in range(nCells)]
                cell_spkcnt: [n_{i} for i in range(nCells)]            
            """
            cell_spkcnt = n[cell, :][np.newaxis, :] # .shape: (1, nTimeBins)
            cell_ratemap = F[cell, :][:, np.newaxis] # .shape: (nFlatPositionBins, 1)
            coeff = 1.0 / (factorial(cell_spkcnt)) # 1/factorial(n_{i}) term # .shape: (1, nTimeBins)

            if np.sum(cell_ratemap) == 0:
                ## zero ratemap for this cell encountered, replace with uniform
                cell_ratemap[:, 0] = 1.0/float(nFlatPositionBins) # replace with uniform ratemap
                print(f'WARN: f"np.sum(cell_ratemap): {cell_ratemap} for cell: {cell}", replacing with uniform!')
                # raise ValueError(f"np.sum(cell_ratemap): {cell_ratemap} for cell: {cell}")

            if use_flat_computation_mode:
                # Single-cell flat Version:

                # _temp = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap)) 
                # cell_prob = cell_prob * _temp

                cell_prob *= (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap)) # product equal using *= ## #TODO 2025-01-14 18:35: - [ ] numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.09 GiB for an array with shape (15124, 18557) and data type float64

                # cell_prob.shape (nFlatPositionBins, nTimeBins)
            else:
                # Full Version:
                # broadcasting
                if debug_intermediates_mode:
                    t0 = ((tau * cell_ratemap) ** cell_spkcnt)
                    t1 = coeff
                    t2 = (np.exp(-tau * cell_ratemap))
                    cell_prob[:, :, cell] = (t0 * t1) * t2
                else:
                    cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap))

        if use_flat_computation_mode:
            # Single-cell flat Version:
            posterior = cell_prob # The product has already been accumulating all along
            posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n
        else:
            # Full Version:
            posterior = np.prod(cell_prob, axis=2) # note this product removes axis=2 (nCells)
            posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n

        return posterior



class Zhang_Two_Step:
    """ Two-Step Decoder from Zhang et al. 2018. """

    @classmethod
    def build_all_positions_matrix(cls, x_values, y_values, debug_print=False):
        """ used to build a grid of position points from xbins and ybins.
        Usage:
            all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
        """
        return build_spanning_grid_matrix(x_values, y_values, debug_print=debug_print)

    @classmethod
    def sigma_t(cls, v_t, K, V, d:float=1.0):
        """ The standard deviation of the Gaussian prior for position. Once computed and normalized, can be used such that it only requires the current position (x_t) to return the correct std_dev at a given timestamp.
        K, V are constants
        d is 0.5 for random walks and 1.0 for linear movements. 
        """
        return K * np.power((v_t / V), d)
    
    @classmethod
    def compute_conditional_probability_x_prev_given_x_t(cls, x_prev, x, sigma_t, C):
        """ Should return a value for all possible current locations x_t. x_prev should be a concrete position, not a matrix of them. """
        # multivariate_normal.pdf()        
        # return C * np.exp(-np.square(np.linalg.norm(x - x_prev, axis=1))/(2.0*np.square(sigma_t)))
        numerator = -np.square(np.linalg.norm(x - x_prev, axis=1)) # (1950,)
        denominator = 2.0*np.square(sigma_t) # (64,29)
        # want output of the shape (64,29)
        return C * np.exp(numerator/denominator)

        # a[..., None] + c[None, None, :]
        # output = multivariate_normal.pdf()
        
    @classmethod
    def compute_scaling_factor_k(cls, flat_p_x_given_n):
        """k can be computed in closed from in a vectorized fashion from the one_step bayesian posterior.
        k is a scaling factor that doesn't depend on x_t. Determined by normalizing P(x_t|n_t, x_prev) over x_t"""        
        out_k = np.append(np.nan, np.nansum(flat_p_x_given_n, axis=0)[:-1]) # we'll have one for each time_window_bin_idx [:, time_window_bin_idx]. Want all except the last element ([:-1])
        return out_k
        
    @classmethod
    def compute_bayesian_two_step_prob_single_timestep(cls, one_step_p_x_given_n, x_prev, all_x, sigma_t, C, k):
        return k * one_step_p_x_given_n * cls.compute_conditional_probability_x_prev_given_x_t(x_prev, all_x, sigma_t, C)
    

@custom_define(slots=False, repr=False, eq=False)
class SingleEpochDecodedResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Values for a single epoch. Class to hold debugging information for a transformation process
     
      from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
       
    """
    p_x_given_n: NDArray = non_serialized_field()
    epoch_info_tuple: Tuple = non_serialized_field() # EpochTuple

    most_likely_positions: NDArray = non_serialized_field()
    most_likely_position_indicies: NDArray = non_serialized_field()

    nbins: int = non_serialized_field()
    time_bin_container: Any = non_serialized_field()
    time_bin_edges: NDArray = non_serialized_field()

    marginal_x: DynamicContainer = non_serialized_field()
    marginal_y: Optional[DynamicContainer] = non_serialized_field()
    marginal_z: Optional[DynamicContainer] = non_serialized_field()

    epoch_data_index: Optional[int] = non_serialized_field()

    # active_num_neighbors: int = serialized_attribute_field()
    # active_neighbors_arr: List = field()

    # start_point: Tuple[float, float] = field()
    # end_point: Tuple[float, float] = field()
    # band_width: float = field()

    @property
    def n_xbins(self) -> int:
        """The n_xbins property."""
        return np.shape(self.p_x_given_n)[0]


    @property
    def num_time_windows(self) -> int:
        """The num_time_windows property."""
        return (self.nbins - 1)


    @property
    def flat_p_x_given_n(self) -> NDArray:
        """The num_time_windows property."""
        return deepcopy(self.p_x_given_n).flatten()


    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes  """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        # content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # return f"{type(self).__name__}({content}\n)"
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"



    @function_attributes(short_name=None, tags=['image', 'multi-color-decoder-overlay', 'MultiDecoderColorOverlayedPosteriors'], input_requires=[], output_provides=[], uses=['MultiDecoderColorOverlayedPosteriors'], used_by=['.build_multi_decoder_color_overlay_image', '.save_posterior_as_image'], creation_date='2025-05-14 02:21', related_items=[])    
    def _perform_build_multi_decoder_color_overlay_image_data(self, spikes_df: pd.DataFrame, xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating], lower_bound_alpha=0.0, drop_below_threshold=1e-3, t_bin_size=0.025, use_four_decoders_version: bool = False, **kwargs):
        """ Builds the RGBA output NDArray img_data for the multi-color decoder overlay (that will be built into a PIL.Image object later).
        
        
        global_decoded_result: SingleEpochDecodedResult = a_result.get_result_for_epoch(0)
        # xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating] = deepcopy(a_decoder.xbin)
        
        _out_img, multi_decoder_color_overlay = global_decoded_result.build_multi_decoder_color_overlay_image(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), xbin=deepcopy(a_decoder.xbin))
        
        
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

        if not use_four_decoders_version:
            decoders_version_name: str = 'two_decoders'

        else:
            decoders_version_name: str = 'four_decoders'
                            

        ## INPUTS: global_decoded_result, xbin
        p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = deepcopy(self.p_x_given_n) # .shape # (59, 4, 69488)
        time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = deepcopy(self.time_bin_container.centers)


        # with VizTracer(output_file=f"viztracer_{get_now_time_str()}-MultiDecoderColorOverlayedPosteriors.json", min_duration=200, tracer_entries=3000000, ignore_frozen=True) as tracer:
        multi_decoder_color_overlay: MultiDecoderColorOverlayedPosteriors = MultiDecoderColorOverlayedPosteriors(spikes_df=spikes_df, p_x_given_n=p_x_given_n, time_bin_centers=time_bin_centers, xbin=xbin, lower_bound_alpha=lower_bound_alpha, drop_below_threshold=drop_below_threshold, t_bin_size=t_bin_size)
        multi_decoder_color_overlay.compute_all(compute_four_decoder_version=use_four_decoders_version, progress_print=False) ## single compute
        img_data: NDArray[ND.Shape["N_TIME_BINS, N_POS_BINS, 4"], np.floating] = multi_decoder_color_overlay.extra_all_t_bins_outputs_dict_dict[decoders_version_name]['all_t_bins_final_overlayed_out_RGBA'].astype(float) # (69488, 59, 4)
        img_data = np.transpose(img_data, axes=(1, 0, 2)) # Convert to (H, W, 4)
        return img_data.astype(float), multi_decoder_color_overlay


    @function_attributes(short_name=None, tags=['image', 'multi-color-decoder-overlay', 'MultiDecoderColorOverlayedPosteriors'], input_requires=[], output_provides=[], uses=['._perform_build_multi_decoder_color_overlay_image_data'], used_by=[], creation_date='2025-05-14 02:21', related_items=[])    
    def build_multi_decoder_color_overlay_image(self, spikes_df: pd.DataFrame, xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating], lower_bound_alpha=0.0, drop_below_threshold=1e-3, t_bin_size=0.025, desired_height=None, desired_width=None, **kwargs):
        """ Builds the final PIL.Image RGBA output image for the multi-color decoder overlay.
        
        
        global_decoded_result: SingleEpochDecodedResult = a_result.get_result_for_epoch(0)
        # xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating] = deepcopy(a_decoder.xbin)
        
        _out_img, multi_decoder_color_overlay = global_decoded_result.build_multi_decoder_color_overlay_image(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), xbin=deepcopy(a_decoder.xbin))
        
        
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image, get_array_as_image, get_array_as_image_stack
        from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind

        
        # ## INPUTS: global_decoded_result, xbin
        # p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = deepcopy(self.p_x_given_n) # .shape # (59, 4, 69488)
        # time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = deepcopy(self.time_bin_container.centers)

        # # with VizTracer(output_file=f"viztracer_{get_now_time_str()}-MultiDecoderColorOverlayedPosteriors.json", min_duration=200, tracer_entries=3000000, ignore_frozen=True) as tracer:
        # multi_decoder_color_overlay: MultiDecoderColorOverlayedPosteriors = MultiDecoderColorOverlayedPosteriors(spikes_df=spikes_df, p_x_given_n=p_x_given_n, time_bin_centers=time_bin_centers, xbin=xbin, lower_bound_alpha=lower_bound_alpha, drop_below_threshold=drop_below_threshold, t_bin_size=t_bin_size)
        # multi_decoder_color_overlay.compute_all() ## single compute
        
        # img_data: NDArray[ND.Shape["N_TIME_BINS, N_POS_BINS, 4"], np.floating] = multi_decoder_color_overlay.extra_all_t_bins_outputs_dict_dict['two_decoders']['all_t_bins_final_overlayed_out_RGBA'].astype(float) # (69488, 59, 4)
        
        # img_data = np.transpose(img_data, axes=(1, 0, 2)) # Convert to (H, W, 4)
        
        img_data, multi_decoder_color_overlay = self._perform_build_multi_decoder_color_overlay_image_data(spikes_df=spikes_df, xbin=xbin, lower_bound_alpha=lower_bound_alpha, drop_below_threshold=drop_below_threshold, t_bin_size=t_bin_size)
        
        kwargs = {}
        desired_height = 400
        desired_width = None
        skip_img_normalization = True
        _out_img = get_array_as_image(img_data, desired_height=desired_height, desired_width=desired_width, skip_img_normalization=skip_img_normalization, export_kind=HeatmapExportKind.RAW_RGBA, **kwargs) # Image.Image - NDArray[ND.Shape["IM_HEIGHT, IM_WIDTH, 4"], np.floating]
        
        return _out_img, multi_decoder_color_overlay


    # Images _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #

    @function_attributes(short_name=None, tags=['image', 'posterior'], input_requires=[], output_provides=[], uses=['get_array_as_image'], used_by=[], creation_date='2024-05-09 05:49', related_items=[])
    def get_posterior_as_image(self, epoch_id_identifier_str: str = 'p_x_given_n', desired_height=None, desired_width=None, export_kind: Optional[HeatmapExportKind] = None, skip_img_normalization=True, export_grayscale:bool=False, **kwargs):
        """ gets the posterior as a colormapped image 
        
        Usage:

            posterior_image = active_captured_single_epoch_result.get_posterior_as_image(desired_height=400)
            posterior_image


        """
        from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image
        from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind


        if export_kind is None:
            export_kind = HeatmapExportKind.COLORMAPPED


        if self.epoch_data_index is not None:
            epoch_id_str = f"{epoch_id_identifier_str}[{self.epoch_data_index}]"
        else:
            epoch_id_str = f"{epoch_id_identifier_str}"

        
        # `raw_RGBA_only_parameters` dict with keys: ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
        raw_RGBA_only_parameters = kwargs.pop('raw_RGBA_only_parameters', {})

        if export_kind.value == HeatmapExportKind.RAW_RGBA.value:
            ## check values in `raw_RGBA_only_parameters`:
            required_keys = ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
            assert np.all([k in raw_RGBA_only_parameters.keys() for k in required_keys]), f"missing required keys:\n\tequired_keys: {required_keys}\n\tlist(raw_RGBA_only_parameters.keys()): {list(raw_RGBA_only_parameters.keys())}\n"
            # build_multi_decoder_color_overlay_image
            _out_img, multi_decoder_color_overlay = self.build_multi_decoder_color_overlay_image(**raw_RGBA_only_parameters)
            img_data = _out_img.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        
        else:
            img_data = self.p_x_given_n.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
            

        return get_array_as_image(img_data, desired_height=desired_height, desired_width=desired_width, skip_img_normalization=skip_img_normalization, export_grayscale=export_grayscale, **kwargs)


    @function_attributes(short_name=None, tags=['export', 'image', 'posterior'], input_requires=[], output_provides=[], uses=['pyphocorehelpers.plotting.media_output_helpers.save_array_as_image', '.build_multi_decoder_color_overlay_image', '._perform_build_multi_decoder_color_overlay_image_data'], used_by=['PosteriorExporting.export_decoded_posteriors_as_images'], creation_date='2024-05-09 05:49', related_items=[])
    def save_posterior_as_image(self, parent_array_as_image_output_folder: Union[Path, str]='', epoch_id_identifier_str: str = 'p_x_given_n', complete_epoch_identifier_str: Optional[str]=None, desired_height=None, desired_width=None, export_kind: Optional[HeatmapExportKind] = None, colormap:str='viridis', skip_img_normalization=True, export_grayscale:bool=False, allow_override_aspect_ratio:bool=False, **kwargs):
        """ saves the posterior to disk
        
        this is where `self.build_multi_decoder_color_overlay_image(...)` should be called for export_kind == 
        
        Usage:

            posterior_image, posterior_save_path = active_captured_single_epoch_result.save_posterior_as_image(parent_array_as_image_output_folder='output', desired_height=400)
            posterior_image


        """
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image
        from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportKind
        
        if isinstance(parent_array_as_image_output_folder, str):
            parent_array_as_image_output_folder = Path(parent_array_as_image_output_folder).resolve()
            
        if kwargs.get('debug_print', False):
            print(f'self.epoch_data_index: {self.epoch_data_index}')

        if parent_array_as_image_output_folder.is_dir():
            assert parent_array_as_image_output_folder.exists(), f"path '{parent_array_as_image_output_folder}' does not exist!"
            if complete_epoch_identifier_str is None:
                ## mode to use
                # active_epoch_data_IDX: int = self.epoch_data_index
                curr_epoch_info_dict = self.epoch_info_tuple._asdict()
                active_epoch_id: int = curr_epoch_info_dict.get('label', None)
                if active_epoch_id is not None:
                    active_epoch_id = int(active_epoch_id)
                    complete_epoch_identifier_str = f"{epoch_id_identifier_str}[{active_epoch_id:03d}]" # 2025-06-03 - 'p_x_given_n[067]'
                else:
                    complete_epoch_identifier_str = f"{epoch_id_identifier_str}"

            assert complete_epoch_identifier_str is not None
            _img_path = parent_array_as_image_output_folder.joinpath(f'{complete_epoch_identifier_str}.png').resolve()
        else:
            _img_path = parent_array_as_image_output_folder.resolve() # already a direct path
            assert complete_epoch_identifier_str is None, f"complete_epoch_identifier_str is provided (not-None), complete_epoch_identifier_str: '{complete_epoch_identifier_str}', but a full path was provided in parent_array_as_image_output_folder: {parent_array_as_image_output_folder.as_posix()}! This will override and ignore complete_epoch_identifier_str. So please only specify one or the other."

        if (desired_height is None) and (desired_width is None):
            # only if the user hasn't provided a desired width OR height should we suggest a height of 100
            desired_height = 100
            

        # `raw_RGBA_only_parameters` dict with keys: ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
        raw_RGBA_only_parameters = kwargs.pop('raw_RGBA_only_parameters', {})

        if (export_kind is not None) and (export_kind.value == HeatmapExportKind.RAW_RGBA.value):
            ## check values in `raw_RGBA_only_parameters`:
            required_keys = ['spikes_df', 'xbin', 'lower_bound_alpha', 'drop_below_threshold', 't_bin_size']
            assert np.all([k in raw_RGBA_only_parameters.keys() for k in required_keys]), f"missing required keys:\n\tequired_keys: {required_keys}\n\tlist(raw_RGBA_only_parameters.keys()): {list(raw_RGBA_only_parameters.keys())}\n"
            # ## need to remove these RAW_RGBA-specific kwarg parameters for the other two cases anyway:
            # lower_bound_alpha = kwargs.pop('lower_bound_alpha', 0.1)
            # drop_below_threshold = kwargs.pop('drop_below_threshold', 1e-2)
            # t_bin_size = kwargs.pop('t_bin_size', 0.025)

            # _out_img, multi_decoder_color_overlay = self.build_multi_decoder_color_overlay_image(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), xbin=deepcopy(a_decoder.xbin), lower_bound_alpha=lower_bound_alpha, drop_below_threshold=drop_below_threshold, t_bin_size=t_bin_size)
            # _out_img, multi_decoder_color_overlay = self.build_multi_decoder_color_overlay_image(**raw_RGBA_only_parameters)
            img_data, multi_decoder_color_overlay = self._perform_build_multi_decoder_color_overlay_image_data(**raw_RGBA_only_parameters) # .shape: (4, n_curr_epoch_time_bins) - (400, 454, 4)
        
        else:
            img_data = self.p_x_given_n.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        
        raw_tuple = save_array_as_image(img_data, desired_height=desired_height, desired_width=desired_width, export_kind=export_kind, colormap=colormap, skip_img_normalization=skip_img_normalization, out_path=_img_path, export_grayscale=export_grayscale, allow_override_aspect_ratio=allow_override_aspect_ratio, **kwargs)
        image_raw, path_raw = raw_tuple
        return image_raw, path_raw


    @function_attributes(short_name=None, tags=['epochs_df', 'reconstruct', 'decoding' , 'pure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-24 13:56', related_items=[])
    def build_pseudo_epochs_df_from_decoding_bins(self, epoch_end_non_overlapping_difference: float=1e-9) -> pd.DataFrame:
        """Build another decoding_bin_epochs_df where we have an epoch with epoch_id for each decoded time bin
        
        Usage:
            ## INPUTS: results2D
            single_continuous_result: SingleEpochDecodedResult = results2D.continuous_results['long'].get_result_for_epoch(0) # SingleEpochDecodedResult
            decoding_bins_epochs_df: pd.DataFrame = single_continuous_result.build_pseudo_epochs_df_from_decoding_bins()
            decoding_bins_epochs_df

        """
        # single_continuous_result.nbins
        # single_continuous_result.time_bin_container.edges
        left_edges = self.time_bin_container.edges[:-1]
        right_edges = self.time_bin_container.edges[1:]
        assert len(left_edges) == len(right_edges), f"len(right_edges): {len(right_edges)}, len(left_edges): {len(left_edges)}"
        assert len(left_edges) == self.time_bin_container.num_bins, f"self.time_bin_container.num_bins: {self.time_bin_container.num_bins}, len(left_edges): {len(left_edges)}"
        # decoding_bins_epochs_df: pd.DataFrame = pd.DataFrame({'start': self.time_bin_container.left_edges, 'stop': self.time_bin_container.right_edges})
        decoding_bins_epochs_df: pd.DataFrame = pd.DataFrame({'start': left_edges, 'stop': right_edges})
        decoding_bins_epochs_df['stop'] = decoding_bins_epochs_df['stop'] - epoch_end_non_overlapping_difference # make non-overlapping by subtracting off 1-nano-second from the end of each
        decoding_bins_epochs_df['duration'] = decoding_bins_epochs_df['stop'] - decoding_bins_epochs_df['start']
        decoding_bins_epochs_df['label'] = decoding_bins_epochs_df.index.to_numpy().astype(int)
        assert len(decoding_bins_epochs_df) == self.nbins, f"len(decoding_bins_epochs_df): {len(decoding_bins_epochs_df)}, self.nbins: {self.nbins}"
        assert np.all(decoding_bins_epochs_df['duration'] > 0.0), f"all durations must be strictly greater than zero"
        
        return decoding_bins_epochs_df.epochs.get_valid_df()
                
    # HDFMixin Conformances ______________________________________________________________________________________________ #
    # def to_hdf(self, file_path):
    #     with h5py.File(file_path, 'w') as f:
    #         for attribute, value in self.__dict__.items():
    #             if isinstance(value, pd.DataFrame):
    #                 value.to_hdf(file_path, key=attribute)
    #             elif isinstance(value, np.ndarray):
    #                 f.create_dataset(attribute, data=value)
    #             # ... handle other attribute types as needed ...    

    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, required_zero_padding=None, leafs_include_epoch_idx:bool=False, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        enable_hdf_testing_mode: bool - default False - if True, errors are not thrown for the first field that cannot be serialized, and instead all are attempted to see which ones work.


        Usage:
            import h5py
            
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            with h5py.File(a_save_path, 'w') as f: ## open the path as a HDF5 file handle:
                _pfnd_obj.to_hdf(f, key='test_pfnd')
            
                
        More Error-tolerant usage:                
            import h5py

            a_decoder_name: str = 'long'
            a_full_decoded_2D_posterior_result: SingleEpochDecodedResult = results2D.continuous_results[a_decoder_name].get_result_for_epoch(0)
            a_save_path = Path('data/a_full_decoded_2D_posterior_result.pkl').resolve()
            if not a_save_path.parent.exists():
                print(f'creating "{a_save_path.parent}"...')
                a_save_path.parent.mkdir(exist_ok=True)
            
            with h5py.File(a_save_path, 'w') as f: ## open the path as a HDF5 file handle:
                a_full_decoded_2D_posterior_result.to_hdf(f, 'a_full_decoded_2D_posterior_result')
                print(f'\tsaved "{a_save_path}".')
                
    
        """
        import h5py
        from pyphocorehelpers.plotting.media_output_helpers import img_data_to_greyscale, get_array_as_image
        
        # assert not isinstance(file_path, (str, Path)), f"pass an already open HDF5 file handle. You passed type(file_path): {type(file_path)}, file_path: {file_path}"
        assert not isinstance(file_path, (str, Path)), f""" pass an already open HDF5 file handle. You passed type(file_path): {type(file_path)}, file_path: {file_path}. Example
            import h5py    
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            with h5py.File(a_save_path, 'w') as f: ## open the path as a HDF5 file handle:
                _pfnd_obj.to_hdf(f, key='test_pfnd')
        """
        

        f = file_path
        if debug_print and enable_hdf_testing_mode:
            print(f'type(f): {type(f)}, f: {f}') # type(f): <class 'h5py._hl.files.File'>, f: <HDF5 file "decoded_epoch_posteriors.h5" (mode r+)>
        # super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, **kwargs)
        # handle custom properties here

        if self.epoch_data_index is not None:
            if required_zero_padding is not None:
                epoch_data_idx_str: str = f"{self.epoch_data_index:0{required_zero_padding}d}"    
            else:
                epoch_data_idx_str: str = f"{self.epoch_data_index:0{len(str(self.epoch_data_index))}d}"
        else:
            epoch_data_idx_str = None
        # OUTPUTS: epoch_data_idx_str
        def _subfn_get_key_str(variable_name):
            """ captures: key, epoch_data_idx_str, leafs_include_epoch_idx 
            """
            _temp_epoch_id_identifier_str: str = f'{key}/{variable_name}'
            if (epoch_data_idx_str is not None) and leafs_include_epoch_idx:
                return f"{_temp_epoch_id_identifier_str}[{epoch_data_idx_str}]"
            else:
                return f"{_temp_epoch_id_identifier_str}"
        

        attribute_type_fields = ['nbins', 'epoch_data_index', 'n_xbins']
        dataset_type_fields = ['most_likely_positions', 'most_likely_position_indicies', 'time_bin_edges'] # 'p_x_given_n', 
        container_type_fields = ['time_bin_container', 'marginal_x', 'marginal_y']
        
        # Get current date and time
        current_time = datetime.now().isoformat()

        # 'p_x_given_n':
        # epoch_id_identifier_str: str = f'{key}/p_x_given_n'
        # epoch_id_str = _subfn_get_key_str('p_x_given_n')
        
        # img_data = self.p_x_given_n.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
        p_x_given_n = deepcopy(self.marginal_x.p_x_given_n)

        f.create_dataset(_subfn_get_key_str('p_x_given_n'), data=p_x_given_n.astype(float))

        # p_x_given_n_image = active_captured_single_epoch_result.get_posterior_as_image(skip_img_normalization=False, export_grayscale=True)
        p_x_given_n_image = img_data_to_greyscale(p_x_given_n)
        f.create_dataset(_subfn_get_key_str('p_x_given_n_grey'), data=p_x_given_n_image.astype(float))
        
        # f.attrs['creation_date'] = current_time
        
        ## Dataset type fields:
        for a_field_name in dataset_type_fields:
            a_val = getattr(self, a_field_name)
            if a_val is not None:
                ## valid value
                # img_data = self.p_x_given_n.astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
                f.create_dataset(_subfn_get_key_str(a_field_name), data=a_val) # TypeError: No conversion path for dtype: dtype('<U24')
                # f.attrs['creation_date'] = current_time

        ## Custom derived:
        # 't_bin_centers':
        t_bin_centers = deepcopy(self.time_bin_container.centers)
        f.create_dataset(_subfn_get_key_str('t_bin_centers'), data=t_bin_centers)
        # f.attrs['creation_date'] = current_time
        
        ## Attributes:
        group = f[key]
        group.attrs['creation_date'] = current_time
        group.attrs['nbins'] = self.nbins
        group.attrs['n_xbins'] = self.n_xbins
        # group.attrs['ndim'] = self.ndim

        epoch_info_tuple = deepcopy(self.epoch_info_tuple) # EpochTuple(Index=28, start=971.8437469999772, stop=983.9541530000279, label='28', duration=12.110406000050716, lap_id=29, lap_dir=1, score=0.36769430044232587, velocity=1.6140523749028528, intercept=1805.019565924132, speed=1.6140523749028528, wcorr=-0.9152062701244238, P_decoder=0.6562437078530542, pearsonr=-0.7228173157676305, travel=0.0324318935144031, coverage=0.19298245614035087, jump=0.0005841121495327102, sequential_correlation=16228.563177472019, monotonicity_score=16228.563177472019, laplacian_smoothness=16228.563177472019, longest_sequence_length=22, longest_sequence_length_ratio=0.4583333333333333, direction_change_bin_ratio=0.19148936170212766, congruent_dir_bins_ratio=0.574468085106383, total_congruent_direction_change=257.92556950947574, total_variation=326.1999849678664, integral_second_derivative=7423.7044320722935, stddev_of_diff=8.368982188902695)
        epoch_info_tuple_included_fields = [k for k in dir(epoch_info_tuple) if ((not k.startswith('_')) and (k not in ['index', 'count']))] # ['Index', 'P_decoder', 'congruent_dir_bins_ratio', 'count', 'coverage', 'direction_change_bin_ratio', 'duration', 'index', 'integral_second_derivative', 'intercept', 'jump', 'label', 'lap_dir', 'lap_id', 'laplacian_smoothness', 'longest_sequence_length', 'longest_sequence_length_ratio', 'monotonicity_score', 'pearsonr', 'score', 'sequential_correlation', 'speed', 'start', 'stddev_of_diff', 'stop', 'total_congruent_direction_change', 'total_variation', 'travel', 'velocity', 'wcorr']
        
        issue_fields = ['is_user_annotated_epoch', 'is_valid_epoch']
        for a_field_name in epoch_info_tuple_included_fields:
            a_val = getattr(epoch_info_tuple, a_field_name)
            if (a_val is not None):
                ## valid value
                group.attrs[a_field_name] = a_val


    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "PfND":
        """ Reads the data from the key in the hdf5 file at file_path """
        raise NotImplementedError("read_hdf not implemented")


from typing import Literal
# Define a type that can only be one of these specific strings
MaskedTimeBinFillType = Literal['ignore', 'last_valid', 'nan_filled', 'dropped'] ## used in `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(...)` to specify how invalid bins (due to too few spikes) are treated.


@custom_define(slots=False, repr=False, eq=False)
class DecodedFilterEpochsResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Container for the results of decoding a set of epochs (filter_epochs) using a decoder (active_decoder) 
    
    This class stores results from decoding from multiple non-contiguous time epochs, each containing many time bins (a variable number according to their length)
        The two representational formats are:
            1. 
    
            
            
    WARNING/BUGS/LIMITATIONS: when there's only one bin in epoch, `time_bin_edges` has a drastically wrong number of elements (e.g. len (30, )) while `time_window_centers` is right
        Workaround: can be fixed with the following code:
    
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        time_window_centers = a_result.time_window_centers[an_epoch_idx]
        time_bin_edges = a_result.time_bin_edges[an_epoch_idx] # (30, )
        
        if (n_time_bins == 1) and (len(time_window_centers) == 1):
            ## fix time_bin_edges -- it has been noticed when there's only one bin, `time_bin_edges` has a drastically wrong number of elements (e.g. len (30, )) while `time_window_centers` is right.
            if len(time_bin_edges) != 2:
                ## fix em
                time_bin_container = a_result.time_bin_containers[an_epoch_idx]
                time_bin_edges = np.array(list(time_bin_container.center_info.variable_extents))
                assert len(time_bin_edges) == 2, f"tried to fix but FAILED!"
                # print(f'fixed time_bin_edges: {time_bin_edges}')

                
            
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        
        
    marginal_y:
        DynamicContainer({'p_x_given_n': array([[0, 0, 0.722646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.722646, 0.722646, 0.722646, 0.722646, 0, 0, 0.722646, 0.722646, 0.722646, 0, 0, 0, 0, 0.722646, 0, 0, 0, 0],
        [1, 1, 0.277354, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.277354, 0.277354, 0.277354, 0.277354, 1, 1, 0.277354, 0.277354, 0.277354, 1, 1, 1, 1, 0.277354, 1, 1, 1, 1]]), 'most_likely_positions_1D': array([1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5])})
       
    """
    decoding_time_bin_size: float = serialized_attribute_field() # the time bin_size in seconds
    filter_epochs: pd.DataFrame = serialized_field() # the filter epochs themselves
    num_filter_epochs: int = serialized_attribute_field() # depends on the number of epochs (`n_epochs`)
    most_likely_positions_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    p_x_given_n_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    marginal_x_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    marginal_y_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    marginal_z_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    most_likely_position_indicies_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    spkcount: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    nbins: np.ndarray = serialized_field(metadata={'shape': ('n_epochs',)}) # an array of the number of time bins in each epoch
    time_bin_containers: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    time_bin_edges: list = non_serialized_field(metadata={'shape': ('n_epochs',)}) # depends on the number of epochs, one per epoch
    epoch_description_list: list[str] = non_serialized_field(default=Factory(list), metadata={'shape': ('n_epochs',)}) # depends on the number of epochs, one for each
    
    ## Optional Helpers
    pos_bin_edges: Optional[NDArray] = serialized_field(default=None, metadata={'desc':'the position bin edges of the decoder that was used to produce the result', 'shape': ('n_pos_bins+1', )})

    @property
    def n_pos_bins(self) -> Optional[int]:
        """The n_pos_bins property."""
        if self.pos_bin_edges is not None:
            return len(self.pos_bin_edges)-1
        else:
            return None

    @property
    def active_filter_epochs(self):
        """ for compatibility """
        return deepcopy(self.filter_epochs)


    @property
    def time_window_centers(self) -> List[NDArray]:
        """ for compatibility """
        return deepcopy([self.time_bin_containers[an_epoch_idx].centers for an_epoch_idx in np.arange(self.num_filter_epochs)])

    @property
    def flat_time_window_centers(self) -> NDArray:
        """ for compatibility """
        return np.hstack(self.time_window_centers) ## a flat list of time_window_centers


    @function_attributes(short_name=None, tags=['single-epoch', 'indexing', 'start-time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=['self.get_result_for_epoch_at_time'])
    def get_result_for_epoch(self, active_epoch_idx: int) -> SingleEpochDecodedResult:
        """ returns a container with the result from a single epoch. 
        NOTE: active_epoch_idx is the "dumb-index" not a smarter epoch id or label or anything like that. See self.get_result_for_epoch_at_time(...) for smarter access.
        
        """
        if active_epoch_idx >= len(self.active_filter_epochs):
            raise ValueError(f"Requested epoch {active_epoch_idx}, but only {len(self.active_filter_epochs)} available.")

        single_epoch_field_names = ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'marginal_z_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges'] # a_decoder_decoded_epochs_result._test_find_fields_by_shape_metadata()
        fields_to_single_epoch_fields_dict = dict(zip(['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'marginal_z_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges'],
            ['most_likely_positions', 'p_x_given_n', 'marginal_x', 'marginal_y', 'marginal_z', 'most_likely_position_indicies', 'nbins', 'time_bin_container', 'time_bin_edges'])) # maps list names to single-epoch specific field names
        
        for field_name in single_epoch_field_names:
            if (not hasattr(self, field_name)) or (getattr(self, field_name, None) is None):
                print(f'WARNING: missing field_name: {field_name}. Setting and continuing.')
                if field_name in ['marginal_z_list']: 
                    setattr(self, field_name, [None for i in np.arange(self.num_filter_epochs)])
                else:                  
                    setattr(self, field_name, [None for i in np.arange(self.num_filter_epochs)])
                
            valid_attr_value = getattr(self, field_name)            
            # print(f'len(valid_attr_value): {len(valid_attr_value)}')
            assert active_epoch_idx < len(valid_attr_value), f"for field_name: {field_name}\n\tactive_epoch_idx: {active_epoch_idx} len(valid_attr_value): {len(valid_attr_value)}"
            
        # END for field_name in single_epoch_field_names...


        values_dict = {fields_to_single_epoch_fields_dict[field_name]:getattr(self, field_name)[active_epoch_idx] for field_name in single_epoch_field_names}
        # a_posterior = self.p_x_given_n_list[active_epoch_idx].copy()
        active_epoch_info_tuple = tuple(ensure_dataframe(self.active_filter_epochs).itertuples(name='EpochTuple'))[active_epoch_idx] # just dumb-indexes into the epochs array
        single_epoch_result: SingleEpochDecodedResult = SingleEpochDecodedResult(**values_dict, epoch_info_tuple=active_epoch_info_tuple, epoch_data_index=active_epoch_idx)
        return single_epoch_result
    
    @function_attributes(short_name=None, tags=['single-epoch', 'indexing', 'start-time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-18 07:36', related_items=['self.get_result_for_epoch'])
    def get_result_for_epoch_at_time(self, epoch_start_time: float) -> SingleEpochDecodedResult:
        """ returns a container with the result from a single epoch, based on a start time
        
        
        .get_result_for_epoch_at_time(epoch_start_time=clicked_epoch[0])
        
        """
        filtered_v = self.filtered_by_epoch_times(included_epoch_start_times=np.atleast_1d(epoch_start_time)) # should only have 1 epoch remaining
        assert len(filtered_v.filter_epochs) == 1, f"len(filtered_v.filter_epochs): {len(filtered_v.filter_epochs)}"
        return filtered_v.get_result_for_epoch(active_epoch_idx=0) # 0 to indicate the first (and only) remaining epoch
        
    
    
    

    @function_attributes(short_name=None, tags=['radon-transform','decoder','line','fit','velocity','speed'], input_requires=[], output_provides=[], uses=['get_radon_transform'], used_by=[], creation_date='2024-02-13 17:25', related_items=[])
    def compute_radon_transforms(self, pos_bin_size:float, xbin_centers: NDArray, nlines:int=8192, margin:float=8, jump_stat=None, n_jobs:int=4, enable_return_neighbors_arr=False) -> pd.DataFrame:
        """ 2023-05-25 - Computes the line of best fit (which gives the velocity) for the 1D Posteriors for each replay epoch using the Radon Transform approch.
        
        # pos_bin_size: the size of the x_bin in [cm]
        if decoder.pf.bin_info is not None:
            pos_bin_size = float(decoder.pf.bin_info['xstep'])
        else:
            ## if the bin_info is for some reason not accessible, just average the distance between the bin centers.
            pos_bin_size = np.diff(decoder.pf.xbin_centers).mean()


        Usage:
        
            epochs_linear_fit_df = compute_radon_transforms(pos_bin_size=pos_bin_size, nlines=8192, margin=16, n_jobs=1)
            
            a_directional_laps_filter_epochs_decoder_result = a_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
            laps_radon_transform_df = compute_radon_transforms(a_directional_pf1D_Decoder, a_directional_laps_filter_epochs_decoder_result)

            Columns:         ['score', 'velocity', 'intercept', 'speed']
        """
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import get_radon_transform
        # active_time_bins = active_epoch_decoder_result.time_bin_edges[0]
        # active_posterior_container = active_epoch_decoder_result.marginal_x_list[0]
        active_posterior = self.p_x_given_n_list # one for each epoch
        active_time_window_centers_list = [t[0] for t in self.time_window_centers] # first index
        ## compute the Radon transform to get the lines of best fit
        extra_outputs = []
        score, velocity, intercept, *extra_outputs = get_radon_transform(active_posterior, decoding_time_bin_duration=self.decoding_time_bin_size, pos_bin_size=pos_bin_size, posteriors=None, nlines=nlines, margin=margin, jump_stat=jump_stat, n_jobs=n_jobs, enable_return_neighbors_arr=enable_return_neighbors_arr, debug_print=True,
                                                                          x0=xbin_centers[0], t0=active_time_window_centers_list)
        epochs_linear_fit_df = pd.DataFrame({'score': score, 'velocity': velocity, 'intercept': intercept, 'speed': np.abs(velocity)})
        return epochs_linear_fit_df, extra_outputs


    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
            DecodedFilterEpochsResult(decoding_time_bin_size: float,
                filter_epochs: neuropy.core.epoch.Epoch,
                num_filter_epochs: int,
                most_likely_positions_list: list | shape (n_epochs),
                p_x_given_n_list: list | shape (n_epochs),
                marginal_x_list: list | shape (n_epochs),
                marginal_y_list: list | shape (n_epochs),
                most_likely_position_indicies_list: list | shape (n_epochs),
                spkcount: list | shape (n_epochs),
                nbins: numpy.ndarray | shape (n_epochs),
                time_bin_containers: list | shape (n_epochs),
                time_bin_edges: list | shape (n_epochs),
                epoch_description_list: list | shape (n_epochs)
            )
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        # content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # return f"{type(self).__name__}({content}\n)"
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"


    # For serialization/pickling: ________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes (_mapping and _keys_at_init). Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        if 'pos_bin_edges' not in state:
            self.__dict__.update(pos_bin_edges=None) ## default to None (for same unpickling)

    def flatten(self):
        """ flattens the result over all epochs to produce one per time bin 
        Usage:
            n_timebins, flat_time_bin_containers, timebins_p_x_given_n = laps_pseudo2D_continuous_specific_decoded_result.flatten()
        
        """
        # returns a flattened version of self over all epochs
        n_timebins = np.sum(self.nbins)
        flat_time_bin_containers = np.hstack(self.time_bin_containers)
        # timebins_p_x_given_n = [].extend(self.p_x_given_n_list)
        # timebins_p_x_given_n = np.hstack(self.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
        # Determine posterior shape
        a_posterior = self.p_x_given_n_list[0]
        posterior_dims = len(np.shape(a_posterior))
        
        if posterior_dims == 2:
            timebins_p_x_given_n = np.hstack(self.p_x_given_n_list)
        elif posterior_dims == 3:
            timebins_p_x_given_n = np.concatenate(self.p_x_given_n_list, axis=-1)
        else:
            raise ValueError("Unsupported posterior shape: {}".format(np.shape(a_posterior)))
        
        # TODO 2023-04-13 -can these squished similar way?: most_likely_positions_list, most_likely_position_indicies_list 
        return n_timebins, flat_time_bin_containers, timebins_p_x_given_n # flat_time_bin_containers: seems to be an NDArray of centers or something instead of containers


    def flatten_to_masked_values(self):
        """ appends np.nan values to the beginning and end of each posterior (adding a start and end timebin as well) to allow flat plotting via matplotlib.
        Looks  like it was depricated by `plot_slices_1D_most_likely_position_comparsions` to plot epoch slices (corresponding to certain periods in time) along the continuous session duration.
        Usage:
            desired_total_n_timebins, updated_is_masked_bin, updated_time_bin_containers, updated_timebins_p_x_given_n = laps_pseudo2D_continuous_specific_decoded_result.flatten_to_masked_values()
        """
        # returns a flattened version of self over all epochs
        updated_is_masked_bin = []
        updated_time_bin_containers = []
        updated_timebins_p_x_given_n = []

        decoding_time_bin_size: float = self.decoding_time_bin_size
        desired_n_timebins = self.nbins + 2 # add two to each element for the start/end bin

        total_n_timebins = np.sum(self.nbins)
        desired_total_n_timebins = np.sum(desired_n_timebins)


        for epoch_idx in np.arange(self.num_filter_epochs):
            a_curr_num_bins: int = self.nbins[epoch_idx]
            updated_curr_num_bins = a_curr_num_bins + 2 # add two (start/end) bins
            a_centers = self.time_bin_containers[epoch_idx].centers
            a_posterior = self.p_x_given_n_list[epoch_idx]
            a_posterior_shape = np.shape(a_posterior)
            n_pos_bins = a_posterior_shape[0]
            posterior_dims = len(a_posterior_shape)
        
            if posterior_dims == 2:
                # 1D (x-only) posterior per time bin
                updated_posterior = np.full((n_pos_bins, updated_curr_num_bins), np.nan)
            elif posterior_dims == 3:
                # 2D (x & y) posterior per time bin
                n_y_pos_bins = a_posterior_shape[1]
                updated_posterior = np.full((n_pos_bins, n_y_pos_bins, updated_curr_num_bins), np.nan)
            else:
                raise ValueError("Unsupported posterior shape: {}".format(a_posterior_shape))
            
            if posterior_dims == 2:
                updated_posterior[:, 1:-1] = a_posterior
            elif posterior_dims == 3:
                updated_posterior[:, :, 1:-1] = a_posterior
            

            # updated_posterior = np.full((n_pos_bins, updated_curr_num_bins), np.nan)
            # updated_posterior[:,1:-1] = a_posterior ## does not work for 2D posteriors - ValueError: could not broadcast input array from shape (76,40,17) into shape (76,17)

            curr_is_masked_bin = np.full((updated_curr_num_bins,), True)
            curr_is_masked_bin[1:-1] = False

            ## Add the start/end bin
            # a_centers.
            updated_time_bin_containers.append([(a_centers[0]-decoding_time_bin_size), list(a_centers), (a_centers[-1]+decoding_time_bin_size)])
            updated_timebins_p_x_given_n.append(updated_posterior)
            updated_is_masked_bin.append(curr_is_masked_bin)

        updated_timebins_p_x_given_n = np.hstack(updated_timebins_p_x_given_n) if posterior_dims == 2 else np.concatenate(updated_timebins_p_x_given_n, axis=-1)
        # updated_timebins_p_x_given_n = np.hstack(updated_timebins_p_x_given_n) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
        updated_time_bin_containers = np.hstack(np.hstack(updated_time_bin_containers))
        updated_is_masked_bin = np.hstack(updated_is_masked_bin)

        assert np.shape(updated_time_bin_containers)[0] == desired_total_n_timebins
        # assert np.shape(updated_timebins_p_x_given_n)[1] == desired_total_n_timebins
        assert np.shape(updated_timebins_p_x_given_n)[-1] == desired_total_n_timebins
        assert np.shape(updated_is_masked_bin)[0] == desired_total_n_timebins

        return desired_total_n_timebins, updated_is_masked_bin, updated_time_bin_containers, updated_timebins_p_x_given_n


    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray) -> NDArray:
        subset = deepcopy(self)
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        return subset.filter_epochs.epochs.find_data_indicies_from_epoch_times(epoch_times=epoch_times)

    def find_epoch_times_to_data_indicies_map(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
        """ returns the a Dict[Union[float, Tuple[float, float]], Union[int, NDArray]] matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            self.find_epoch_times_to_data_indicies_map
        
        """
        subset = deepcopy(self)
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        return subset.filter_epochs.epochs.find_epoch_times_to_data_indicies_map(epoch_times=epoch_times, atol=atol, t_column_names=t_column_names)
            

    def filtered_by_epoch_times(self, included_epoch_start_times) -> "DecodedFilterEpochsResult":
        """ Returns a copy of itself with the fields with the n_epochs related metadata sliced by the included_epoch_indicies found from the rows that match `included_epoch_start_times`.       
        """
        subset = deepcopy(self)
        original_num_filter_epochs = subset.num_filter_epochs
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        found_data_indicies = subset.filter_epochs.epochs.find_data_indicies_from_epoch_times(epoch_times=included_epoch_start_times)
        return subset.filtered_by_epochs(found_data_indicies)


    def filtered_by_epochs(self, included_epoch_indicies, debug_print=False) -> "DecodedFilterEpochsResult":
        """Returns a copy of itself with the fields with the n_epochs related metadata sliced by the included_epoch_indicies.
        
        MOSTLY TESTED 2024-03-11 - WORKING

        """
        subset = deepcopy(self)
        original_num_filter_epochs = subset.num_filter_epochs
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()

        ## Convert to the real-deal: pure indicies
        old_fashioned_indicies = np.array([subset.filter_epochs.index.get_loc(a_loc_idx) for a_loc_idx in included_epoch_indicies])
        if debug_print:
            print(f'old_fashioned_indicies: {old_fashioned_indicies}')

        ## Need to have the indicies before applying the filter:
        subset.filter_epochs = subset.filter_epochs.loc[included_epoch_indicies] # the evil `.iloc[...]` creeps in again with `IndexError: positional indexers are out-of-bounds`
        subset.most_likely_positions_list = [subset.most_likely_positions_list[i] for i in old_fashioned_indicies] # that's obviously not going to work because .loc[...] values are used. I need that magic trick that the new AI taught me -- `.index.get_loc(start_index)`
        subset.p_x_given_n_list = [subset.p_x_given_n_list[i] for i in old_fashioned_indicies]
        subset.marginal_x_list = [subset.marginal_x_list[i] for i in old_fashioned_indicies]
        subset.marginal_y_list = [subset.marginal_y_list[i] for i in old_fashioned_indicies]
        subset.marginal_z_list = [subset.marginal_z_list[i] for i in old_fashioned_indicies]
        subset.most_likely_position_indicies_list = [subset.most_likely_position_indicies_list[i] for i in old_fashioned_indicies]
        subset.spkcount = [subset.spkcount[i] for i in old_fashioned_indicies]
        if len(old_fashioned_indicies) > 0:
            subset.nbins = subset.nbins[old_fashioned_indicies] # can be subset because it's an ndarray. `IndexError: arrays used as indices must be of integer (or boolean) type`: occurs because when it is empty it seems to default to float64 dtype
        else:    
            subset.nbins = [] # empty list
        
        subset.time_bin_containers = [subset.time_bin_containers[i] for i in old_fashioned_indicies]
        subset.num_filter_epochs = len(included_epoch_indicies)
        subset.time_bin_edges = [subset.time_bin_edges[i] for i in old_fashioned_indicies]
        if len(subset.epoch_description_list) == original_num_filter_epochs:
            # sometimes epoch_description_list is empty and so it doesn't need to be subsetted.
            subset.epoch_description_list = [subset.epoch_description_list[i] for i in old_fashioned_indicies]
            
        # Only `decoding_time_bin_size` is unchanged
        return subset
    

    @function_attributes(short_name=None, tags=['validate', 'time_bins', 'debug'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-07 16:35', related_items=[])
    def validate_time_bins(self):
        """validates a `DecodedFilterEpochsResult` object -- ensuring that all lists are of consistent sizes and that within the lists all the time bins have the correct properties
        """
        def _subfn_check_all_epoch_lists_same_length(a_train_decoded_results):
            single_epoch_field_names = ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'marginal_z_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges'] # DecodedFilterEpochsResult._test_find_fields_by_shape_metadata(desired_keys_subset=desired_keys_subset)
            list_field_length_dict: Dict[str, int] = {field_name:len(getattr(a_train_decoded_results, field_name)) for field_name in single_epoch_field_names}
            assert np.allclose(np.array(list(list_field_length_dict.values())), a_train_decoded_results.num_filter_epochs), f"np.array(list(list_field_length_dict.values())): {np.array(list(list_field_length_dict.values()))} differs from a_train_decoded_results.num_filter_epochs: {a_train_decoded_results.num_filter_epochs}"

        # checks all list fields are the same length as a_train_decoded_results.num_filter_epochs
        _subfn_check_all_epoch_lists_same_length(a_train_decoded_results=self)

        for an_epoch_idx in np.arange(self.num_filter_epochs):
            epoch_lbl_str: str = f"Epoch[{an_epoch_idx}]: "
            a_most_likely_positions_list = self.most_likely_positions_list[an_epoch_idx]
            a_p_x_given_n = self.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
            a_n_bins_n_time_bins: int = self.nbins[an_epoch_idx]
            # a_most_likely_positions_list_n_time_bins, a_most_likely_positions_list_n_pos_bins = np.shape(a_most_likely_positions_list)
            a_most_likely_positions_list_n_time_bins: int = np.shape(a_most_likely_positions_list)[0]
            

            ## Everything is valid only for 1D, don't check 2D at all:
            # assert np.ndim(a_p_x_given_n) == 2, epoch_lbl_str + f"np.ndim(a_p_x_given_n): {np.ndim(a_p_x_given_n)}"
            
            if np.ndim(a_p_x_given_n) == 2:            
                ## 1D position:
                n_pos_bins_posterior, n_time_bins_posterior = np.shape(a_p_x_given_n) # np.shape(a_p_x_given_n): (62, 9)
            elif np.ndim(a_p_x_given_n) == 3:
                # 2D position
                n_pos_x_bins_posterior, n_pos_y_bins_posterior, n_time_bins_posterior = np.shape(a_p_x_given_n) # np.shape(a_p_x_given_n): (62, 9)
                # np.shape(self.p_x_given_n_list[an_epoch_idx-1])
                # (57, 4, 3)
                # np.shape(self.p_x_given_n_list[an_epoch_idx-2])
                # (57, 4, 5)
                # np.shape(self.p_x_given_n_list[an_epoch_idx-0])
                # (57, 1, 16)
                # np.shape(self.p_x_given_n_list[an_epoch_idx-2])
                # (57, 4, 5)
                # np.shape(self.p_x_given_n_list[an_epoch_idx-9])
                # (57, 4, 2)


            else:
                raise NotImplementedError(f'Unexpected number of dimensions in posteriors! {epoch_lbl_str} np.ndim(a_p_x_given_n): {np.ndim(a_p_x_given_n)}"')
            
            assert a_n_bins_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_n_bins_n_time_bins: {a_n_bins_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}\n\ta_train_decoded_results.nbins: {self.nbins}"
            
            ## Check position bin agreement:
            assert a_most_likely_positions_list_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_most_likely_positions_list_n_time_bins: {a_most_likely_positions_list_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}"
            # assert a_most_likely_positions_list_n_pos_bins == n_pos_bins, f"a_most_likely_positions_list_n_pos_bins: {a_most_likely_positions_list_n_pos_bins} != n_pos_bins: {n_pos_bins}"

            # time_window_centers = a_train_decoded_results.time_bin_containers[an_epoch_idx].centers
            time_window_centers = self.time_window_centers[an_epoch_idx]
            a_time_window_centers_n_time_bins = len(time_window_centers)
            assert a_time_window_centers_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_time_window_centers_n_time_bins: {a_time_window_centers_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}"
            
            a_time_bin_container = self.time_bin_containers[an_epoch_idx]
            a_time_bin_container_n_time_bins = a_time_bin_container.num_bins
            assert a_time_bin_container_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_time_bin_container_n_time_bins: {a_time_bin_container_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}"
            
            # ## `time_bin_edges` are where the problems come from:
            # # AssertionError: a_time_bin_edges_n_time_bins: 282 != n_time_bins_posterior: 1
            # # [678.314 678.315 678.316 678.317 678.318 678.319 678.32 678.321 678.322 678.323 678.324 678.325 678.326 678.327 678.328 678.329 678.33 678.331 678.332 678.333 678.334 678.335 678.336 678.337 678.338 678.339 678.34 678.341 678.342 678.343 678.344 678.345 678.346 678.347 678.348 678.349 678.35 678.351 678.352 678.353 678.354 678.355 678.356 678.357 678.358 678.359 678.36 678.361 678.362 678.363 678.364 678.365 678.366 678.367 678.368 678.369 678.37 678.371 678.372 678.373 678.374 678.375 678.376 678.377 678.378 678.379 678.38 678.381 678.382 678.383 678.384 678.385 678.386 678.387 678.388 678.389 678.39 678.391 678.392 678.393 678.394 678.395 678.396 678.397 678.398 678.399 678.4 678.401 678.402 678.403 678.404 678.405 678.406 678.407 678.408 678.409 678.41 678.411 678.412 678.413 678.414 678.415 678.416 678.417 678.418 678.419 678.42 678.421 678.422 678.423 678.424 678.425 678.426 678.427 678.428 678.429 678.43 678.431 678.432 678.433 678.434 678.435 678.436 678.437 678.438 678.439 678.44 678.441 678.442 678.443 678.444 678.445 678.446 678.447 678.448 678.449 678.45 678.451 678.452 678.453 678.454 678.455 678.456 678.457 678.458 678.459 678.46 678.461 678.462 678.463 678.464 678.465 678.466 678.467 678.468 678.469 678.47 678.471 678.472 678.473 678.474 678.475 678.476 678.477 678.478 678.479 678.48 678.481 678.482 678.483 678.484 678.485 678.486 678.487 678.488 678.489 678.49 678.491 678.492 678.493 678.494 678.495 678.496 678.497 678.498 678.499 678.5 678.501 678.502 678.503 678.504 678.505 678.506 678.507 678.508 678.509 678.51 678.511 678.512 678.513 678.514 678.515 678.516 678.517 678.518 678.519 678.52 678.521 678.522 678.523 678.524 678.525 678.526 678.527 678.528 678.529 678.53 678.531 678.532 678.533 678.534 678.535 678.536 678.537 678.538 678.539 678.54 678.541 678.542 678.543 678.544 678.545 678.546 678.547 678.548 678.549 678.55 678.551 678.552 678.553 678.554 678.555 678.556 678.557 678.558 678.559 678.56 678.561 678.562 678.563 678.564 678.565 678.566 678.567 678.568 678.569 678.57 678.571 678.572 678.573 678.574 678.575 678.576 678.577 678.578 678.579 678.58 678.581 678.582 678.583 678.584 678.585 678.586 678.587 678.588 678.589 678.59 678.591 678.592 678.593 678.594 678.595 678.596]
            # a_time_bin_edges = self.time_bin_edges[an_epoch_idx]
            # a_time_bin_edges_n_time_bins = len(a_time_bin_edges)-1
            # assert a_time_bin_edges_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_time_bin_edges_n_time_bins: {a_time_bin_edges_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}\n\t {a_time_bin_edges}"
            
            # a_time_bin_container-based time_window_centers
            a_time_bin_container_time_window_centers = a_time_bin_container.centers
            a_time_bin_container_time_window_centers_n_time_bins = len(a_time_bin_container_time_window_centers)
            assert a_time_bin_container_time_window_centers_n_time_bins == n_time_bins_posterior, epoch_lbl_str + f"a_time_bin_container_time_window_centers_n_time_bins: {a_time_bin_container_time_window_centers_n_time_bins} != n_time_bins_posterior: {n_time_bins_posterior}"

            # time_window_centers = a_train_decoded_results.time_window_centers
            # a_time_window_centers_n_time_bins = len(time_window_centers)
            # assert a_time_window_centers_n_time_bins == n_time_bins, f"a_time_window_centers_n_time_bins: {a_time_window_centers_n_time_bins} != n_time_bins_posterior: {n_time_bins}"


    @function_attributes(short_name=None, tags=['marginal', 'direction', 'track_id', 'NEEDS_GENERALIZATION'], input_requires=[], output_provides=[], uses=['DirectionalPseudo2DDecodersResult.determine_directional_likelihoods', 'DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods', 'DirectionalPseudo2DDecodersResult.determine_non_marginalized_decoder_likelihoods'], used_by=['self.compute_marginals'], creation_date='2024-10-08 00:40', related_items=[]) 
    @classmethod
    def perform_compute_marginals(cls, filter_epochs_decoder_result: Union[List[NDArray], List[DynamicContainer], NDArray, "DecodedFilterEpochsResult"], filter_epochs: pd.DataFrame, epoch_idx_col_name: str = 'lap_idx', epoch_start_t_col_name: str = 'lap_start_t', additional_transfer_column_names: Optional[List[str]]=None, auto_transfer_all_columns:bool=True): # -> tuple[tuple["DecodedMarginalResultTuple", "DecodedMarginalResultTuple", Tuple[List[DynamicContainer], Any, Any, pd.DataFrame]], pd.DataFrame]:
        """Computes and initializes the marginal properties
        
        (epochs_directional_marginals_tuple, epochs_track_identity_marginals_tuple, epochs_non_marginalized_decoder_marginals_tuple), epochs_marginals_df = a_result.compute_marginals(additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir'])
        epochs_directional_marginals, epochs_directional_all_epoch_bins_marginal, epochs_most_likely_direction_from_decoder, epochs_is_most_likely_direction_LR_dir  = epochs_directional_marginals_tuple
        epochs_track_identity_marginals, epochs_track_identity_all_epoch_bins_marginal, epochs_most_likely_track_identity_from_decoder, epochs_is_most_likely_track_identity_Long = epochs_track_identity_marginals_tuple
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = epochs_non_marginalized_decoder_marginals_tuple
        
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
        

        ## Make sure it is the Pseudo2D (all-directional) decoder by checking shape:
        p_x_given_n_list = DirectionalPseudo2DDecodersResult.get_proper_p_x_given_n_list(filter_epochs_decoder_result)
        p_x_given_n_list_second_dim_sizes = np.array([np.shape(a_p_x_given_n)[1] for a_p_x_given_n in p_x_given_n_list])
        is_pseudo2D_decoder = np.all((p_x_given_n_list_second_dim_sizes == 4)) # only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        

        epochs_df: pd.DataFrame = ensure_dataframe(deepcopy(filter_epochs))
        

        if is_pseudo2D_decoder:
            epochs_directional_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(filter_epochs_decoder_result)
            epochs_directional_marginals, epochs_directional_all_epoch_bins_marginal, epochs_most_likely_direction_from_decoder, epochs_is_most_likely_direction_LR_dir  = epochs_directional_marginals_tuple
            epochs_track_identity_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods(filter_epochs_decoder_result)
            epochs_track_identity_marginals, epochs_track_identity_all_epoch_bins_marginal, epochs_most_likely_track_identity_from_decoder, epochs_is_most_likely_track_identity_Long = epochs_track_identity_marginals_tuple
            epochs_non_marginalized_decoder_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_non_marginalized_decoder_likelihoods(filter_epochs_decoder_result, debug_print=False)
            non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = epochs_non_marginalized_decoder_marginals_tuple
        else:
            is_track_identity_only_pseudo2D_decoder: bool = np.all((p_x_given_n_list_second_dim_sizes == 2)) # only works with the Pseudo2D (all-directional) decoder with posteriors with .shape[1] == 4, corresponding to ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            assert is_track_identity_only_pseudo2D_decoder
            raise NotImplementedError(f'2025-03-26 14:06 Not finished, do the other method instead (see git commit)')
            # epochs_directional_marginals_tuple = (None, None)
            # epochs_directional_marginals, epochs_directional_all_epoch_bins_marginal, epochs_most_likely_direction_from_decoder, epochs_is_most_likely_direction_LR_dir  = epochs_directional_marginals_tuple
            # epochs_track_identity_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_long_short_likelihoods(filter_epochs_decoder_result)
            # epochs_track_identity_marginals, epochs_track_identity_all_epoch_bins_marginal, epochs_most_likely_track_identity_from_decoder, epochs_is_most_likely_track_identity_Long = epochs_track_identity_marginals_tuple
            # epochs_non_marginalized_decoder_marginals_tuple = DirectionalPseudo2DDecodersResult.determine_non_marginalized_decoder_likelihoods(filter_epochs_decoder_result, debug_print=False)
            # non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = epochs_non_marginalized_decoder_marginals_tuple
            

        ## Build combined marginals df:
        # epochs_marginals_df = pd.DataFrame(np.hstack((epochs_directional_all_epoch_bins_marginal, epochs_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        epochs_marginals_df = pd.DataFrame(np.hstack((non_marginalized_decoder_all_epoch_bins_marginal, epochs_directional_all_epoch_bins_marginal, epochs_track_identity_all_epoch_bins_marginal)), columns=['long_LR', 'long_RL', 'short_LR', 'short_RL', 'P_LR', 'P_RL', 'P_Long', 'P_Short'])
        epochs_marginals_df[epoch_idx_col_name] = epochs_marginals_df.index.to_numpy()
        epochs_marginals_df[epoch_start_t_col_name] = epochs_df['start'].to_numpy()
        
        ## ensure we have the generic columns too (duplicated):
        if (epoch_idx_col_name != 'epoch_idx') and ('epoch_idx' not in epochs_marginals_df):
            epochs_marginals_df['epoch_idx'] = epochs_marginals_df[epoch_idx_col_name]
        if (epoch_start_t_col_name != 'epoch_start_t') and ('epoch_start_t' not in epochs_marginals_df):
            epochs_marginals_df['epoch_start_t'] = epochs_marginals_df[epoch_start_t_col_name]

        # epochs_marginals_df['stop'] = epochs_epochs_df['stop'].to_numpy()
        # epochs_marginals_df['label'] = epochs_epochs_df['label'].to_numpy()
        if auto_transfer_all_columns and (additional_transfer_column_names is None):
            ## if no columns are explcitly specified, transfer all columns
            additional_transfer_column_names = list(epochs_df.columns)
            
        if additional_transfer_column_names is not None:
            for a_col_name in additional_transfer_column_names:
                if ((a_col_name in epochs_df) and (a_col_name not in epochs_marginals_df)):
                    epochs_marginals_df[a_col_name] = epochs_df[a_col_name].to_numpy()
                else:
                    print(f'WARN: extra column: "{a_col_name}" was specified but not present in epochs_epochs_df. Skipping.')
        
        
        return (epochs_directional_marginals_tuple, epochs_track_identity_marginals_tuple, epochs_non_marginalized_decoder_marginals_tuple), epochs_marginals_df
        


    @function_attributes(short_name=None, tags=['marginal', 'direction', 'track_id', 'NEEDS_GENERALIZATION'], input_requires=[], output_provides=[], uses=['cls.perform_compute_marginals'], used_by=[], creation_date='2024-10-08 00:40', related_items=[]) 
    def compute_marginals(self, epoch_idx_col_name: str = 'lap_idx', epoch_start_t_col_name: str = 'lap_start_t', additional_transfer_column_names: Optional[List[str]]=None, auto_transfer_all_columns:bool=True): # -> tuple[tuple["DecodedMarginalResultTuple", "DecodedMarginalResultTuple", Tuple[List[DynamicContainer], Any, Any, pd.DataFrame]], pd.DataFrame]:
        """Computes and initializes the marginal properties
        
        (epochs_directional_marginals_tuple, epochs_track_identity_marginals_tuple, epochs_non_marginalized_decoder_marginals_tuple), epochs_marginals_df = a_result.compute_marginals(additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir'])
        epochs_directional_marginals, epochs_directional_all_epoch_bins_marginal, epochs_most_likely_direction_from_decoder, epochs_is_most_likely_direction_LR_dir  = epochs_directional_marginals_tuple
        epochs_track_identity_marginals, epochs_track_identity_all_epoch_bins_marginal, epochs_most_likely_track_identity_from_decoder, epochs_is_most_likely_track_identity_Long = epochs_track_identity_marginals_tuple
        non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = epochs_non_marginalized_decoder_marginals_tuple
        
        
        """
        epochs_epochs_df: pd.DataFrame = ensure_dataframe(deepcopy(self.filter_epochs))
        return self.perform_compute_marginals(filter_epochs_decoder_result=self, filter_epochs=epochs_epochs_df, epoch_idx_col_name=epoch_idx_col_name, epoch_start_t_col_name=epoch_start_t_col_name, additional_transfer_column_names=additional_transfer_column_names, auto_transfer_all_columns=auto_transfer_all_columns)

        

    @function_attributes(short_name=None, tags=['per_time_bin', 'marginals', 'IMPORTANT'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-09 07:06', related_items=[])
    def build_per_time_bin_marginals_df(self, active_marginals_tuple: Tuple, columns_tuple: Tuple, transfer_column_names_list: Optional[List[str]]=None) -> pd.DataFrame:
        """ Used to build the `per_time_bin` outputs (as opposed to the `per_epoch` outputs)
        
        active_marginals=ripple_track_identity_marginals, columns=['P_LR', 'P_RL']
        active_marginals=ripple_track_identity_marginals, columns=['P_Long', 'P_Short']
        
        _build_multiple_per_time_bin_marginals(a_decoder_result=a_decoder_result, active_marginals_tuple=(laps_directional_all_epoch_bins_marginal, laps_track_identity_all_epoch_bins_marginal), columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short']))
        
        transfer_column_names_list: List[str] = ['maze_id', 'lap_dir', 'lap_id']
                
        Creates Columns: ['center_t', ]
        Creates new df with columns: ['epoch_idx', 't_bin_center', 'epoch_idx']
        
        History: Extracted from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalPseudo2DDecodersResult._build_multiple_per_time_bin_marginals` on 2024-10-09 
        
        History #TODO 2025-05-02 18:53: - [ ] renamed ['epoch_df_idx', ] to ['parent_epoch_df_idx', ]
        """
        parent_epoch_label_col_name: str = 'label' # column in `filter_epochs_df` that can be used to index into the epochs
        filter_epochs_df = deepcopy(self.filter_epochs)
        if not isinstance(filter_epochs_df, pd.DataFrame):
            filter_epochs_df = filter_epochs_df.to_dataframe()
            
        filter_epochs_df['center_t'] = (filter_epochs_df['start'] + (filter_epochs_df['duration']/2.0))
        
        # self.num_filter_epochs: 664
        flat_time_bin_centers_column = np.concatenate([curr_epoch_time_bin_container.centers for curr_epoch_time_bin_container in self.time_bin_containers]) # 4343
        half_time_bin_size: float = self.time_bin_containers[0].edge_info.step / 2.0
        flat_time_bin_start_edge_column = flat_time_bin_centers_column - half_time_bin_size
        flat_time_bin_stop_edge_column = flat_time_bin_centers_column + half_time_bin_size

        all_columns = []
        all_epoch_extracted_posteriors = []
        assert len(active_marginals_tuple) == len(columns_tuple)
        
        # _common_epoch_df_column_dict = {}
        _common_epoch_df_column_dict = None
        for marginal_tuple_idx, active_marginals, active_columns in zip(np.arange(len(active_marginals_tuple)), active_marginals_tuple, columns_tuple):
            epoch_extracted_posteriors = [a_result['p_x_given_n'] for a_result in active_marginals]
            #TODO 2025-08-13 17:28: - [ ] GOAL - Add the 'most_likely_position_1D' for each time bin:
            epoch_extracted_most_likely_positions_1D = None
            # if (marginal_tuple_idx == (len(active_marginals_tuple)-1)):
            ## only for last tuple result, corresponding to `non_marginalized_decoder_marginals`
            epoch_extracted_most_likely_positions_1D = [a_result.get('most_likely_positions_1D', None) for a_result in active_marginals]
            if len([v for v in epoch_extracted_most_likely_positions_1D if (v is not None)]) != len(epoch_extracted_posteriors):
                ## if some entries are None (more likely ALL are None), recompute from p_x_given_n but this only works for `non_marginalized_decoder_marginals`:
                epoch_extracted_most_likely_positions_1D = deepcopy(self.most_likely_positions_list)

            if _common_epoch_df_column_dict is None:
                # NOTE: these values are the same for every iteration of this loop (as the number of epochs in filter_epochs don't change:
                n_epoch_time_bins = [np.shape(a_posterior)[-1] for a_posterior in epoch_extracted_posteriors]
                result_t_bin_idx_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=i) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)])
                epoch_df_idx_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=filter_epochs_df.index[i]) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)])


                # set `_common_epoch_df_column_dict` so it is no longer None:
                # _common_epoch_df_column_dict = {'result_t_bin_idx': result_t_bin_idx_column, 'epoch_df_idx': epoch_df_idx_column, 'parent_epoch_label': epoch_label_column, 'parent_epoch_duration': epoch_duration_column}
                # _common_epoch_df_column_dict = {'result_t_bin_idx': result_t_bin_idx_column, 'epoch_df_idx': epoch_df_idx_column}
                _common_epoch_df_column_dict = {'parent_epoch_df_idx': epoch_df_idx_column, 'result_t_bin_idx': result_t_bin_idx_column}

                # ## All
                # epochs_repeated_epoch_index: NDArray = np.hstack([np.full((n_bins, ), i) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
                # epochs_repeated_sub_epoch_time_bin_index: NDArray = np.hstack([np.arange(n_bins) for n_bins in n_epoch_time_bins]) # np.shape(epochs_t_centers) # (19018,)
                # ## Redundant but very helpful parent_epoch_* columns
                # epochs_repeated_parent_epoch_start_t: NDArray = np.hstack([np.full((n_bins, ), filter_epochs_df['start'].values[i]) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
                # epochs_repeated_parent_epoch_stop_t: NDArray = np.hstack([np.full((n_bins, ), filter_epochs_df['stop'].values[i]) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)

                ## All
                epochs_repeated_epoch_index: NDArray = np.concatenate([np.full((n_bins, ), i) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
                epochs_repeated_sub_epoch_time_bin_index: NDArray = np.concatenate([np.arange(n_bins) for n_bins in n_epoch_time_bins]) # np.shape(epochs_t_centers) # (19018,)
                ## Redundant but very helpful parent_epoch_* columns
                epochs_repeated_parent_epoch_start_t: NDArray = np.concatenate([np.full((n_bins, ), filter_epochs_df['start'].values[i]) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
                epochs_repeated_parent_epoch_stop_t: NDArray = np.concatenate([np.full((n_bins, ), filter_epochs_df['stop'].values[i]) for i, n_bins in enumerate(n_epoch_time_bins)]) # np.shape(epochs_p_x_given_n) # (2, 19018)
                epoch_label_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=filter_epochs_df[parent_epoch_label_col_name].values[i]) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)])
                epoch_duration_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=filter_epochs_df['duration'].values[i]) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)]) #TODO 2025-04-18 21:05: - [ ] Added 'parent_epoch_duration' to output dataframe!
                # epoch_neuronal_participation_column = np.concatenate([np.full((an_epoch_time_bins, ), fill_value=filter_epochs_df['participation'].values[i]) for i, an_epoch_time_bins in enumerate(n_epoch_time_bins)])
                
                _common_epoch_df_column_dict.update(**{'epoch_id': epochs_repeated_epoch_index, 'sub_epoch_time_bin_index': epochs_repeated_sub_epoch_time_bin_index, 'parent_epoch_id': epochs_repeated_epoch_index, 'parent_epoch_label': epoch_label_column, 'parent_epoch_start_t': epochs_repeated_parent_epoch_start_t, 'parent_epoch_end_t': epochs_repeated_parent_epoch_stop_t, 'parent_epoch_duration': epoch_duration_column, 'most_likely_positions_1D': np.nan})
                

            if epoch_extracted_most_likely_positions_1D is not None:
                time_bin_extracted_most_likely_positions_1D_column = np.concatenate([np.atleast_1d(an_epoch_extracted_most_likely_positions_1D[:, 0]) for i, an_epoch_extracted_most_likely_positions_1D in enumerate(epoch_extracted_most_likely_positions_1D)]) 
                assert len(time_bin_extracted_most_likely_positions_1D_column) == len(epoch_label_column), f"len(time_bin_extracted_most_likely_positions_1D_column): {len(time_bin_extracted_most_likely_positions_1D_column)}, len(epoch_label_column): {len(epoch_label_column)}"
                _common_epoch_df_column_dict.update(**{'most_likely_positions_1D': time_bin_extracted_most_likely_positions_1D_column, })
                print(f'2025-08-13 19:18 Adding "most_likely_positions_1D" column to `epoch_time_bin_marginals_df`.')

            all_columns.extend(active_columns)
            # all_epoch_extracted_posteriors = np.hstack((all_epoch_extracted_posteriors, epoch_extracted_posteriors))
            all_epoch_extracted_posteriors.append(np.hstack((epoch_extracted_posteriors)))
            # all_epoch_extracted_posteriors.extend(epoch_extracted_posteriors)

        all_epoch_extracted_posteriors = np.vstack(all_epoch_extracted_posteriors) # (4, n_time_bins) - (4, 5495)
        epoch_time_bin_marginals_df = pd.DataFrame(all_epoch_extracted_posteriors.T, columns=all_columns)
        ## add all extra columns (parent epoch columns, etc):
        for k, v in _common_epoch_df_column_dict.items():
            epoch_time_bin_marginals_df[k] = v
            
        # epoch_time_bin_marginals_df['result_t_bin_idx'] = result_t_bin_idx_column # a.k.a result label
        # epoch_time_bin_marginals_df['epoch_df_idx'] = epoch_df_idx_column
        # epoch_time_bin_marginals_df['parent_epoch_label'] = epoch_label_column
        # epoch_time_bin_marginals_df['parent_epoch_duration'] = epoch_duration_column
        # epoch_time_bin_marginals_df['parent_epoch_neuronal_participation'] = epoch_neuronal_participation_column

        if (len(flat_time_bin_centers_column) < len(epoch_time_bin_marginals_df)):
            # 2024-01-25 - This fix DOES NOT HELP. The constructed size is the same as the existing `flat_time_bin_centers_column`.
            
            # bin errors are occuring:
            print(f'encountering bin issue! flat_time_bin_centers_column: {np.shape(flat_time_bin_centers_column)}. len(epoch_time_bin_marginals_df): {len(epoch_time_bin_marginals_df)}. Attempting to fix.')
            # find where the indicies are less than two bins
            # miscentered_bin_indicies = np.where(n_epoch_time_bins < 2)
            # replace those centers with just the center of the epoch
            t_bin_centers_list = []
            for epoch_idx, curr_epoch_time_bin_container in enumerate(self.time_bin_containers):
                curr_epoch_n_time_bins = n_epoch_time_bins[epoch_idx]
                if (curr_epoch_n_time_bins < 2):
                    an_epoch_center = filter_epochs_df['center_t'].to_numpy()[epoch_idx]
                    t_bin_centers_list.append([an_epoch_center]) # list containing only the single epoch center
                else:
                    t_bin_centers_list.append(curr_epoch_time_bin_container.centers) 

            flat_time_bin_centers_column = np.concatenate(t_bin_centers_list)
            print(f'\t fixed flat_time_bin_centers_column: {np.shape(flat_time_bin_centers_column)}')

        epoch_time_bin_marginals_df['label'] = epoch_time_bin_marginals_df.index.values.astype(int)
        epoch_time_bin_marginals_df['start'] = deepcopy(flat_time_bin_start_edge_column) 
        epoch_time_bin_marginals_df['t_bin_center'] = deepcopy(flat_time_bin_centers_column) 
        epoch_time_bin_marginals_df['stop'] = deepcopy(flat_time_bin_stop_edge_column)

        # except ValueError:
            # epoch_time_bin_marginals_df['t_bin_center'] = deepcopy(a_decoder_result.filter_epochs['center_t'].to_numpy()[miscentered_bin_indicies])
        
        ## add columns from parent df:
        if transfer_column_names_list is not None:
            for a_test_transfer_column_name in transfer_column_names_list:
                # a_test_transfer_column_name: str = 'maze_id'
                epoch_time_bin_marginals_df[a_test_transfer_column_name] = epoch_time_bin_marginals_df['parent_epoch_df_idx'].map(lambda idx: filter_epochs_df.loc[idx, a_test_transfer_column_name])
            
        return epoch_time_bin_marginals_df
    
    

    @function_attributes(short_name=None, tags=['mask', 'unit-spike-counts', 'pure'], input_requires=[], output_provides=[], uses=['spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask'], used_by=[], creation_date='2025-03-04 01:32', related_items=[])
    def mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(self, spikes_df: pd.DataFrame, min_num_spikes_per_bin_to_be_considered_active:int=1, min_num_unique_active_neurons_per_time_bin:int=3, masked_bin_fill_mode:MaskedTimeBinFillType='last_valid') -> Tuple["DecodedFilterEpochsResult", Tuple[NDArray, NDArray]]:
        """ Returns a copy of itself, masked by finding periods where there is insufficient firing to decode based on the provided paramters, copies the decoded result and returns a version with positions back-filled from the last bin that did meet the minimum firing criteria
        
        Pure: does not modify self
        
        a_decoded_result.p_x_given_n_list[0].shape # (59, 2, 69487)
        a_decoded_result.most_likely_position_indicies_list[0].shape # .shape (2, 69487)
        a_decoded_result.most_likely_positions_list[0].shape # .shape (69487, 2)


        spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
        maksed_pseudo2D_continuous_specific_decoded_result, mask_index_tuple = pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
        # (is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list) = mask_index_tuple
        maksed_pseudo2D_continuous_specific_decoded_result
                
        #TODO 2025-03-04 10:10: - [ ] Seems like `a_decoder` is just passed-through unaltered. Could refactor into a classmethod of `DecodedFilterEpochsResult`
        
        #TODO 2025-03-26 05:55: - [ ] Perhaps spike requirement should be in Hz, so it's normalized by the time_bin_size instead of pure spikes. Although you can't go lower than one spike.... so it does seem discrete somewhat.
        
        
        """
        from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo, get_bin_edges

        assert masked_bin_fill_mode in ['ignore', 'last_valid', 'nan_filled', 'dropped']
        
        # a_decoder = deepcopy(a_decoder)
        a_decoded_result: DecodedFilterEpochsResult = deepcopy(self) ## copy self to make the decoded result duplicate
        
        num_filter_epochs: int = a_decoded_result.num_filter_epochs
        
        # time_bin_edges: NDArray = deepcopy(results1D.continuous_results['global'].time_bin_edges[0])
        # time_bin_edges_list: List[NDArray] = deepcopy(a_decoded_result.time_bin_edges)
        # assert len(time_bin_edges_list) == num_filter_epochs
        
        is_time_bin_active_list = []
        inactive_mask_list = []
        
        all_time_bin_indicies_list = []
        last_valid_indices_list = []

        for i in np.arange(num_filter_epochs):
            ## Mask each output value
            # inactive_mask_indicies = np.where(inactive_mask)[0]
            *num_spatial_dims_list, num_time_bins = np.shape(a_decoded_result.p_x_given_n_list[i])
            if len(num_spatial_dims_list) == 2: 
                # 2D
                num_positions, num_y_bins = num_spatial_dims_list            
            elif len(num_spatial_dims_list) == 1:
                # 1D
                num_positions = num_spatial_dims_list
            else:
                raise NotImplementedError(f'len(num_spatial_dims_list): {len(num_spatial_dims_list)}: num_spatial_dims_list: {num_spatial_dims_list} but expected 2 or 3')

            a_time_bin_edges: NDArray = deepcopy(a_decoded_result.time_bin_edges[i])
            if (len(a_time_bin_edges) != (num_time_bins+1)):
                #@IgnoreException
                print(f'WARN: Epoch[{i}]: len(a_time_bin_edges): {len(a_time_bin_edges)} != (num_time_bins+1): {(num_time_bins+1)}.') # continuing.
                # raise IndexError(f'len(a_time_bin_edges): {len(a_time_bin_edges)} != (num_time_bins+1): {(num_time_bins+1)}') #@IgnoreException
                # continue
                break
            else:
                assert len(a_time_bin_edges) == (num_time_bins+1)
        
            unit_specific_time_binned_spike_counts, unique_units, (is_time_bin_active, inactive_mask, mask_rgba) = spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask(time_bin_edges=a_time_bin_edges,
                                                                                                                                                                                    min_num_spikes_per_bin_to_be_considered_active=min_num_spikes_per_bin_to_be_considered_active,
                                                                                                                                                                                    min_num_unique_active_neurons_per_time_bin=min_num_unique_active_neurons_per_time_bin)
            
            # Make a copy of the original data before masking
            original_data = a_decoded_result.p_x_given_n_list[i].copy()
            all_time_bin_indicies = np.arange(num_time_bins, dtype=int) ## all time bins


            if masked_bin_fill_mode != 'ignore':
                # Mask inactive time bins with NaN in all modes except ignore mode
                # Use arr[..., inactive_mask], which works for any number of dimensions:
                a_decoded_result.p_x_given_n_list[i][..., inactive_mask] = np.nan
                a_decoded_result.most_likely_position_indicies_list[i][..., inactive_mask] = -1 # use -1 instead of np.nan as it needs to be integer
                a_decoded_result.most_likely_positions_list[i][inactive_mask, ...] = np.nan

            if masked_bin_fill_mode == 'last_valid':
                ## backfill from last_valid decoded position
                last_valid_indices = np.zeros(num_time_bins, dtype=int)
                current_valid_idx = 0
                # Fill invalid time bins with the last valid value - EFFICIENT IMPLEMENTATION
                if np.any(is_time_bin_active):  # Only proceed if we have some valid values
                    # Calculate "last valid index" lookup array - very efficient O(n) operation
                    for t in np.arange(num_time_bins):
                        if is_time_bin_active[t]:
                            current_valid_idx = t
                        last_valid_indices[t] = current_valid_idx
                    
                    ## when done, have `last_valid_indices`
                    # print(f'last_valid_indices: {last_valid_indices}')
                    a_decoded_result.p_x_given_n_list[i][..., all_time_bin_indicies] = original_data[..., last_valid_indices]

                    # Also fix the most_likely_position arrays using the same technique
                    # For most_likely_position_indicies_list (shape: 2, num_time_bins)
                    a_decoded_result.most_likely_position_indicies_list[i][:, all_time_bin_indicies] = a_decoded_result.most_likely_position_indicies_list[i][:, last_valid_indices]
                    
                    # For most_likely_positions_list (shape: num_time_bins, 2)
                    a_decoded_result.most_likely_positions_list[i][all_time_bin_indicies, ...] = a_decoded_result.most_likely_positions_list[i][last_valid_indices, ...] # Use a dimension-agnostic approach:

                else:
                    ## no valid time bins
                    print(f'WARN: Epoch[{i}]: with {num_time_bins} time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.')

            elif masked_bin_fill_mode == 'dropped':
                ## just drop the invalid bins by selecting via the `is_time_bin_active` (active_mask):
                # Drop inactive time bins by selecting only active ones
                a_decoded_result.p_x_given_n_list[i] = a_decoded_result.p_x_given_n_list[i][..., is_time_bin_active]
                a_decoded_result.most_likely_position_indicies_list[i] = a_decoded_result.most_likely_position_indicies_list[i][:, is_time_bin_active]
                a_decoded_result.most_likely_positions_list[i] = a_decoded_result.most_likely_positions_list[i][is_time_bin_active, :]

                a_binning_container: BinningContainer = deepcopy(a_decoded_result.time_bin_containers[i])
                # a_binning_container.centers = a_binning_container.centers[is_time_bin_active]
                a_sliced_centers = deepcopy(a_decoded_result.time_bin_containers[i].centers[is_time_bin_active])
                center_info = BinningContainer.build_center_binning_info(centers=a_sliced_centers, variable_extents=a_binning_container.center_info.variable_extents)
                
                try:
                    a_decoded_result.time_bin_edges[i] = get_bin_edges(a_sliced_centers) #
                    ## make whole new container
                    a_decoded_result.time_bin_containers[i] = BinningContainer(centers=a_sliced_centers, edges=a_decoded_result.time_bin_edges[i])
                
                except IndexError as e:
                    if len(a_sliced_centers) == 0:
                        ## no center => no edges
                        a_decoded_result.time_bin_edges[i] = np.array([])
                        ## make whole new container
                        edge_info = BinningInfo(variable_extents=a_binning_container.edge_info.variable_extents, step=a_binning_container.edge_info.step, num_bins=0)
                        a_decoded_result.time_bin_containers[i] = BinningContainer(centers=a_sliced_centers, edges=a_decoded_result.time_bin_edges[i], edge_info=edge_info, center_info=center_info)
                
                    else:
                        assert len(a_sliced_centers) == 1, f"a_sliced_centers: {a_sliced_centers} -- len(a_sliced_centers): {len(a_sliced_centers)}"
                        assert len(a_binning_container.center_info.variable_extents) == 2, f"a_binning_container.center_info.variable_extents: {a_binning_container.center_info.variable_extents}"
                        a_decoded_result.time_bin_edges[i] = np.array(a_binning_container.center_info.variable_extents) ## use the extents directly
                        ## make whole new container
                        a_decoded_result.time_bin_containers[i] = BinningContainer(centers=a_sliced_centers, edges=a_decoded_result.time_bin_edges[i])
                
                    
                except Exception as e:
                    raise e
                

                # a_decoded_result.time_bin_containers[i] = a_decoded_result.time_bin_containers[i][is_time_bin_active]

                # a_decoded_result.time_bin_edges[i] = a_time_bin_edges[is_time_bin_active]

                # a_decoded_result.time_bin_edges[i] = a_time_bin_edges[is_time_bin_active] ## for sure wrong
                # a_decoded_result.nbins[i] = len(a_time_bin_edges[is_time_bin_active]) # 2025-03-20 16:46 -  IndexError: boolean index did not match indexed array along dimension 0; dimension is 239 but corresponding boolean dimension is 238 
                a_decoded_result.nbins[i] = np.sum(is_time_bin_active) ## this should count the number of bin centers, previously (pre 2025-03-20 16:50) it was counting the number of bin edges which was logically wrong.
                
                a_decoded_result.spkcount[i] = a_decoded_result.spkcount[i][:, is_time_bin_active] # (80, 66) - (n_neurons, n_epoch_t_bins[i])
                ## maybe messing up: epoch_description_list,
                last_valid_indices = None # we don't need `last_valid_indices` in these modes
                
            elif masked_bin_fill_mode == 'nan_filled':
                ## just NaN out the invalid bins, which we've already done as a pre-processing step
                last_valid_indices = None # we don't need `last_valid_indices` in these modes
                pass
            elif masked_bin_fill_mode == 'ignore':
                ## do nothing, not even NaN out the invalid values.
                last_valid_indices = None # we don't need `last_valid_indices` in these modes
                pass
            else:
                raise NotImplementedError(f"masked_bin_fill_mode: '{masked_bin_fill_mode}' was not one of the known valid modes: ['last_valid', 'nan_filled', 'dropped']")


            # Add the marginal container to the list
            curr_unit_marginal_x, curr_unit_marginal_y = BasePositionDecoder.perform_build_marginals(p_x_given_n=a_decoded_result.p_x_given_n_list[i], most_likely_positions=a_decoded_result.most_likely_positions_list[i], debug_print=False)
            if curr_unit_marginal_x is not None:
                a_decoded_result.marginal_x_list[i] = curr_unit_marginal_x
            if curr_unit_marginal_y is not None:
                a_decoded_result.marginal_y_list[i] = curr_unit_marginal_y
            if curr_unit_marginal_z is not None:
                a_decoded_result.marginal_z_list[i] = curr_unit_marginal_z          


            ## END if np.any(is_time_bin_active)
            is_time_bin_active_list.append(is_time_bin_active)
            inactive_mask_list.append(last_valid_indices)
            all_time_bin_indicies_list.append(all_time_bin_indicies)
            last_valid_indices_list.append(last_valid_indices)
                    
        #END for i in np.arange(num_filter_epochs)    
        
        return a_decoded_result, (is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list)


    @function_attributes(short_name=None, tags=['pseudo2D', 'timeline-track', '1D', 'split-to-1D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-26 07:23', related_items=['pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersContinuouslyDecodedResult.split_pseudo2D_continuous_result_to_1D_continuous_result'])
    def split_pseudo2D_result_to_1D_result(self, pseudo2D_decoder_names_list: Optional[str]=None, a_pseudo2D_decoder: Optional[Any]=None, xbin_centers: Optional[NDArray]=None, ybin_centers: Optional[NDArray]=None, debug_print=False) -> Dict[types.DecoderName, "DecodedFilterEpochsResult"]:
        """ Get 1D representations of the Pseudo2D track (4 decoders) so they can be plotted on seperate tracks and bin-debugged independently.

        This returns the "all-decoders-sum-to-1-across-time' normalization style that Kamran likes

        xbin_centers: the optional position x-bin centers
        ybin_centers: the optional position x-bin centers, emphatically not the decoder centers


        Usage:        
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult

            ## INPUTS: laps_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
            unique_decoder_names = ['long', 'short']
            laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            masked_laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = masked_laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)

            # OUTPUTS: laps_pseudo2D_split_to_1D_continuous_results_dict, masked_laps_pseudo2D_split_to_1D_continuous_results_dict


        """
        if pseudo2D_decoder_names_list is None:
            pseudo2D_decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        
        p_x_given_n_shapes_list = [np.shape(v) for v in self.p_x_given_n_list]  #  [(59, 2, 66), (59, 2, 102), (59, 2, 226), ...]
        p_x_given_n_shapes_list = np.vstack(p_x_given_n_shapes_list) # (84, 3)
        posterior_ndim: int = (np.shape(p_x_given_n_shapes_list)[-1]-1) # 1 or 2
        if a_pseudo2D_decoder is not None:
            assert xbin_centers is None, f"don't provide both xbin_centers and a_pseudo2D_decoder"
            assert ybin_centers is None, f"don't provide both ybin_centers and a_pseudo2D_decoder"
            original_position_data_shape = deepcopy(a_pseudo2D_decoder.original_position_data_shape[:-1]) ## all but the pseudo dim
            xbin_centers: Optional[NDArray] = deepcopy(a_pseudo2D_decoder.xbin_centers)
            ybin_centers: Optional[NDArray] = None

        if posterior_ndim == 1:
            print(f'WARN: already 1D')
            raise ValueError(f'ALREADY 1D, cannot split!')
            return pseudo2D_decoder_names_list
        else:
            spatial_n_bin_sizes = p_x_given_n_shapes_list[:, :posterior_ndim] ## get the spatial columns, and they should all be the same
            assert np.all(spatial_n_bin_sizes == spatial_n_bin_sizes[0, :]), f"all rows must have the same number of spatial bins, but spatial_n_bin_sizes: {spatial_n_bin_sizes}"
            n_xbins, n_ybins = spatial_n_bin_sizes[0, :]
            if debug_print:
                print(f'n_xbins: {n_xbins}, n_ybins: {n_ybins}')
            ## we will reduce along the y-dim dimension

            if xbin_centers is None:
                print(f'WARN: xbin_centers was not provided, so the output 1D `output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions_list` WILL BE INCORRECT!')

            ## Extract the Pseudo2D results as separate 1D tracks
            ## Split across the 2nd axis to make 1D posteriors that can be displayed in separate dock rows:
            assert n_ybins == len(pseudo2D_decoder_names_list), f"for pseudo2D_decoder_names_list: {pseudo2D_decoder_names_list}\n\texpected the len(pseudo2D_decoder_names_list): {len(pseudo2D_decoder_names_list)} pseudo-y bins for the decoder in n_ybins. but found n_ybins: {n_ybins}"
        
            output_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {}
            for i, a_decoder_name in enumerate(pseudo2D_decoder_names_list):
                ## make separate `DecodedFilterEpochsResult` objects                
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name] = deepcopy(self) ## copy the whole pseudo2D result
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n_list = [np.squeeze(p_x_given_n[:, i, :]) for p_x_given_n in output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n_list] ## or could squish them here

                # #TODO 2025-06-13 14:29: - [X] fix busted-up columns: ['most_likely_position_indicies_list', 'most_likely_positions'
                output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies_list = [np.squeeze(np.argmax(p_x_given_n, axis=0)) for p_x_given_n in output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].p_x_given_n_list]  # average along time dimension
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions_list = None
                if (xbin_centers is not None) and (ybin_centers is not None):
                    ## NOTE: I think 2D position decoding isn't implemented here.
                    # much more efficient than the other implementation. Result is # (85844, 2)
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions_list = [np.vstack((xbin_centers[most_likely_position_indicies[0,:]], self.ybin_centers[most_likely_position_indicies[1,:]])).T for most_likely_position_indicies in output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies_list]
                elif (xbin_centers is not None):
                    # 1D Decoder case:
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions_list = [np.squeeze(xbin_centers[most_likely_position_indicies]) for most_likely_position_indicies in output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies_list]
                else:
                    output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_positions_list = [[] for most_likely_position_indicies in output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].most_likely_position_indicies_list]
                # #TODO 2025-06-13 14:30: - [ ] Finish recomputing marginals
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].marginal_x_list = None
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].marginal_y_list = None
                # output_pseudo2D_split_to_1D_continuous_results_dict[a_decoder_name].compute_marginals()


            ## END for i, a_de...
            return output_pseudo2D_split_to_1D_continuous_results_dict


    @function_attributes(short_name=None, tags=['pseudo2D', 'marginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-08 08:02', related_items=[])
    def get_pseudo2D_result_to_pseudo2D_marginalization_result(self, pseudo2D_decoder_names_list: Optional[str]=None, debug_print=False) -> Tuple[NDArray, NDArray]:
        """ Get marginalization of the Pseudo2D track (4 decoders) so they can be plotted on seperate tracks and bin-debugged independently.

        Usage:
            ## INPUTS: laps_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
            flat_time_window_centers, flat_marginal_y_p_x_given_n = laps_pseudo2D_continuous_specific_decoded_result.get_pseudo2D_result_to_pseudo2D_marginalization_result(pseudo2D_decoder_names_list=unique_decoder_names)
            flat_marginal_y_p_x_given_n


        """
        if pseudo2D_decoder_names_list is None:
            pseudo2D_decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        
        marginal_y_p_x_given_n_list = [v.p_x_given_n for v in self.marginal_y_list]  #  [(2, 66), (2, 102), (2, 226), ...] (n_ybins, n_epoch_t_bins[i])
        flat_marginal_y_p_x_given_n: NDArray = np.hstack(marginal_y_p_x_given_n_list) # (2, 19018)
        n_ybins, n_flat_tbin_centers = np.shape(flat_marginal_y_p_x_given_n)
        if debug_print:
            print(f'n_flat_tbin_centers: {n_flat_tbin_centers}, n_ybins: {n_ybins}')

        flat_time_window_centers: NDArray = np.hstack(self.time_window_centers)
        assert len(flat_time_window_centers) == n_flat_tbin_centers
        
        ## we will reduce along the y-dim dimension  
        assert n_ybins == len(pseudo2D_decoder_names_list), f"for pseudo2D_decoder_names_list: {pseudo2D_decoder_names_list}\n\texpected the len(pseudo2D_decoder_names_list): {len(pseudo2D_decoder_names_list)} pseudo-y bins for the decoder in n_ybins. but found n_ybins: {n_ybins}"
    
        return flat_time_window_centers, flat_marginal_y_p_x_given_n


    @classmethod
    def perform_add_additional_epochs_columns(cls, a_result: "DecodedFilterEpochsResult", session_name: str, t_delta: float, **kwargs) -> "DecodedFilterEpochsResult":
        """ adds the session-related extra columns to the result's .filter_epochs

        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        a_result = DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, session_name=session_name, t_delta=t_delta)

        """
        active_filter_epochs: pd.DataFrame = ensure_dataframe(deepcopy(a_result.filter_epochs))        
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        active_filter_epochs = active_filter_epochs.across_session_identity.add_session_df_columns(session_name=session_name, curr_session_t_delta=t_delta, **kwargs) # , time_bin_size=None, time_col='ripple_start_t'
        a_result.filter_epochs = active_filter_epochs ## assign
        return a_result


    def add_additional_epochs_columns(self, session_name: str, t_delta: float, **kwargs) -> pd.DataFrame:
        """ updates self.filter_epochs

        session_name: str = curr_active_pipeline.session_name
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        a_result.add_additional_epochs_columns(session_name=session_name, t_delta=t_delta)

        """
        self.perform_add_additional_epochs_columns(a_result=self, session_name=session_name, t_delta=t_delta, **kwargs)
        # active_filter_epochs: pd.DataFrame = ensure_dataframe(deepcopy(self.filter_epochs))        
        # # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        # active_filter_epochs = active_filter_epochs.across_session_identity.add_session_df_columns(session_name=session_name, curr_session_t_delta=t_delta, **kwargs) # , time_bin_size=None, time_col='ripple_start_t'
        # self.filter_epochs = active_filter_epochs ## assign
        return self.filter_epochs


        

# ==================================================================================================================== #
# Placemap Position Decoders                                                                                           #
# ==================================================================================================================== #
from neuropy.utils.mixins.binning_helpers import GridBinDebuggableMixin, DebugBinningInfo, BinnedPositionsMixin

# ==================================================================================================================== #
# Stateless Decoders (New 2023-04-06)                                                                                  #
# ==================================================================================================================== #

@custom_define(slots=False, eq=False)
class BasePositionDecoder(HDFMixin, AttrsBasedClassHelperMixin, ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin, NeuronUnitSlicableObjectProtocol, BinnedPositionsMixin):
    """ 2023-04-06 - A simplified data-only version of the decoder that serves to remove all state related to specific computations to make each run independent 
    Stores only the raw inputs that are used to decode, with the user specifying the specifics for a given decoding (like time_time_sizes, etc.


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder


    """
    pf: PfND = serialized_field(repr=keys_only_repr)

    neuron_IDXs: np.ndarray = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})
    neuron_IDs: np.ndarray = serialized_field(default=None, is_computable=True, metadata={'shape': ('n_neurons',)})
    F: np.ndarray = non_serialized_field(default=None, repr=False, metadata={'shape': ('n_flat_position_bins','n_neurons',)})
    P_x: np.ndarray = non_serialized_field(default=None, repr=False, metadata={'shape': ('n_flat_position_bins',)})

    setup_on_init:bool = non_serialized_field(default=True)
    post_load_on_init:bool = non_serialized_field(default=False)
    debug_print: bool = non_serialized_field(default=False)

    # Properties _________________________________________________________________________________________________________ #

    # placefield properties:
    @property
    def ratemap(self) -> Ratemap:
        return self.pf.ratemap

    @property
    def ndim(self) -> int:
        return int(self.pf.ndim)

    @property
    def num_neurons(self) -> int:
        """The num_neurons property."""
        return self.ratemap.n_neurons # np.shape(self.neuron_IDs) # or self.ratemap.n_neurons

    # ratemap properties (xbin & ybin)  
    @property
    def xbin(self) -> NDArray:
        return self.ratemap.xbin
    @property
    def ybin(self) -> NDArray:
        return self.ratemap.ybin
    @property
    def xbin_centers(self) -> NDArray:
        return self.ratemap.xbin_centers
    @property
    def ybin_centers(self) -> NDArray:
        return self.ratemap.ybin_centers
    @property
    def n_xbin_edges(self) -> int:
        return len(self.xbin) 
    @property
    def n_ybin_edges(self) -> Optional[int]:
        """ the number of ybin edges. """
        if self.ybin is None:
            return None
        else:
             return len(self.ybin)
    @property
    def n_xbin_centers(self) -> int:
        return (len(self.xbin) - 1) # the -1 is to get the counts for the centers only
    @property
    def n_ybin_centers(self) -> Optional[int]:
        if self.ybin is None:
            return None
        else:
             return (len(self.ybin) - 1) # the -1 is to get the counts for the centers only
    @property
    def pos_bin_size(self) -> Union[float, Tuple[float, float]]:
        """ extracts pos_bin_size: the size of the x_bin in [cm], from the decoder. 
        returns a tuple if 2D or a single float if 1D
        """
        return self.pf.pos_bin_size
    
    @property
    def original_position_data_shape(self):
        """The original_position_data_shape property."""
        return np.shape(self.pf.occupancy)    

    @property
    def flat_position_size(self):
        """The flat_position_size property."""
        return np.shape(self.F)[0] # like 288

    # PeakLocationRepresentingMixin + ContinuousPeakLocationRepresentingMixin conformances:
    @property
    def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.ratemap.PeakLocationRepresentingMixin_peak_curves_variable
    
    @property
    def ContinuousPeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.ratemap.ContinuousPeakLocationRepresentingMixin_peak_curves_variable
    
    
    
    # ==================================================================================================================== #
    # Initialization                                                                                                       #
    # ==================================================================================================================== #
    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # Perform the primary setup to build the placefield
        if self.setup_on_init:
            self.setup()
            if self.post_load_on_init:
                self.post_load()
        else:
            assert (not self.post_load_on_init), f"post_load_on_init can't be true if setup_on_init isn't true!"


    @classmethod
    def init_from_stateful_decoder(cls, stateful_decoder: "BayesianPlacemapPositionDecoder"):
        """ 2023-04-06 - Creates a new instance of this class from a stateful decoder. """
        # Create the new instance:
        new_instance = cls(pf=deepcopy(stateful_decoder.pf), debug_print=stateful_decoder.debug_print)
        # Return the new instance:
        return new_instance


    @classmethod
    def init_from_placefields(cls, pf: PfND, debug_print=False, **kwargs):
        """ 2023-04-06 - Creates a new instance of this class from a placefields object. """
        # Create the new instance:
        new_instance = cls(pf=deepcopy(pf), debug_print=debug_print, **kwargs)
        # Return the new instance:
        return new_instance



    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        
        self._setup_computation_variables()

    @post_deserialize
    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_computation_variables()

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids, defer_compute_all:bool=False): # defer_compute_all:bool = False
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: neuron_sliced_decoder.pf, ... (much more)

        defer_compute_all: bool - should be set to False if you want to manually decode using custom epochs or something later. Otherwise it will compute for all spikes automatically.
            TODO 2023-04-06 - REMOVE this argument. it is unused. It exists just for backwards compatibility with the stateful decoder.
        """
        # call .get_by_id(ids) on the placefield (pf):
        neuron_sliced_pf: PfND = self.pf.get_by_id(ids)
        ## apply the neuron_sliced_pf to the decoder:
        neuron_sliced_decoder = BasePositionDecoder(neuron_sliced_pf, setup_on_init=self.setup_on_init, post_load_on_init=self.post_load_on_init, debug_print=self.debug_print)
        return neuron_sliced_decoder
    
    @function_attributes(short_name=None, tags=['epoch', 'slice', 'restrict'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-29 19:08', related_items=[])
    def replacing_computation_epochs(self, epochs: Union[Epoch, pd.DataFrame]):
        """Implementors return a copy of themselves with their computation epochs (contained in their placefields at `self.pf`) replaced by the provided ones. The existing epochs are unrelated and do not need to be related to the new ones.
        """
        new_epochs_obj: Epoch = Epoch(ensure_dataframe(deepcopy(epochs)).epochs.get_valid_df()).get_non_overlapping()
        ## Restrict the PfNDs:
        epoch_replaced_pf1D: PfND = self.pf.replacing_computation_epochs(deepcopy(new_epochs_obj))
        ## apply the neuron_sliced_pf to the decoder:
        updated_decoder = BasePositionDecoder(pf=epoch_replaced_pf1D, setup_on_init=self.setup_on_init, post_load_on_init=self.post_load_on_init, debug_print=self.debug_print)
        return updated_decoder


    def conform_to_position_bins(self, target_one_step_decoder, force_recompute=True):
        """ After the underlying placefield (self.pf)'s position bins are changed by calling pf.conform_to_position_bins(...) externally, the computations for the decoder will be messed up (and out of sync).
            Calling this function detects this issue.
            # 2022-12-09 - We want to be able to have both long/short track placefields have the same spatial bins.
            Usage:
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)

            Usage 1D:
                long_pf1D = long_results.pf1D
                short_pf1D = short_results.pf1D
                
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)


            Usage 2D:
                long_pf2D = long_results.pf2D
                short_pf2D = short_results.pf2D
                long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        """
        did_recompute = False
        # Update the internal placefield's position bins if they're not the same as the target_one_step_decoder's:
        self.pf, did_update_internal_pf_bins = self.pf.conform_to_position_bins(target_one_step_decoder.pf)
        
        # Update the one_step_decoders after the short bins have been updated:
        if (force_recompute or did_update_internal_pf_bins): # or (self.p_x_given_n.shape[0] < target_one_step_decoder.p_x_given_n.shape[0])
            # Compute:
            print(f'self will be re-binned to match target_one_step_decoder...')
            self.setup()
            # self.compute_all()
            did_recompute = True # set the update flag
        else:
            # No changes needed:
            did_recompute = False

        return self, did_recompute

    def to_1D_maximum_projection(self, defer_compute_all:bool=True):
        """ returns a copy of the decoder that is 1D 
            defer_compute_all: TODO 2023-04-06 - REMOVE this argument. it is unused. It exists just for backwards compatibility with the stateful decoder.
        """
        # Perform the projection. Can only be ran once.
        new_copy_decoder = deepcopy(self)
        new_copy_decoder.pf = new_copy_decoder.pf.to_1D_maximum_projection() # project the placefields to 1D
        # new_copy_decoder.pf.compute()
        # Test the projection to make sure the dimensionality is correct:
        test_projected_ratemap = new_copy_decoder.pf.ratemap
        assert test_projected_ratemap.ndim == 1, f"projected 1D ratemap must be of dimension 1 but is {test_projected_ratemap.ndim}"
        test_projected_placefields = new_copy_decoder.pf
        assert test_projected_placefields.ndim == 1, f"projected 1D placefields must be of dimension 1 but is {test_projected_placefields.ndim}"
        test_projected_decoder = new_copy_decoder
        assert test_projected_decoder.ndim == 1, f"projected 1D decoder must be of dimension 1 but is {test_projected_decoder.ndim}"
        new_copy_decoder.setup()
        return new_copy_decoder

    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #
    @function_attributes(short_name='decode_specific_epochs', tags=['decode', 'epochs', 'specific'], input_requires=[], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['BayesianPlacemapPositionDecoder.perform_decode_specific_epochs', 'pre_build_epochs_decoding_result', 'perform_pre_built_specific_epochs_decoding'], used_by=['decode_using_new_decoders'], related_items=['get_proper_global_spikes_df'])
    def decode_specific_epochs(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size:float=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
        """
        History:
            Split `perform_decode_specific_epochs` into two subfunctions: `_build_decode_specific_epochs_result_shell` and `_perform_decoding_specific_epochs`

        Uses:
            BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...)
        """
        ## Equivalent:
        # return self.perform_decode_specific_epochs(self, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
        pre_built_epochs_decoding_result = self.pre_build_epochs_decoding_result(spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
        return self.perform_pre_built_specific_epochs_decoding(filter_epochs_decoder_result=pre_built_epochs_decoding_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
    
    @function_attributes(short_name=None, tags=['pre-build', 'efficiency'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-29 00:00', related_items=['decode_specific_epochs', 'perform_pre_built_specific_epochs_decoding'])
    def pre_build_epochs_decoding_result(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size:float=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DynamicContainer:
        """ Builds the results used to call `.perform_pre_built_specific_epochs_decoding(...)`
        History:
            Split from `self.decode_specific_epochs` to allow reuse of the epochs for efficiency
        """
        return self._build_decode_specific_epochs_result_shell(neuron_IDs=self.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
    
    @function_attributes(short_name=None, tags=['decode', 'efficiency'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-29 00:00', related_items=['decode_specific_epochs', 'pre_build_epochs_decoding_result'])
    def perform_pre_built_specific_epochs_decoding(self, filter_epochs_decoder_result: DynamicContainer, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
        """ Called with the results from `.pre_build_epochs_decoding_result(...)`
        History:
            Split from `self.decode_specific_epochs` to allow reuse of the epochs for efficiency
        """
        return self._perform_decoding_specific_epochs(active_decoder=self, filter_epochs_decoder_result=filter_epochs_decoder_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
    

    # ==================================================================================================================== #
    # Non-Modifying Methods:                                                                                               #
    # ==================================================================================================================== #
    @function_attributes(short_name='decode', tags=['MAIN', 'decode', 'pure'], input_requires=[], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions', 'ZhangReconstructionImplementation.neuropy_bayesian_prob'],
        used_by=['BayesianPlacemapPositionDecoder.perform_decode_specific_epochs'])
    def decode(self, unit_specific_time_binned_spike_counts, time_bin_size:float, output_flat_versions=False, debug_print=True):
        """ decodes the neural activity from its internal placefields, returning its posterior and the predicted position 
        Does not alter the internal state of the decoder (doesn't change internal most_likely_positions or posterior, etc)
        
        flat_outputs_container is returned IFF output_flat_versions is True
        
        Inputs:
            unit_specific_time_binned_spike_counts: np.array of shape (num_cells, num_time_bins) - e.g. (69, 20717)

        Requires:
            .P_x
            .F
            .debug_print
            .xbin_centers, .ybin_centers
            .original_position_data_shape
            
        Uses:
            BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions(...)
            ZhangReconstructionImplementation.neuropy_bayesian_prob(...)

        Usages: 
            Used by BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...) to do the actual decoding after building the appropriate spike counts.
        
        """
        ## Capture inputs for debugging:
        computation_debug_mode = False
        if computation_debug_mode:
            debug_intermediates_mode=True
            use_flat_computation_mode=False
            # np.savez_compressed('test_parameters-neuropy_bayesian_prob_2023-04-07.npz', **{'tau':time_bin_size, 'P_x':self.P_x, 'F':self.F, 'n':unit_specific_time_binned_spike_counts})
        else:
            debug_intermediates_mode=False
            use_flat_computation_mode=True

        num_cells = np.shape(unit_specific_time_binned_spike_counts)[0]    
        num_time_windows = np.shape(unit_specific_time_binned_spike_counts)[1]
        if debug_print:
            print(f'num_cells: {num_cells}, num_time_windows: {num_time_windows}')
        with WrappingMessagePrinter(f'decode(...) called. Computing {num_time_windows} windows for final_p_x_given_n...', begin_line_ending='... ', finished_message='decode completed.', enable_print=(debug_print or self.debug_print)):
            if time_bin_size is None:
                print(f'time_bin_size is None, using internal self.time_bin_size.')
                time_bin_size = self.time_bin_size
            
            # Single sweep decoding:
            curr_flat_p_x_given_n = ZhangReconstructionImplementation.neuropy_bayesian_prob(time_bin_size, self.P_x, self.F, unit_specific_time_binned_spike_counts, debug_intermediates_mode=debug_intermediates_mode, use_flat_computation_mode=use_flat_computation_mode, debug_print=(debug_print or self.debug_print))
            if debug_print:
                print(f'curr_flat_p_x_given_n.shape: {curr_flat_p_x_given_n.shape}')
            # all computed
            # Reshape the output variables:
            p_x_given_n = np.reshape(curr_flat_p_x_given_n, (*self.original_position_data_shape, num_time_windows)) # changed for compatibility with 1D decoder
            most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(curr_flat_p_x_given_n, self.original_position_data_shape)

            ## Flat properties:
            if output_flat_versions:
                flat_outputs_container = DynamicContainer(flat_p_x_given_n=curr_flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies)
            else:
                # No flat versions
                flat_outputs_container = None
                
            if self.ndim > 1:
                most_likely_positions = np.vstack((self.xbin_centers[most_likely_position_indicies[0,:]], self.ybin_centers[most_likely_position_indicies[1,:]])).T # much more efficient than the other implementation. Result is # (85844, 2)
            else:
                # 1D Decoder case:
                most_likely_positions = np.squeeze(self.xbin_centers[most_likely_position_indicies[0,:]])
        
            return most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container
            
    @function_attributes(short_name='hyper_perform_decode', tags=['decode', 'pure'], input_requires=['self.neuron_IDs'], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['self.decode', 'BayesianPlacemapPositionDecoder.perform_build_marginals', 'epochs_spkcount'],
        used_by=[])
    def hyper_perform_decode(self, spikes_df, decoding_time_bin_size=0.1, t_start=None, t_end=None, output_flat_versions=False, debug_print=False):
        """ Fully decodes the neural activity from its internal placefields, internally calling `self.decode(...)` and then in addition building the marginals and additional outputs.

        Does not alter the internal state of the decoder (doesn't change internal most_likely_positions or posterior, etc)

        Requires:
            self.neuron_IDs

        Uses:
            self.decode(...)
            BayesianPlacemapPositionDecoder.perform_build_marginals(...)
            epochs_spkcount(...)
        """
        # Range of the maze epoch (where position is valid):
        if t_start is None:
            t_maze_start = spikes_df[spikes_df.spikes.time_variable_name].loc[spikes_df.x.first_valid_index()] # 1048
            t_start = t_maze_start
    
        if t_end is None:
            t_maze_end = spikes_df[spikes_df.spikes.time_variable_name].loc[spikes_df.x.last_valid_index()] # 68159707
            t_end = t_maze_end
        
        epochs_df = pd.DataFrame({'start':[t_start],'stop':[t_end],'label':['epoch']})

        ## final step is to time_bin (relative to the start of each epoch) the time values of remaining spikes
        spkcount, included_neuron_ids, n_tbin_centers, time_bin_containers_list = epochs_spkcount(spikes_df, epochs=epochs_df, bin_size=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=self.neuron_IDs, debug_print=debug_print)
        spkcount = spkcount[0]
        n_tbin_centers = n_tbin_centers[0]
        time_bin_container = time_bin_containers_list[0] # neuropy.utils.mixins.binning_helpers.BinningContainer
        # original_time_bin_container = deepcopy(self.time_binning_container) ## compared to this, we lose the last time_bin (which is partial)
        # is_time_bin_included_in_new = np.isin(original_time_bin_container.centers, time_bin_container.centers) ## see which of the original time bins vanished in the new `time_bin_container`
        ## drop previous values for compatibility
        
        
        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = self.decode(spkcount, time_bin_size=decoding_time_bin_size, output_flat_versions=output_flat_versions, debug_print=debug_print)
        curr_unit_marginal_x, curr_unit_marginal_y = self.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
        return time_bin_container, p_x_given_n, most_likely_positions, curr_unit_marginal_x, curr_unit_marginal_y, flat_outputs_container


    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    @function_attributes(short_name='setup_computation_variables', tags=['pr'], input_requires=[], output_provides=[], uses=['self.pf', 'ZhangReconstructionImplementation.build_concatenated_F'], used_by=[], creation_date='2023-04-06 13:49')
    def _setup_computation_variables(self):
        """ Uses `self.pf` and sets up the computation variables F, P_x, neuron_IDs, neuron_IDXs

            maps: (40, 48, 6)
            np.shape(f_i[i]): (48, 6)
            np.shape(F_i[i]): (288, 1)
            np.shape(F): (288, 40)
            np.shape(P_x): (288, 1)
            
        History:
            Used to be named `_setup_concatenated_F` but renamed because it computes more than that.
        """
        # assert self.pf.ratemap.n_neurons > 0, f"self.pf.ratemap.n_neurons == 0! (No Neurons!)"
        if self.pf.ratemap.n_neurons == 0:
            print(f"self.pf.ratemap.n_neurons == 0! (No Neurons!)")
            raise ValueError(f"self.pf.ratemap.n_neurons == 0! (No Neurons!)")
        
        self.neuron_IDXs, self.neuron_IDs, f_i, F_i, self.F, self.P_x = ZhangReconstructionImplementation.build_concatenated_F(self.pf, debug_print=self.debug_print) # fails when `self.pf.ratemap.n_neurons == 0` aka `self.pf.ratemap.ndim == 0`
        if not isinstance(self.neuron_IDs, np.ndarray):
            self.neuron_IDs = np.array(self.neuron_IDs)
        if not isinstance(self.neuron_IDXs, np.ndarray):
            self.neuron_IDXs = np.array(self.neuron_IDXs)

    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #

    @function_attributes(short_name='prune_to_shared_aclus_only', tags=['decoder','aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-06 12:19')
    @classmethod
    def prune_to_shared_aclus_only(cls, long_decoder, short_decoder):
        """ determines the neuron_IDs present in both long and short decoders (shared aclus) and returns two copies of long_decoder and short_decoder that only contain the shared_aclus

        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

            shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff = BayesianPlacemapPositionDecoder.prune_to_shared_aclus_only(long_decoder, short_decoder)
            n_neurons = len(shared_aclus)

        """

        long_neuron_IDs, short_neuron_IDs = long_decoder.neuron_IDs, short_decoder.neuron_IDs
        # long_neuron_IDs, short_neuron_IDs = long_results_obj.original_1D_decoder.neuron_IDs, short_results_obj.original_1D_decoder.neuron_IDs
        long_short_pf_neurons_diff = _compare_computation_results(long_neuron_IDs, short_neuron_IDs)

        ## Get the normalized_tuning_curves only for the shared aclus (that are common across (long/short/global):
        shared_aclus = long_short_pf_neurons_diff.intersection #.shape (56,)
        # n_neurons = len(shared_aclus)
        # long_is_included = np.isin(long_neuron_IDs, shared_aclus)  #.shape # (104, 63)
        # short_is_included = np.isin(short_neuron_IDs, shared_aclus)

        # Restrict the long decoder to only the aclus present on both decoders:
        long_shared_aclus_only_decoder = long_decoder.get_by_id(shared_aclus)
        short_shared_aclus_only_decoder = short_decoder.get_by_id(shared_aclus) # short currently has a subset of long's alcus so this isn't needed, but just for symmetry do it anyway
        return shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff


    @classmethod
    def _build_decode_specific_epochs_result_shell(cls, neuron_IDs: NDArray, spikes_df: pd.DataFrame, filter_epochs: Union[Epoch, pd.DataFrame], decoding_time_bin_size:float=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DynamicContainer:
        """Precomputes the time_binned spikes for the filter epochs since this is the slowest part of `perform_decode_specific_epochs`

        NOTE: Uses active_decoder.decode(...) to actually do the decoding
        
        Args:
            new_2D_decoder (_type_): _description_
            spikes_df (_type_): _description_
            filter_epochs (_type_): _description_
            decoding_time_bin_size (float, optional): _description_. Defaults to 0.05.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            DecodedFilterEpochsResult: _description_

        Usage:
            filter_epochs_decoder_result = cls._build_decode_specific_epochs_result_shell(neuron_IDs=active_decoder.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)

        """
        filter_epochs_decoder_result = DynamicContainer(most_likely_positions_list=[], p_x_given_n_list=[], marginal_x_list=[], marginal_y_list=[], marginal_z_list=[], most_likely_position_indicies_list=[])
        
        if isinstance(filter_epochs, pd.DataFrame):
            filter_epochs_df = filter_epochs
        else:
            filter_epochs_df = filter_epochs.to_dataframe()
            
        if debug_print:
            print(f'filter_epochs: {filter_epochs.n_epochs}')
        ## Get the spikes during these epochs to attempt to decode from:
        filter_epoch_spikes_df = deepcopy(spikes_df)
        ## Add the epoch ids to each spike so we can easily filter on them:
        filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
        filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] != -1] # Drop all non-included spikes
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

        ## final step is to time_bin (relative to the start of each epoch) the time values of remaining spikes
        spkcount, included_neuron_ids, n_tbin_centers, time_bin_containers_list = epochs_spkcount(filter_epoch_spikes_df, epochs=filter_epochs, bin_size=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=neuron_IDs, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
        num_filter_epochs = len(n_tbin_centers) # one for each epoch in filter_epochs

        # apply np.atleast_1d to all
        for i, a_n_bins in enumerate(n_tbin_centers):
            time_bin_containers_list[i].centers = np.atleast_1d(time_bin_containers_list[i].centers)
            time_bin_containers_list[i].edges = np.atleast_1d(time_bin_containers_list[i].edges)
            # time_bin_containers_list[i].nbins = np.atleast_1d(time_bin_containers_list[i].edges)

            # [(a_n_bins == len(time_bin_containers_list[i].centers)) ]
        assert np.all([(a_n_bins == len(time_bin_containers_list[i].centers)) for i, a_n_bins in enumerate(n_tbin_centers)])
        # assert np.all([(a_n_bins == (len(time_bin_containers_list[i].edges)-1)) for i, a_n_bins in enumerate(nbins)]) # don't know why this wouldn't be true, but it's okay if it isn't I guess
        assert np.all([(a_n_bins == time_bin_containers_list[i].num_bins) for i, a_n_bins in enumerate(n_tbin_centers)])

        filter_epochs_decoder_result.spkcount = spkcount
        filter_epochs_decoder_result.nbins = n_tbin_centers
        filter_epochs_decoder_result.time_bin_containers = time_bin_containers_list
        filter_epochs_decoder_result.decoding_time_bin_size = decoding_time_bin_size
        filter_epochs_decoder_result.filter_epochs = filter_epochs
        filter_epochs_decoder_result.num_filter_epochs = num_filter_epochs

        
        if debug_print:
            print(f'num_filter_epochs: {num_filter_epochs}, nbins: {n_tbin_centers}') # the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]

        # bins = np.arange(epoch.start, epoch.stop, 0.001)
        filter_epochs_decoder_result.most_likely_positions_list = []
        filter_epochs_decoder_result.p_x_given_n_list = []
        # filter_epochs_decoder_result.marginal_x_p_x_given_n_list = []
        filter_epochs_decoder_result.most_likely_position_indicies_list = []
        # filter_epochs_decoder_result.time_bin_centers = []
        filter_epochs_decoder_result.time_bin_edges = []
        
        filter_epochs_decoder_result.marginal_x_list = []
        filter_epochs_decoder_result.marginal_y_list = []
        filter_epochs_decoder_result.marginal_z_list = []
        
        return filter_epochs_decoder_result

    @classmethod
    def _perform_decoding_specific_epochs(cls, active_decoder: "BasePositionDecoder", filter_epochs_decoder_result: DynamicContainer, use_single_time_bin_per_epoch: bool=False, enable_slow_debugging_time_bin_validation: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
        """ Actually performs the computation
        
        NOTE: Uses active_decoder.decode(...) to actually do the decoding
        
        NOTE 2025-02-29: when `enable_slow_debugging_time_bin_validation==True`, this function takes more than twice as long due to a deepcopy!
        """ 
        """ NOTE 2025-02-20 09:40: even when ``enable_slow_debugging_time_bin_validation==False`, this function takes forever because it prints a ton of statements. 
        ERROR: epochs_spkcount(...): epoch[559], nbins[559]: 1 - TODO 2024-08-07 19:11: Building BinningContainer for epoch with fewer than 2 edges (occurs when epoch duration is shorter than the bin size). Using the epoch.start, epoch.stop as the two edges (giving a single bin) but this might be off and cause problems, as they are the edges of the epoch but maybe not "real" edges?
	    ERROR (cont.): even after this hack `slide_view` is not updated, so the returned spkcount is not valid and has the old (wrong, way too many) number of bins. This results in decoded posteriors/postitions/etc with way too many bins downstream. see `SOLUTION 2024-08-07 20:08: - [ ] Recompute the Invalid Quantities with the known correct number of time bins` for info.
        
        
        Occurs with epochs_decoding_time_bin_size = 1.0, frame_divide_bin_size = 1.0

        At end of runs:
            curr_filter_epoch_time_bin_size = 1.0
            curr_epoch_num_time_bins = 1.0
            len(invalid_indicies_list): 505
            np.shape(p_x_given_n): (59, 424)
            _arr_lengths: [570, 570, 570, 570]
        """
        # Looks like we're iterating over each epoch in filter_epochs:
        ## Validate the list lengths for the zip:
        _arr_lengths = [len(v) for v in (np.arange(filter_epochs_decoder_result.num_filter_epochs), filter_epochs_decoder_result.spkcount, filter_epochs_decoder_result.nbins, filter_epochs_decoder_result.time_bin_containers)]
        assert np.allclose(_arr_lengths, filter_epochs_decoder_result.num_filter_epochs), f"all arrays should be equal or the zip will be limited to the fewest items, but _arr_lengths: {_arr_lengths}"
        

        ## Set the static decoder properties
        filter_epochs_decoder_result.pos_bin_edges = deepcopy(active_decoder.xbin)
        assert len(_arr_lengths) > 0, f"no epochs?!?!"
        n_epochs: int = _arr_lengths[0] ## all the same, so any of them works

        # active_decoder.neuron_IDs
        

        for i, curr_filter_epoch_spkcount, curr_epoch_num_time_bins, curr_filter_epoch_time_bin_container in zip(np.arange(filter_epochs_decoder_result.num_filter_epochs), filter_epochs_decoder_result.spkcount, filter_epochs_decoder_result.nbins, filter_epochs_decoder_result.time_bin_containers):
            ## New 2022-09-26 method with working time_bin_centers_list returned from epochs_spkcount
            a_time_bin_edges = np.atleast_1d(curr_filter_epoch_time_bin_container.edges)
            filter_epochs_decoder_result.time_bin_edges.append(a_time_bin_edges)
            if use_single_time_bin_per_epoch:
                assert curr_filter_epoch_time_bin_container.num_bins == 1
                curr_filter_epoch_time_bin_size: float = curr_filter_epoch_time_bin_container.edge_info.step # get the variable time_bin_size from the epoch object
            else:
                curr_filter_epoch_time_bin_size: float = filter_epochs_decoder_result.decoding_time_bin_size
                
            # # validation:
            # n_neurons, n_time_bins = np.shape(curr_filter_epoch_spkcount)
            # assert curr_epoch_num_time_bins == n_time_bins, f"curr_epoch_num_time_bins: {curr_epoch_num_time_bins} != n_time_bins (from curr_filter_epoch_spkcount): {n_time_bins}"
            # a_time_bin_edges_n_time_bins = np.shape(a_time_bin_edges)[0] - 1
            # assert a_time_bin_edges_n_time_bins == n_time_bins, f"a_time_bin_edges_n_time_bins: {a_time_bin_edges_n_time_bins} != n_time_bins (from curr_filter_epoch_spkcount): {n_time_bins}\n\t {a_time_bin_edges}"
            
            most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = active_decoder.decode(curr_filter_epoch_spkcount, time_bin_size=curr_filter_epoch_time_bin_size, output_flat_versions=False, debug_print=debug_print)
            most_likely_positions = np.atleast_1d(most_likely_positions)
            p_x_given_n = np.atleast_1d(p_x_given_n)

            filter_epochs_decoder_result.most_likely_positions_list.append(np.atleast_1d(most_likely_positions))
            filter_epochs_decoder_result.p_x_given_n_list.append(np.atleast_1d(p_x_given_n))
            filter_epochs_decoder_result.most_likely_position_indicies_list.append(np.atleast_1d(most_likely_position_indicies))

            # Add the marginal container to the list
            curr_unit_marginal_x, curr_unit_marginal_y = cls.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
            filter_epochs_decoder_result.marginal_x_list.append(curr_unit_marginal_x)
            filter_epochs_decoder_result.marginal_y_list.append(curr_unit_marginal_y)
        ## end for
        
        ## 2024-08-07: POST-HOC VALIDATE
        invalid_indicies_list = np.where([len(tc.centers) != len(xs) for tc, xs in zip(filter_epochs_decoder_result.time_bin_containers, filter_epochs_decoder_result.most_likely_positions_list)])[0]
        if len(invalid_indicies_list) > 0:
            for invalid_idx in invalid_indicies_list:
                ## find non matching indicies
                
                # ==================================================================================================================== #
                # SOLUTION 2024-08-07 20:08: - [ ] Recompute the Invalid Quantities with the known correct number of time bins:        #
                # ==================================================================================================================== #
                ## TODO 2024-08-07 20:27: - [ ] FUTURE: the core issue is being introduced in `epochs_spkcount` or whatever as a result of the strange slide. It may occur both n=1 and n=2 bins, maybe only n=2 now.
                
                #TODO 2024-08-16 01:10: - [ ] is `self.nbins` being updated?


                # Steps:
                ## 1. replace the edges/centers and their info in `filter_epochs_decoder_result.time_bin_containers`
                ## 2. with correct number of bins, the computed values need to be fixed: most_likely_positions, most_likely_position_indicies, p_x_given_n, spkcount, and marginals
        
                filter_epochs_decoder_result.time_bin_containers[invalid_idx] = BinningContainer.init_from_edges(edges=filter_epochs_decoder_result.time_bin_containers[invalid_idx].edges, edge_info=None)
                ## Now time bin properties are valid - time_bin_containers, n_bins, time_bin_edges
                
                ## testing:
                # invalid_p_x_given_n = deepcopy(filter_epochs_decoder_result.p_x_given_n_list[invalid_idx]).T 
                # invalid_pos = deepcopy(filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx]).T # all most_likely_positions_1D, most_likely_positions_1D, most_likely_positions_1D
                
                ## Fix known invalid quantities (with extra entries relative to time bins) -
                good_indicies = deepcopy(filter_epochs_decoder_result.time_bin_containers[invalid_idx].center_info.bin_indicies)
                
                # time_bin_edges, epoch_description_list
                if filter_epochs_decoder_result.time_bin_edges[invalid_idx] is not None:
                    filter_epochs_decoder_result.time_bin_edges[invalid_idx] = deepcopy(filter_epochs_decoder_result.time_bin_containers[invalid_idx].edges) # filter_epochs_decoder_result.time_bin_edges[invalid_idx][good_indicies]
                    
                # if filter_epochs_decoder_result.epoch_description_list[invalid_idx] is not None:
                #     filter_epochs_decoder_result.epoch_description_list[invalid_idx] = filter_epochs_decoder_result.epoch_description_list[invalid_idx][good_indicies]
                
                # marginals:
                if filter_epochs_decoder_result.marginal_x_list[invalid_idx] is not None:
                    filter_epochs_decoder_result.marginal_x_list[invalid_idx].most_likely_positions_1D = filter_epochs_decoder_result.marginal_x_list[invalid_idx].most_likely_positions_1D[good_indicies]
                    filter_epochs_decoder_result.marginal_x_list[invalid_idx].p_x_given_n = filter_epochs_decoder_result.marginal_x_list[invalid_idx].p_x_given_n[:, good_indicies]

                ## marginal_y_list
                if filter_epochs_decoder_result.marginal_y_list[invalid_idx] is not None:
                    filter_epochs_decoder_result.marginal_y_list[invalid_idx].most_likely_positions_1D = filter_epochs_decoder_result.marginal_y_list[invalid_idx].most_likely_positions_1D[good_indicies]
                    filter_epochs_decoder_result.marginal_y_list[invalid_idx].p_x_given_n = filter_epochs_decoder_result.marginal_y_list[invalid_idx].p_x_given_n[:, good_indicies]
                    

                ## marginal_z_list
                if filter_epochs_decoder_result.marginal_z_list[invalid_idx] is not None:
                    filter_epochs_decoder_result.marginal_z_list[invalid_idx].most_likely_positions_1D = filter_epochs_decoder_result.marginal_z_list[invalid_idx].most_likely_positions_1D[good_indicies]
                    filter_epochs_decoder_result.marginal_z_list[invalid_idx].p_x_given_n = filter_epochs_decoder_result.marginal_z_list[invalid_idx].p_x_given_n[:, good_indicies]
                

                ndim = np.ndim(filter_epochs_decoder_result.p_x_given_n_list[invalid_idx]) # np.shape(filter_epochs_decoder_result.p_x_given_n_list[invalid_idx]) (57, 4, 16)

                if ndim == 3:
                    ## 2D Position Case (less tested)
                    assert np.shape(filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx])[0] == 2, f"filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx] expected to be of shape (2, n_t_bins), but shape {np.shape(filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx])}"
                    filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx] = filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx][:, good_indicies] ## (2, n_t_bins) - (2, 16)  -> (2, 1) DONE
                    
                    assert np.shape(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx])[1] == 2, f"filter_epochs_decoder_result.most_likely_positions_list[invalid_idx] expected to be of shape (n_t_bins, 2), but shape {np.shape(filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx])}"
                    filter_epochs_decoder_result.most_likely_positions_list[invalid_idx] = filter_epochs_decoder_result.most_likely_positions_list[invalid_idx][good_indicies, :] ## (2, n_t_bins): (16, 2) -> (2, 1)

                    filter_epochs_decoder_result.p_x_given_n_list[invalid_idx] = filter_epochs_decoder_result.p_x_given_n_list[invalid_idx][..., good_indicies]
                    filter_epochs_decoder_result.spkcount[invalid_idx] = filter_epochs_decoder_result.spkcount[invalid_idx][:, good_indicies] ## okay for 2D? (80, 16) -> (80, 1)
                    ## do post-hoc checking:
                    assert (len(filter_epochs_decoder_result.time_bin_containers[invalid_idx].centers) == len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx])), f"even after fixing invalid_idx: {invalid_idx}: len(time_bin_containers[invalid_idx].centers): {len(filter_epochs_decoder_result.time_bin_containers[invalid_idx].centers)} != len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx]): {len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx])} "
                    
                else:
                    ## 1D Position Case (what this validation and fix was designed for)
                    filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx] = filter_epochs_decoder_result.most_likely_position_indicies_list[invalid_idx][good_indicies] ## (n_epoch_time_bins, ) one position for each time bin in the replay
                    filter_epochs_decoder_result.most_likely_positions_list[invalid_idx] = filter_epochs_decoder_result.most_likely_positions_list[invalid_idx][good_indicies] ## okay for 2D?
                    filter_epochs_decoder_result.p_x_given_n_list[invalid_idx] = filter_epochs_decoder_result.p_x_given_n_list[invalid_idx][:, good_indicies]
                    filter_epochs_decoder_result.spkcount[invalid_idx] = filter_epochs_decoder_result.spkcount[invalid_idx][good_indicies] ## okay for 2D?
                    ## do post-hoc checking:
                    assert (len(filter_epochs_decoder_result.time_bin_containers[invalid_idx].centers) == len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx])), f"even after fixing invalid_idx: {invalid_idx}: len(time_bin_containers[invalid_idx].centers): {len(filter_epochs_decoder_result.time_bin_containers[invalid_idx].centers)} != len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx]): {len(filter_epochs_decoder_result.most_likely_positions_list[invalid_idx])} "

        assert np.all([(len(filter_epochs_decoder_result.most_likely_positions_list[i]) == len(filter_epochs_decoder_result.time_bin_containers[i].centers)) for i, a_n_bins in enumerate(filter_epochs_decoder_result.nbins)])

        out_result: DecodedFilterEpochsResult = DecodedFilterEpochsResult(**filter_epochs_decoder_result.to_dict()) # dump the dynamic dict as kwargs into the class
        
        if enable_slow_debugging_time_bin_validation:
            out_result.validate_time_bins() ## one last full check! raises assertions if any time bins are still off. Actually very slow!
            
        return out_result
    

    @function_attributes(short_name='perform_decode_specific_epochs', tags=['decode','specific_epochs','epoch', 'classmethod'], input_requires=[], output_provides=[], uses=['active_decoder.decode', 'add_epochs_id_identity', 'epochs_spkcount', 'cls.perform_build_marginals'], used_by=[''], creation_date='2022-12-04 00:00')
    @classmethod
    def perform_decode_specific_epochs(cls, active_decoder, spikes_df: pd.DataFrame, filter_epochs: Union[Epoch, pd.DataFrame], decoding_time_bin_size:float=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
        """Uses the decoder to decode the nerual activity (provided in spikes_df) for each epoch in filter_epochs

        History:
            Split `perform_decode_specific_epochs` into two subfunctions: `_build_decode_specific_epochs_result_shell` and `_perform_decoding_specific_epochs`
        """
        # build output result object:
        filter_epochs_decoder_result = cls._build_decode_specific_epochs_result_shell(neuron_IDs=active_decoder.neuron_IDs, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
        return cls._perform_decoding_specific_epochs(active_decoder=active_decoder, filter_epochs_decoder_result=filter_epochs_decoder_result, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
    
    
    @classmethod
    def perform_build_marginals(cls, p_x_given_n, most_likely_positions, debug_print=False):
        """ builds the marginal distributions, which for the 1D decoder are the same as the main posterior.


        # For 1D Decoder:
            p_x_given_n.shape # (63, 106)
            most_likely_positions.shape # (106,)

            curr_unit_marginal_x['p_x_given_n'].shape # (63, 106)
            curr_unit_marginal_x['most_likely_positions_1D'].shape # (106,)

            curr_unit_marginal_y: None

        External validations:

            assert np.allclose(curr_epoch_result['marginal_x']['p_x_given_n'], curr_epoch_result['p_x_given_n']), f"1D Decoder should have an x-posterior equal to its own posterior"
            assert np.allclose(curr_epoch_result['marginal_x']['most_likely_positions_1D'], curr_epoch_result['most_likely_positions']), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"

        """
        is_1D_decoder = (most_likely_positions.ndim < 2) # check if we're dealing with a 1D decoder, in which case there is no y-marginal (y doesn't exist)
        if debug_print:
            print(f'perform_build_marginals(...): is_1D_decoder: {is_1D_decoder}')
            print(f"\t{p_x_given_n = }\n\t{most_likely_positions = }")
            print(f"\t{np.shape(p_x_given_n) = }\n\t{np.shape(most_likely_positions) = }")

        # p_x_given_n_shape = np.shape(p_x_given_n)

        # Compute Marginal 1D Posterior:
        ## Build a container to hold the marginal distribution and its related values:
        curr_unit_marginal_x = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
        
        if is_1D_decoder:
            # 1D Decoder:
            # p_x_given_n should come in with shape (x_bins, time_bins)
            curr_unit_marginal_x.p_x_given_n = p_x_given_n.copy() # Result should be [x_bins, time_bins]
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n # / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0) # should already be normalized but do it again anyway just in case (so there's a normalized distribution at each timestep)
            # for the 1D decoder case, there are no y-positions
            curr_unit_marginal_y = None

        else:
            # a 2D decoder
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(p_x_given_n, 1)) # sum over all y. Result should be [x_bins x time_bins]
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_x: {curr_unit_marginal_x.p_x_given_n.shape}')

            # y-axis marginal:
            curr_unit_marginal_y = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(p_x_given_n, 0)) # sum over all x. Result should be [y_bins x time_bins]
            curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_y.p_x_given_n.ndim == 0:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_y.p_x_given_n.ndim == 1:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_y.p_x_given_n.shape}')
                
        ## Add the most-likely positions to the posterior_x container:
        if is_1D_decoder:
            ## 1D Decoder Case, there is no y marginal (y doesn't exist)
            if debug_print:
                print(f'np.shape(most_likely_positions): {np.shape(most_likely_positions)}')
            curr_unit_marginal_x.most_likely_positions_1D = np.atleast_1d(most_likely_positions).T # already 1D positions, don't need to extract x-component

            # Validate 1D Conditions:
            assert np.allclose(curr_unit_marginal_x['p_x_given_n'], p_x_given_n, equal_nan=True), f"1D Decoder should have an x-posterior equal to its own posterior"
            assert np.allclose(curr_unit_marginal_x['most_likely_positions_1D'], most_likely_positions, equal_nan=True), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"


            # # Same as np.amax(x, axis=-1)
            # np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)


        else:
            curr_unit_marginal_x.most_likely_positions_1D = most_likely_positions[:,0].T
            curr_unit_marginal_y.most_likely_positions_1D = most_likely_positions[:,1].T
        
        return curr_unit_marginal_x, curr_unit_marginal_y

    @classmethod
    def perform_compute_most_likely_positions(cls, flat_p_x_given_n, original_position_data_shape):
        """ Computes the most likely positions at each timestep from flat_p_x_given_n and the shape of the original position data """
        most_likely_position_flat_indicies = np.argmax(flat_p_x_given_n, axis=0)
        most_likely_position_indicies = np.array(np.unravel_index(most_likely_position_flat_indicies, original_position_data_shape)) # convert back to an array
        # np.shape(most_likely_position_flat_indicies) # (85841,)
        # np.shape(most_likely_position_indicies) # (2, 85841)
        return most_likely_position_flat_indicies, most_likely_position_indicies

    @classmethod
    def perform_compute_forward_filled_positions(cls, most_likely_positions: np.ndarray, is_non_firing_bin: np.ndarray) -> np.ndarray:
        """ applies the forward fill to a copy of the positions based on the is_non_firing_bin boolean array
        zero_bin_indicies.shape # (9307,)
        self.most_likely_positions.shape # (11880, 2)
        
        # NaN out the position bins that were determined to contain 0 spikes
        # Forward fill the now NaN positions with the last good value (for the both axes)
        
        """
        revised_most_likely_positions = most_likely_positions.copy()
        # NaN out the position bins that were determined without any spikes
        if (most_likely_positions.ndim < 2):
            revised_most_likely_positions[is_non_firing_bin] = np.nan 
        else:
            revised_most_likely_positions[is_non_firing_bin, :] = np.nan 
        # Forward fill the now NaN positions with the last good value (for the both axes):
        revised_most_likely_positions = np_ffill_1D(revised_most_likely_positions.T).T
        return revised_most_likely_positions

    @function_attributes(short_name=None, tags=['filter', 'firing_rate', 'frate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-22 19:48', related_items=[])
    def filtered_by_frate(self, minimum_inclusion_fr_Hz: float = 5.0, debug_print=True):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)
        
        """
        return BasePositionDecoder.perform_filter_by_frate(self, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=debug_print)

    @classmethod
    def perform_filter_by_frate(cls, a_decoder, minimum_inclusion_fr_Hz: float = 5.0, debug_print=True):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)
        
        Usage:
            minimum_inclusion_fr_Hz: float = 5.0
            filtered_decoder_list = [filtered_by_frate(a_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True) for a_decoder in (track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)]

        """
        a_pf: PfND = a_decoder.pf # neuropy.analyses.placefields.PfND
        a_ratemap = a_pf.ratemap
        original_neuron_ids = deepcopy(a_ratemap.neuron_ids)
        is_aclu_included = (a_ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz)
        included_aclus = np.array(a_ratemap.neuron_ids)[is_aclu_included]
        if debug_print:
            print(f'len(original_neuron_ids): {len(original_neuron_ids)}, len(included_aclus): {len(included_aclus)}')
        modified_decoder = a_decoder.get_by_id(included_aclus)
        return modified_decoder


    # ==================================================================================================================== #
    # HDF5 Methods:                                                                                                        #
    # ==================================================================================================================== #
    # def to_hdf(self, file_path, key: str, **kwargs):
    #         """ Saves the object to key in the hdf5 file specified by file_path
    #         Usage:
    #             hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
    #             _pfnd_obj: PfND = long_one_step_decoder_1D.pf
    #             _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
    #         """
    #         raise NotImplementedError # implementor must override!


    # ==================================================================================================================== #
    # GridBinDebuggableMixin Conformances                                                                                  #
    # ==================================================================================================================== #
    def get_debug_binning_info(self) -> DebugBinningInfo:
        """Returns relevant debug info about the binning configuration

        Returns:
            DebugBinningInfo: Contains binning dimensions and sizes
        """  
        _obj = self.pf.get_debug_binning_info()
        _obj.nTimeBins = None
        return _obj


    # ==================================================================================================================== #
    # Average Speed/Acceleration per Position Bin                                                                          #
    # ==================================================================================================================== #
    @classmethod
    def _compute_group_stats_for_var(cls, active_position_df: pd.DataFrame, xbin, ybin, variable_name:str = 'speed', debug_print=False):
        """Can compute aggregate statistics (such as the mean) for any column of the position dataframe.

        Args:
            active_position_df (_type_): _description_
            xbin (_type_): _description_
            ybin (_type_): _description_
            variable_name (str, optional): _description_. Defaults to 'speed'.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if ybin is None:
            # Assume 1D:
            ndim = 1
            binned_col_names = ['binned_x',]

        else:
            # otherwise assume 2D:
            ndim = 2
            binned_col_names = ['binned_x','binned_y']


        # For each unique binned_x and binned_y value, what is the average velocity_x at that point?
        position_bin_dependent_specific_average_velocities = active_position_df.groupby(binned_col_names)[variable_name].agg([np.nansum, np.nanmean, np.nanmin, np.nanmax]).reset_index() #.apply(lambda g: g.mean(skipna=True)) #.agg((lambda x: x.mean(skipna=False)))
        if debug_print:
            print(f'np.shape(position_bin_dependent_specific_average_velocities): {np.shape(position_bin_dependent_specific_average_velocities)}')
        # position_bin_dependent_specific_average_velocities # 1856 rows
        if debug_print:
            print(f'np.shape(output): {np.shape(output)}')
            print(f"np.shape(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()): {np.shape(position_bin_dependent_specific_average_velocities['binned_x'])}")
            max_binned_x_index = np.max(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1) # 63
            print(f'max_binned_x_index: {max_binned_x_index}')
        # (len(xbin), len(ybin)): (59, 21)

        if ndim == 1:
            # 1D Position:
            output = np.zeros((len(xbin),)) # (65,)
            output[position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1] = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy() # ValueError: shape mismatch: value array of shape (1856,) could not be broadcast to indexing result of shape (1856,2,30)

        else:
            # 2D+ Position:
            if debug_print:
                max_binned_y_index = np.max(position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1) # 28
                print(f'max_binned_y_index: {max_binned_y_index}')
            output = np.zeros((len(xbin), len(ybin))) # (65, 30)
            output[position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1, position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1] = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy() # ValueError: shape mismatch: value array of shape (1856,) could not be broadcast to indexing result of shape (1856,2,30)

        return output


    @classmethod
    def _compute_avg_speed_at_each_position_bin(cls, active_position_df: pd.DataFrame, xbin: NDArray, ybin: NDArray, debug_print=False):
        """ compute the average speed at each position x. Used only by the two-step decoder above. """
        outputs = dict()
        outputs['speed'] = cls._compute_group_stats_for_var(active_position_df=active_position_df, xbin=xbin, ybin=ybin, variable_name='speed')
        # outputs['velocity_x'] = _compute_group_stats_for_var(active_position_df=active_position_df, xbin=xbin, ybin=ybin, variable_name='velocity_x')
        # outputs['acceleration_x'] = _compute_group_stats_for_var(active_position_df=active_position_df, xbin=xbin, ybin=ybin, variable_name='acceleration_x')
        return outputs['speed']



    @function_attributes(short_name=None, tags=['two-step-decoder', 'compute', 'pure'], input_requires=[], output_provides=[], uses=['cls._compute_avg_speed_at_each_position_bin', 'Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep', 'self.perform_build_marginals'], used_by=[], creation_date='2025-06-30 09:07', related_items=[])
    @classmethod    
    def perform_compute_two_step_decoder(cls, active_decoder: "BasePositionDecoder", active_one_step_result: DecodedFilterEpochsResult, pos_df: pd.DataFrame, debug_print=False):
        """ computes the "two-step" position posterior decoding from the one-step results and decoder

        Heavy lifting performed by `Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep`

        active_one_step_decoder.flat_p_x_given_n.shape # (472, 66151)
        np.shape(active_one_step_decoder.p_x_given_n) # (59, 8, 66151)
        np.shape(active_one_step_decoder.P_x) # (472, 1)
        active_one_step_decoder.original_position_data_shape # (59, 8)
        active_one_step_decoder.flat_position_size # 472


        Usage:

            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder

            pos_df = deepcopy(results2D.pos_df) # computation_result.sess.position.to_dataframe()

            two_step_decoder_result = BasePositionDecoder.perform_compute_two_step_decoder(active_decoder=a_new_global2D_decoder, pos_df=pos_df)

        """
        from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
        from neuropy.core.position import PositionComputedDataMixin
        
        # self = active_decoder: "BasePositionDecoder"
        active_xbins = active_decoder.xbin
        active_ybins = active_decoder.ybin

        active_xbin_centers = active_decoder.xbin_centers
        active_ybin_centers = active_decoder.ybin_centers

        pos_df = pos_df.position.adding_binned_position_columns(xbin_edges=active_xbins, ybin_edges=active_ybins, active_computation_config=None)

        avg_speed_per_pos = active_decoder._compute_avg_speed_at_each_position_bin(pos_df, xbin=active_xbin_centers, ybin=active_ybin_centers, debug_print=debug_print)

        if debug_print:
            print(f'np.shape(avg_speed_per_pos): {np.shape(avg_speed_per_pos)}')

        max_speed = np.nanmax(avg_speed_per_pos)
        # max_speed # 73.80995983236636
        min_speed = np.nanmin(avg_speed_per_pos)
        # min_speed # 0.0
        K_over_V = 60.0 / max_speed # K_over_V = 0.8128984236852197

        # K = 1.0
        K = K_over_V
        V = 1.0
        sigma_t_all = Zhang_Two_Step.sigma_t(avg_speed_per_pos, K, V, d=1.0) # np.shape(sigma_t_all): (64, 29)
        if debug_print:
            print(f'np.shape(sigma_t_all): {np.shape(sigma_t_all)}')

        # normalize sigma_t_all:
        _OLD_two_step_decoder_result = DynamicParameters.init_from_dict({'xbin':active_xbins, 'ybin':active_ybins,
                                                                'avg_speed_per_pos': avg_speed_per_pos,
                                                                'K':K, 'V':V,
                                                                'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
        })

        _OLD_two_step_decoder_result['C'] = 1.0
        _OLD_two_step_decoder_result['k'] = 1.0

        if (active_decoder.ndim < 2):
            assert active_decoder.ndim == 1, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {active_decoder.ndim}"
            print(f'prev_one_step_bayesian_decoder.ndim == 1, so using [0.0] as active_ybin_centers.')
            active_ybin_centers = np.array([0.0])

        else:
            # 2D Case:
            assert active_decoder.ndim == 2, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {active_decoder.ndim}"
            active_ybin_centers = active_decoder.ybin_centers

        # pre-allocate outputs:
        _OLD_two_step_decoder_result['all_x'], _OLD_two_step_decoder_result['flat_all_x'], original_data_shape = Zhang_Two_Step.build_all_positions_matrix(active_decoder.xbin_centers, active_ybin_centers) # all_x: (64, 29, 2), flat_all_x: (1856, 2)
        _OLD_two_step_decoder_result['original_all_x_shape'] = original_data_shape # add the original data shape to the computed data

        # Pre-allocate output:
        active_two_step_result: DecodedFilterEpochsResult = deepcopy(active_one_step_result) ## copy the one-step result
        num_filter_epochs: int = active_one_step_result.num_filter_epochs
        ## Add extra fields to the two-step `DecodedFilterEpochsResult`
        active_two_step_result.all_scaling_factors_k = []
        active_two_step_result.p_x_given_n_and_x_prev_list = []

        for an_epoch_idx in np.arange(num_filter_epochs):
            ## single epoch:
            # flat_p_x_given_n: NDArray = np.atleast_1d(active_two_step_result.p_x_given_n_list[an_epoch_idx].flatten())
            
            curr_single_epoch_result: SingleEpochDecodedResult = active_two_step_result.get_result_for_epoch(active_epoch_idx=an_epoch_idx)
            curr_epoch_p_x_given_n: NDArray = curr_single_epoch_result.p_x_given_n ## flatten all spatial dimensions (all dimension but the last)
            if debug_print:
                print("\tOriginal shape:", np.shape(curr_epoch_p_x_given_n))

            # curr_epoch_active_n_time_bins: int = curr_single_epoch_result.num_time_windows ## (one less than needed
            curr_epoch_active_n_time_bins: int = curr_single_epoch_result.nbins ## (one less than needed
            
            # Flatten all but the last dimension
            new_shape = (-1, curr_epoch_p_x_given_n.shape[-1])
            curr_epoch_flat_p_x_given_n: NDArray = curr_epoch_p_x_given_n.reshape(new_shape)
            # print("Resulting array:\n", flat_p_x_given_n)
            if debug_print:
                print("\tShape:", np.shape(curr_epoch_flat_p_x_given_n))

            # flat_p_x_given_n: NDArray = curr_single_epoch_result.p_x_given_n
            flat_p_x_given_n_and_x_prev = np.full_like(curr_epoch_flat_p_x_given_n, 0.0) # fill with NaNs. 
            p_x_given_n_and_x_prev = np.full_like(curr_epoch_p_x_given_n, 0.0) # fill with NaNs. Pre-allocate output
            
            ## Add extra fields to the two-step result
            active_two_step_result.most_likely_positions_list[an_epoch_idx] = np.zeros((2, curr_epoch_active_n_time_bins)) # (2, 85841)

            if debug_print:
                print(f'np.shape(prev_one_step_bayesian_decoder.p_x_given_n): {np.shape(curr_epoch_p_x_given_n)}')

            all_scaling_factors_k = Zhang_Two_Step.compute_scaling_factor_k(curr_epoch_flat_p_x_given_n)

            # TODO: Efficiency: This will be inefficient, but do a slow iteration. 
            for time_window_bin_idx in np.arange(curr_epoch_active_n_time_bins):
                ## Iterate through all time bins within the epoch:
                curr_t_step_flat_p_x_given_n = curr_epoch_flat_p_x_given_n[:, time_window_bin_idx] # this gets the specific n_t for this time window           
                active_k = all_scaling_factors_k[time_window_bin_idx] # get the specific k value
                     
                # previous positions as determined by the two-step decoder: this uses the two_step previous position instead of the one_step previous position:
                if time_window_bin_idx > 0:
                    ## have a valid previous timestep to use (t-1)
                    prev_x_position = active_two_step_result.most_likely_positions_list[an_epoch_idx][:, time_window_bin_idx-1] # TODO: is this okay for 1D as well?
                    # active_k = two_step_decoder_result['k']
                    if debug_print:
                        print(f'np.shape(prev_x_position): {np.shape(prev_x_position)}')

                    # Flat version:
                    flat_p_x_given_n_and_x_prev[:,time_window_bin_idx] = Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep(curr_t_step_flat_p_x_given_n, prev_x_position, _OLD_two_step_decoder_result['flat_all_x'], _OLD_two_step_decoder_result['flat_sigma_t_all'], _OLD_two_step_decoder_result['C'], active_k) # output shape (1856, )            


                else:
                    # no valid previous timestep because t=0:
                    prev_x_position = None # active_two_step_result.most_likely_positions_list[an_epoch_idx][:, time_window_bin_idx]
                    # Flat version:
                    flat_p_x_given_n_and_x_prev[:,time_window_bin_idx] = deepcopy(curr_t_step_flat_p_x_given_n) ## same as p_x_given_n # output shape (1856, )            

                    
                if (active_decoder.ndim < 2):
                    # 1D case:
                    p_x_given_n_and_x_prev[:,time_window_bin_idx] = flat_p_x_given_n_and_x_prev[:,time_window_bin_idx] # used to be (original_data_shape[0], original_data_shape[1])
                else:
                    # 2D:
                    p_x_given_n_and_x_prev[:,:,time_window_bin_idx] = np.reshape(flat_p_x_given_n_and_x_prev[:,time_window_bin_idx], (original_data_shape[0], original_data_shape[1]))        
            ## END for time_window_bin_idx in np.arange(curr_epoch_active_n_time_bins)...
            
            active_two_step_result.p_x_given_n_and_x_prev_list.append(p_x_given_n_and_x_prev) # (59, 8, 400)
            

            # POST-hoc most-likely computations: Compute the most-likely positions from the p_x_given_n_and_x_prev:
            # # np.shape(self.most_likely_position_indicies) # (2, 85841)
            """ Computes the most likely positions at each timestep from flat_p_x_given_n_and_x_prev """
            most_likely_position_flat_indicies = np.argmax(flat_p_x_given_n_and_x_prev, axis=0)

            # Adds `most_likely_position_flat_max_likelihood_values`:
            # Same as np.amax(x, axis=axis_idx)
            # axis_idx = 0
            # x = two_step_decoder_result['flat_p_x_given_n_and_x_prev']
            # index_array = two_step_decoder_result['most_likely_position_flat_indicies']
            # two_step_decoder_result['most_likely_position_flat_max_likelihood_values'] = np.take_along_axis(x, np.expand_dims(index_array, axis=axis_idx), axis=axis_idx).squeeze(axis=axis_idx)
            most_likely_position_flat_max_likelihood_values = np.take_along_axis(flat_p_x_given_n_and_x_prev, np.expand_dims(most_likely_position_flat_indicies, axis=0), axis=0).squeeze(axis=0) # get the flat maximum values
            if debug_print:
                print(f"\t{most_likely_position_flat_max_likelihood_values.shape = }")
            # Reshape the maximum values:
            # two_step_decoder_result['most_likely_position_max_likelihood_values'] = np.array(np.unravel_index(two_step_decoder_result['most_likely_position_flat_max_likelihood_values'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array

            # np.shape(self.most_likely_position_flat_indicies) # (85841,)
            active_two_step_result.most_likely_position_indicies_list[an_epoch_idx] = np.array(np.unravel_index(most_likely_position_flat_indicies, active_decoder.original_position_data_shape)) # convert back to an array
            # np.shape(self.most_likely_position_indicies) # (2, 85841)
            active_two_step_result.most_likely_positions_list[an_epoch_idx][0, :] = active_decoder.xbin_centers[active_two_step_result.most_likely_position_indicies_list[an_epoch_idx][0, :]]
            if active_decoder.ndim > 1:
                active_two_step_result.most_likely_positions_list[an_epoch_idx][1, :] = active_decoder.ybin_centers[active_two_step_result.most_likely_position_indicies_list[an_epoch_idx][1, :]]
            # two_step_decoder_result['sigma_t_all'] = sigma_t_all # set sigma_t_all                

            ## For some reason we set up the two-step decoder's most_likely_positions with the tranposed shape compared to the one-step decoder:
            active_two_step_result.most_likely_positions_list[an_epoch_idx] = active_two_step_result.most_likely_positions_list[an_epoch_idx].T

            ## Once done, compute marginals for the two-step:
            curr_unit_marginal_x, curr_unit_marginal_y = active_decoder.perform_build_marginals(p_x_given_n_and_x_prev, active_two_step_result.most_likely_positions_list[an_epoch_idx], debug_print=debug_print)
            _OLD_two_step_decoder_result['marginal'] = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
        ## END for i in np.arange(num_filter_epochs)...
        

        return active_two_step_result, _OLD_two_step_decoder_result





# ==================================================================================================================== #
# Bayesian Decoder                                                                                                     #
# ==================================================================================================================== #
@custom_define(slots=False, eq=False)
class BayesianPlacemapPositionDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder): # needs `HDFMixin, `
    """ Holds the placefields. Can be called on any spike data to compute the most likely position given the spike data.

    Used to try to decode everything in one go, meaning it took the parameters (like the time window) and the spikes to decode as well and did the computation internally, but the concept of a decoder is that it is a stateless object that can be called on any spike data to decode it, so this concept is depricated.

    Holds a PfND object in self.pf that is used for decoding.


    Call Hierarchy:
        Path 1:
            .decode_specific_epochs(...)
                BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...)
                    .decode(...)
        Path 2:
            .compute_all(...)
                .hyper_perform_decode(...)
                    .decode(...)
                .compute_corrected_positions(...)

    """
    ## Time Binning:
    time_bin_size: float = None # these are actually required, but not allowed to be missing default values because of the inheritance
    spikes_df: pd.DataFrame = None # these are actually required, but not allowed to be missing default values because of the inheritance

    time_binning_container: BinningContainer = None
    unit_specific_time_binned_spike_counts: np.ndarray = None
    total_spike_counts_per_window: np.ndarray = None

    ## Computed Results:
    flat_p_x_given_n: np.ndarray = None
    p_x_given_n: np.ndarray = None
    most_likely_position_flat_indicies: np.ndarray = None
    most_likely_position_indicies: np.ndarray = None
    marginal: DynamicContainer = None
    most_likely_positions: np.ndarray = None
    revised_most_likely_positions: np.ndarray = None


    # time_binning_container accessors ___________________________________________________________________________________ #
    @property
    def time_window_edges(self):
        return self.time_binning_container.edges
    @property
    def time_window_edges_binning_info(self):
        return self.time_binning_container.edge_info

    @property
    def time_window_centers(self):
        return self.time_binning_container.centers

    @property
    def time_window_center_binning_info(self):
        return self.time_binning_container.center_info

    @property
    def num_time_windows(self):
        """The num_time_windows property."""
        return self.time_window_center_binning_info.num_bins

    @property
    def active_time_windows(self):
        """The num_time_windows property."""        
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        active_time_windows = [(window_starts[i], window_ends[i]) for i in self.time_window_center_binning_info.bin_indicies]
        return active_time_windows
    
    @property
    def active_time_window_centers(self):
        """The active_time_window_centers property are the center timepoints for each window. """        
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        active_window_midpoints = window_starts + ((window_ends - window_starts) / 2.0)
        return active_window_midpoints
    
    @property
    def is_non_firing_time_bin(self):
        """A boolean array that indicates whether each time window has no spikes. Requires self.total_spike_counts_per_window"""
        return (self.total_spike_counts_per_window == 0)
        # return np.where(self.total_spike_counts_per_window == 0)[0]
            

    @classmethod
    def serialized_key_allowlist(cls):
        input_keys = ['time_bin_size', 'pf', 'spikes_df', 'debug_print']
        # intermediate_keys = ['unit_specific_time_binned_spike_counts', 'time_window_edges', 'time_window_edges_binning_info']
        saved_result_keys = ['flat_p_x_given_n']
        return input_keys + saved_result_keys
    
    @classmethod
    def from_dict(cls, val_dict):
        new_obj = BayesianPlacemapPositionDecoder(time_bin_size=val_dict.get('time_bin_size', 0.25), pf=val_dict.get('pf', None), spikes_df=val_dict.get('spikes_df', None), setup_on_init=val_dict.get('setup_on_init', True), post_load_on_init=val_dict.get('post_load_on_init', False), debug_print=val_dict.get('debug_print', False))
        return new_obj

    
    def add_precomputed_decoded_result(self, decoded_result: SingleEpochDecodedResult):
        """ post-hoc adds the result computed 
        """        
        self.flat_p_x_given_n = decoded_result.flat_p_x_given_n
        self.p_x_given_n = deepcopy(decoded_result.p_x_given_n)
        self.most_likely_position_flat_indicies = None
        self.most_likely_position_indicies = deepcopy(decoded_result.most_likely_position_indicies)
        self.most_likely_positions = deepcopy(decoded_result.most_likely_positions)
        
        if decoded_result.marginal_x is not None:
            self.marginal.x = deepcopy(decoded_result.marginal_x)
            self.marginal.x.p_x_given_n_and_x_prev = None
            self.marginal.x.two_step_most_likely_positions_1D = None
            
        if decoded_result.marginal_y is not None:
            self.marginal.y = deepcopy(decoded_result.marginal_y)
            self.marginal.y.p_x_given_n_and_x_prev = None
            self.marginal.y.two_step_most_likely_positions_1D = None
            

        if decoded_result.marginal_z is not None:
            self.marginal.z = deepcopy(decoded_result.marginal_z)
            self.marginal.z.p_x_given_n_and_x_prev = None
            self.marginal.z.two_step_most_likely_positions_1D = None
            
        self.revised_most_likely_positions = None 
        

    # ==================================================================================================================== #
    # Methods                                                                                                              #
    # ==================================================================================================================== #
    
    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_computation_variables()
            self._setup_time_bin_spike_counts_N_i()
            # self._setup_time_window_centers()
            self.p_x_given_n = self._reshape_output(self.flat_p_x_given_n)
            self.compute_most_likely_positions()

    def setup(self):
        # This version should override the base class version to finish the more extended setup of the new properties
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        
        self._setup_computation_variables()
        # Could pre-filter the self.spikes_df by the 
        
        self.time_binning_container = None
        self.unit_specific_time_binned_spike_counts = None
        self.total_spike_counts_per_window = None
        
        self._setup_time_bin_spike_counts_N_i()
        
        # pre-allocate outputs:
        self.flat_p_x_given_n = None # np.zeros((self.flat_position_size, self.num_time_windows))
        self.p_x_given_n = None
        self.most_likely_position_flat_indicies = None
        self.most_likely_position_indicies = None
        self.marginal = None 
        
    def debug_dump_print(self):
        """ dumps the state for debugging purposes """
        variable_names_dict = dict(summary = ['ndim', 'num_neurons', 'num_time_windows'],
            time_variable_names = ['time_bin_size', 'time_window_edges', 'time_window_edges_binning_info', 'time_window_centers','time_window_center_binning_info'],
            binned_spikes = ['unit_specific_time_binned_spike_counts', 'total_spike_counts_per_window'],
            intermediate_computations = ['F', 'P_x'],
            posteriors = ['p_x_given_n'],
            most_likely = ['most_likely_positions'],
            marginals = ['marginal'],
            other_variables = ['neuron_IDXs', 'neuron_IDs']
        )
        for a_category_name, variable_names_list in variable_names_dict.items():
            print(f'# {a_category_name}:')
            # print(f'\t {variable_names_list}:')
            for a_variable_name in variable_names_list:
                a_var_value = getattr(self, a_variable_name)
                a_var_shape = safe_get_variable_shape(a_var_value)
                if a_var_shape is None:
                    if isinstance(a_var_value, (int, float, np.number)):
                        a_var_shape = a_var_value # display the value directly if we can
                    else:
                        a_var_shape = 'SCALAR' # otherwise just output the literal text "SCALAR"
                print(f'\t {a_variable_name}: {a_var_shape}')

        # TODO: Handle marginals:
        # self.marginal.x

    # External Updating __________________________________________________________________________________________________ #

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids, defer_compute_all:bool=False):
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: neuron_sliced_decoder.pf, ... (much more)

            defer_compute_all: bool - should be set to False if you want to manually decode using custom epochs or something later. Otherwise it will compute for all spikes automatically.
        """
        neuron_sliced_decoder = super().get_by_id(ids, defer_compute_all=defer_compute_all)
        ## Recompute:
        if not defer_compute_all:
            neuron_sliced_decoder.compute_all() # does recompute, updating internal variables. TODO EFFICIENCY 2023-03-02 - This is overkill and I could filter the tuning_curves and etc directly, but this is easier for now. 
        return neuron_sliced_decoder

    def conform_to_position_bins(self, target_one_step_decoder, force_recompute=True):
        """ After the underlying placefield (self.pf)'s position bins are changed by calling pf.conform_to_position_bins(...) externally, the computations for the decoder will be messed up (and out of sync).
            Calling this function detects this issue.
            # 2022-12-09 - We want to be able to have both long/short track placefields have the same spatial bins.
            Usage:
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)

            Usage 1D:
                long_pf1D = long_results.pf1D
                short_pf1D = short_results.pf1D
                
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)


            Usage 2D:
                long_pf2D = long_results.pf2D
                short_pf2D = short_results.pf2D
                long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        """
        self, did_recompute = super().conform_to_position_bins(target_one_step_decoder, force_recompute=force_recompute)
        if did_recompute or (self.p_x_given_n.shape[0] < target_one_step_decoder.p_x_given_n.shape[0]):
            self.setup() # re-setup the decoder
            self.compute_all()
            did_recompute = True
        return self, did_recompute


    def add_two_step_decoder_results(self, two_step_decoder_result):
        """ adds the results from the computed two_step_decoder to self (the one_step_decoder)
        ## In this new mode we'll add the two-step properties to the original one-step decoder:
        ## Adds the directly accessible properties to the active_one_step_decoder after they're computed in the active_two_step_decoder so that they can be plotted with the same functions/etc.

        Inputs:
            two_step_decoder_result: computation_result.computed_data[two_step_decoder_key]
 
        """
        
        # None initialize two-step properties on the one_step_decoder:
        self.p_x_given_n_and_x_prev = None
        self.two_step_most_likely_positions = None

        self.marginal.x.p_x_given_n_and_x_prev = None
        self.marginal.x.two_step_most_likely_positions_1D = None

        if self.marginal.y is not None:
            self.marginal.y.p_x_given_n_and_x_prev = None
            self.marginal.y.two_step_most_likely_positions_1D = None

        # Set the two-step properties on the one-step decoder:
        self.p_x_given_n_and_x_prev = two_step_decoder_result.p_x_given_n_and_x_prev.copy()
        self.two_step_most_likely_positions = two_step_decoder_result.most_likely_positions.copy()

        self.marginal.x.p_x_given_n_and_x_prev = two_step_decoder_result.marginal.x.p_x_given_n.copy()
        self.marginal.x.two_step_most_likely_positions_1D = two_step_decoder_result.marginal.x.most_likely_positions_1D.copy()

        if self.marginal.y is not None:
            self.marginal.y.p_x_given_n_and_x_prev = two_step_decoder_result.marginal.y.p_x_given_n.copy()
            self.marginal.y.two_step_most_likely_positions_1D = two_step_decoder_result.marginal.y.most_likely_positions_1D.copy()


    @function_attributes(short_name='to_1D_maximum_projection', tags=['updated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-06 22:06')
    def to_1D_maximum_projection(self, defer_compute_all:bool=True):
        """ returns a copy of the decoder that is 1D """
        # Perform the projection. Can only be ran once.
        new_copy_decoder = super().to_1D_maximum_projection(defer_compute_all=defer_compute_all)
        # new_copy_decoder.setup()
        if not defer_compute_all:
            new_copy_decoder.compute_all() # Needed?
        return new_copy_decoder

    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    def _setup_time_bin_spike_counts_N_i(self, debug_print=False):
        """ updates: 
        
        .time_binning_container
        # .time_window_edges, 
        # .time_window_edges_binning_info
        
        .unit_specific_time_binned_spike_counts
        .total_spike_counts_per_window
        
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        ## need to create new time_window_edges from the self.time_bin_size:
        if debug_print:
            print(f'WARNING: _setup_time_bin_spike_counts_N_i(): updating self.time_window_edges and self.time_window_edges_binning_info ...')

        self.unit_specific_time_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, time_bin_size=self.time_bin_size, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
        if debug_print:
            print(f'from ._setup_time_bin_spike_counts_N_i():')
            print(f'\ttime_window_edges.shape: {time_window_edges.shape}') #  (11882,)
            print(f'unit_specific_time_binned_spike_counts.shape: {self.unit_specific_time_binned_spike_counts.shape}') # (70, 11881)
        
        self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        
        # Here we should filter the outputs by the actual self.neuron_IDXs
        # assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        if np.shape(self.unit_specific_time_binned_spike_counts)[0] > len(self.neuron_IDXs):
            # Drop the irrelevant indicies:
            self.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[self.neuron_IDXs,:] # Drop the irrelevent indicies
        
        assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        self.total_spike_counts_per_window = np.sum(self.unit_specific_time_binned_spike_counts, axis=0) # gets the total number of spikes during each window (across all placefields)

    def _reshape_output(self, output_probability):
        return np.reshape(output_probability, (*self.original_position_data_shape, self.num_time_windows)) # changed for compatibility with 1D decoder
    
    def _flatten_output(self, output):
        """ the inverse of _reshape_output(output) to flatten the position coordinates into a single flat dimension """
        return np.reshape(output, (self.flat_position_size, self.num_time_windows)) # Trying to flatten the position coordinates into a single flat dimension
    
    
    
    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #

    def compute_all(self, debug_print=False):
        """ computes all the outputs of the decoder, and stores them in the class instance 

        Uses:
            self.hyper_perform_decode(...)
            self.compute_corrected_positions(...)

        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        should_use_safe_time_binning: bool = False
        
        if should_use_safe_time_binning:
            original_time_bin_container = deepcopy(self.time_binning_container) ## compared to this, we lose the last time_bin (which is partial)
        
        
        with WrappingMessagePrinter(f'compute_all final_p_x_given_n called. Computing windows...', begin_line_ending='... ', finished_message='compute_all completed.', enable_print=(debug_print or self.debug_print)):
            ## Single sweep decoding:

            ## 2022-09-23 - Epochs-style encoding (that works):
            self.time_binning_container, self.p_x_given_n, self.most_likely_positions, curr_unit_marginal_x, curr_unit_marginal_y, flat_outputs_container = self.hyper_perform_decode(self.spikes_df, decoding_time_bin_size=self.time_bin_size, output_flat_versions=True, debug_print=(debug_print or self.debug_print)) ## this is where it's getting messed up
            if should_use_safe_time_binning:
                num_extra_bins_in_old: int = original_time_bin_container.num_bins - self.time_binning_container.num_bins
                # np.isin(original_time_bin_container.centers, self.time_binning_container.centers)
                # is_time_bin_included_in_new = np.isin(original_time_bin_container.centers, self.time_binning_container.centers) ## see which of the original time bins vanished in the new `time_bin_container`
                ## drop previous values for compatibility
                if num_extra_bins_in_old > 0:
                    ## UPDATES: self.is_non_firing_time_bin, self.unit_specific_time_binned_spike_counts
                    ## find how many time bins to be dropped:
                    # is_time_bin_included_in_new = np.array([np.isin(v, self.time_binning_container.centers) for v in original_time_bin_container.centers])
                    old_time_bins_to_remove = original_time_bin_container.centers[-num_extra_bins_in_old:]
                    assert np.all(np.logical_not([np.isin(v, self.time_binning_container.centers) for v in old_time_bins_to_remove]))
                    self.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[:, :-num_extra_bins_in_old]
                    self.total_spike_counts_per_window = self.total_spike_counts_per_window[:-num_extra_bins_in_old]
                    
            # self._setup_time_bin_spike_counts_N_i(debug_print=True) # updates: self.time_binning_container, self.unit_specific_time_binned_spike_counts, self.total_spike_counts_per_window
            # self.unit_specific_time_binned_spike_counts

            self.marginal = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
            assert isinstance(self.time_binning_container, BinningContainer) # Should be neuropy.utils.mixins.binning_helpers.BinningContainer
            self.compute_corrected_positions() ## this seems to fail for pf1D
            
            ## set flat properties for compatibility (I guess)
            self.flat_p_x_given_n = flat_outputs_container.flat_p_x_given_n
            self.most_likely_position_flat_indicies = flat_outputs_container.most_likely_position_flat_indicies

    def compute_most_likely_positions(self):
        """ Computes the most likely positions at each timestep from self.flat_p_x_given_n """        
        raise NotImplementedError
        self.most_likely_position_flat_indicies, self.most_likely_position_indicies = self.perform_compute_most_likely_positions(self.flat_p_x_given_n, self.original_position_data_shape)
        # np.shape(self.most_likely_position_flat_indicies) # (85841,)
        # np.shape(self.most_likely_position_indicies) # (2, 85841)


    @function_attributes(short_name=None, tags=['BROKEN', 'PROBLEM'], input_requires=['.total_spike_counts_per_window', '.most_likely_positions'], output_provides=[], uses=['._setup_time_bin_spike_counts_N_i'], used_by=[], creation_date='2025-01-14 13:03', related_items=[])
    def compute_corrected_positions(self):
        """ computes the revised most likely positions by taking into account the time-bins that had zero spikes and extrapolating position from the prior successfully decoded time bin
        
        Requires:
            .total_spike_counts_per_window
            .most_likely_positions
            
        Updates:
            .revised_most_likely_positions
            .marginal's .x & .y .revised_most_likely_positions_1D
        


        TODO: CRITICAL: CORRECTNESS: 2022-02-25: This was said not to be working for 1D somewhere else in the code, but I don't know if it's working or not. It doesn't seem to be.
            2025-01-16 04:08 - Indeed, I'm getting errors in the 1D case and it might be the cause of repeatedly bad decoded positions at the end caps
        
        Just recompute `self.is_non_firing_time_bin` --> self.total_spike_counts_per_window
        
        if np.shape(self.unit_specific_time_binned_spike_counts)[0] > len(self.neuron_IDXs):
            # Drop the irrelevant indicies:
            self.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[self.neuron_IDXs,:] # Drop the irrelevent indicies
        
        assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        self.total_spike_counts_per_window = np.sum(self.unit_specific_time_binned_spike_counts, axis=0) # gets the total number of spikes during each window (across all placefields)
        

        self._setup_time_bin_spike_counts_N_i(debug_print=True) # updates: self.time_binning_container, self.unit_specific_time_binned_spike_counts, self.total_spike_counts_per_window

        """
        ## Find the bins that don't have any spikes in them:
        # zero_bin_indicies = np.where(self.total_spike_counts_per_window == 0)[0]
        # is_non_firing_bin = self.is_non_firing_time_bin
        # assert (len(self.is_non_firing_time_bin) == self.num_time_windows), f"len(self.is_non_firing_time_bin): {len(self.is_non_firing_time_bin)}, self.num_time_windows: {self.num_time_windows}" # 2025-01-13 17:43 Added constraint because this is supposed to be correct
        
        should_use_safe_time_binning: bool = False
        if should_use_safe_time_binning:
            if (len(self.is_non_firing_time_bin) != self.num_time_windows):
                ## time windows aren't correct after computing for some reason, call `self._setup_time_bin_spike_counts_N_i()` to recompute them
                print(f'WARN: f"len(self.is_non_firing_time_bin): {len(self.is_non_firing_time_bin)}, self.num_time_windows: {self.num_time_windows}", trying to recompute them....')
                self._setup_time_bin_spike_counts_N_i(debug_print=False) # updates: self.time_binning_container, self.unit_specific_time_binned_spike_counts, self.total_spike_counts_per_window        

            assert (len(self.is_non_firing_time_bin) == self.num_time_windows), f"len(self.is_non_firing_time_bin): {len(self.is_non_firing_time_bin)}, self.num_time_windows: {self.num_time_windows}" # 2025-01-13 17:43 Added constraint because this is supposed to be correct        
        else:
            print(f'WARN: not using safe time binning!')


        is_non_firing_bin = np.where(self.is_non_firing_time_bin)[0] # TEMP: do this to get around the indexing issue. TODO: IndexError: boolean index did not match indexed array along dimension 0; dimension is 11880 but corresponding boolean dimension is 11881
        self.revised_most_likely_positions = self.perform_compute_forward_filled_positions(self.most_likely_positions, is_non_firing_bin=is_non_firing_bin)
        
        if self.marginal is not None:
            _revised_marginals = self.perform_build_marginals(self.p_x_given_n, self.revised_most_likely_positions, debug_print=False) # Stupid way of doing this, but w/e
            if self.marginal.x is not None:
                # self.marginal.x.revised_most_likely_positions_1D = self.perform_compute_forward_filled_positions(self.marginal.x.most_likely_positions_1D, is_non_firing_bin=is_non_firing_bin)
                self.marginal.x.revised_most_likely_positions_1D = _revised_marginals[0].most_likely_positions_1D.copy()
            if self.marginal.y is not None:
                # self.marginal.y.revised_most_likely_positions_1D = self.perform_compute_forward_filled_positions(self.marginal.y.most_likely_positions_1D, is_non_firing_bin=is_non_firing_bin)
                self.marginal.y.revised_most_likely_positions_1D =  _revised_marginals[1].most_likely_positions_1D.copy()

        return self.revised_most_likely_positions

    # ==================================================================================================================== #
    # GridBinDebuggableMixin Conformances                                                                                  #
    # ==================================================================================================================== #
    def get_debug_binning_info(self) -> DebugBinningInfo:
        """Returns relevant debug info about the binning configuration

        Returns:
            DebugBinningInfo: Contains binning dimensions and sizes
        """  
        _obj = self.pf.get_debug_binning_info()
        _obj.nTimeBins = self.num_time_windows
        return _obj

    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #


    # `perform_compute_spike_count_and_firing_rate_normalizations` is the only classmethod not ported over to the new stateless class
    @classmethod
    def perform_compute_spike_count_and_firing_rate_normalizations(cls, pho_custom_decoder):
        """ Computes several different normalizations of binned firing rate and spike counts
        
        Usage:
            pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
            unit_specific_time_binned_outputs = perform_compute_spike_count_and_firing_rate_normalizations(pho_custom_decoder)
            spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple
            
            
        TESTING CODE:
        
            pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
            print(f'most_likely_positions: {np.shape(pho_custom_decoder.most_likely_positions)}') # most_likely_positions: (3434, 2)
            unit_specific_time_binned_outputs = perform_compute_spike_count_and_firing_rate_normalizations(pho_custom_decoder)
            spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple:

            # pho_custom_decoder.unit_specific_time_binned_spike_counts.shape # (64, 1717)
            unit_specific_binned_spike_count_mean = np.nanmean(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
            unit_specific_binned_spike_count_var = np.nanvar(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
            unit_specific_binned_spike_count_median = np.nanmedian(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)

            unit_specific_binned_spike_count_mean
            unit_specific_binned_spike_count_median
            # unit_specific_binned_spike_count_mean.shape # (64, )

        """
        # produces a fraction which indicates which proportion of the window's firing belonged to each unit (accounts for global changes in firing rate (each window is scaled by the toial spikes of all cells in that window)
        unit_specific_time_binned_spike_proportion_global_fr_normalized = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.total_spike_counts_per_window
        unit_specific_time_binned_firing_rate = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.time_window_edges_binning_info.step
        # produces a unit firing rate for each window that accounts for global changes in firing rate (each window is scaled by the firing rate of all cells in that window
        unit_specific_time_binned_firing_rate_global_fr_normalized = unit_specific_time_binned_spike_proportion_global_fr_normalized / pho_custom_decoder.time_window_edges_binning_info.step
        # Return the computed values, leaving the original data unchanged.
        return unit_specific_time_binned_spike_proportion_global_fr_normalized, unit_specific_time_binned_firing_rate, unit_specific_time_binned_firing_rate_global_fr_normalized


from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder import ClusterlessRTCPositionDecoder

