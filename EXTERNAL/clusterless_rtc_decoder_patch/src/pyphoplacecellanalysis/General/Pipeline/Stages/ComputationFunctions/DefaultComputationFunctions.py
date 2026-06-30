from copy import deepcopy
import sys
from typing import List, Optional
from nptyping import NDArray
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step, ClusterlessRTCPositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import ClusterlessDecodingParameters, build_multiunits_from_array, build_multiunits_from_session

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder, computation_precidence_specifying_function, global_function

from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingResult, LeaveOneOutDecodingAnalysisResult, _analyze_leave_one_out_decoding_results ## !!DO_NOT_REMOVE_DILL!! 2023-05-26 - Required to unpickle pipelines, imported just for dill compatibility

# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # For _perform_new_position_decoding_computation

# ### For _perform_recursive_latent_placefield_decoding
# from neuropy.utils import position_util
# from neuropy.core import Position
# from neuropy.analyses.placefields import perform_compute_placefields

"""-------------- Specific Computation Functions to be registered --------------"""

class DefaultComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 1 # must be done after PlacefieldComputations
    _is_global = False

    @computation_precidence_specifying_function(overriden_computation_precidence=-0.1)
    @function_attributes(short_name='lap_direction_determination', tags=['laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-24 13:04', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe(), curr_active_pipeline.computation_results[computation_filter_name].sess.laps.to_dataframe()['is_LR_dir']), is_global=False)
    def _perform_lap_direction_determination(computation_result: ComputationResult, **kwargs):
        """ Adds the 'is_LR_dir' column to the laps dataframe and updates 'lap_dir' if needed.        
        """
        computation_result.sess.laps.update_lap_dir_from_net_displacement(pos_input=computation_result.sess.position) # confirmed in-place
        # computation_result.sess.laps.update_lap_dir_from_net_displacement(pos_input=computation_result.sess.position)
        # curr_sess.laps.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end) # this doesn't make sense for the filtered sessions unfortunately.
        return computation_result # no changes except to the internal sessions
    

    @function_attributes(short_name='position_decoding', tags=['decoding', 'position'],
                          input_requires=["computation_result.computation_config.pf_params.time_bin_size", "computation_result.computed_data['pf1D']", "computation_result.computed_data['pf2D']"], output_provides=["computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"], uses=['BayesianPlacemapPositionDecoder'], used_by=[], creation_date='2023-09-12 17:30', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D_Decoder'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D_Decoder']), is_global=False)
    def _perform_position_decoding_computation(computation_result: ComputationResult, override_decoding_time_bin_size: Optional[float]=None, **kwargs):
        """ Builds the 1D & 2D Placefield Decoder 
        
            ## - [ ] TODO: IMPORTANT!! POTENTIAL_BUG: Should this passed-in spikes_df actually be the filtered spikes_df that was used to compute the placefields in PfND? That would be `prev_output_result.computed_data['pf2D'].filtered_spikes_df`
                TODO: CORRECTNESS: Consider whether spikes_df or just the spikes_df used to compute the pf2D should be passed. Only the cells used to build the decoder should be used to decode, that much is certain.
                    - 2022-09-15: it appears that the 'filtered_spikes_df version' improves issues with decoder jumpiness and general inaccuracy that was present when using the session spikes_df: **previously it was just jumping to random points far from the animal's location and then sticking there for a long time**.
        
        """
        placefield_computation_config = computation_result.computation_config.pf_params # should be a PlacefieldComputationParameters
        if override_decoding_time_bin_size is not None:
            old_time_bin_size = computation_result.computation_config.pf_params.time_bin_size
            print(f'changing computation_result.computation_config.pf_params.time_bin_size from {old_time_bin_size} -> {override_decoding_time_bin_size}')
            did_change: bool = (override_decoding_time_bin_size != old_time_bin_size)
            computation_result.computation_config.pf_params.time_bin_size = override_decoding_time_bin_size
            print(f'\tdid_change: {did_change}')
            
        ## filtered_spikes_df version:
        computation_result.computed_data['pf1D_Decoder'] = BayesianPlacemapPositionDecoder(time_bin_size=placefield_computation_config.time_bin_size, pf=computation_result.computed_data['pf1D'], spikes_df=computation_result.computed_data['pf1D'].filtered_spikes_df.copy(), debug_print=False)
        assert (len(computation_result.computed_data['pf1D_Decoder'].is_non_firing_time_bin) == computation_result.computed_data['pf1D_Decoder'].num_time_windows), f"len(self.is_non_firing_time_bin): {len(computation_result.computed_data['pf1D_Decoder'].is_non_firing_time_bin)}, self.num_time_windows: {computation_result.computed_data['pf1D_Decoder'].num_time_windows}"
        computation_result.computed_data['pf1D_Decoder'].compute_all() # this is what breaks it

        if ('pf2D' in computation_result.computed_data) and (computation_result.computed_data.get('pf2D', None) is not None):
            pf = computation_result.computed_data['pf2D']

            computation_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(time_bin_size=placefield_computation_config.time_bin_size, pf=pf, spikes_df=computation_result.computed_data['pf2D'].filtered_spikes_df.copy(), debug_print=False)
            computation_result.computed_data['pf2D_Decoder'].compute_all() # Changing to fIXED grid_bin_bounds ===> MUCH (10x?) slower than before
        else:
            computation_result.computed_data['pf2D_Decoder'] = None
            
        return computation_result
    

    @function_attributes(short_name='position_decoding_clusterless', tags=['decoding', 'position', 'clusterless'],
                          input_requires=["computation_result.computed_data['pf1D']", "computation_result.computed_data['pf2D']"], output_provides=["computation_result.computed_data['pf1D_ClusterlessDecoder']", "computation_result.computed_data['pf2D_ClusterlessDecoder']"], uses=['ClusterlessRTCPositionDecoder', 'ClusterlessClassifier'], used_by=[], creation_date='2026-06-30 00:00', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data.get('pf1D_ClusterlessDecoder', None), curr_active_pipeline.computation_results[computation_filter_name].computed_data.get('pf2D_ClusterlessDecoder', None)), is_global=False)
    def _perform_clusterless_position_decoding_computation(computation_result: ComputationResult, sampling_frequency_hz: float = 1000.0, multiunits=None, rtc_time=None, clusterless_params: Optional[ClusterlessDecodingParameters] = None, **kwargs):
        """ Builds clusterless 1D & 2D position decoders using replay_trajectory_classification on PfND spatial grids. """
        clusterless_params = clusterless_params if clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=sampling_frequency_hz)
        sess = computation_result.sess

        def _build_decoder_for_pf(pf):
            pos_df = pf.filtered_pos_df
            source_times = pos_df['t'].to_numpy(dtype=float) if 't' in pos_df.columns else pos_df['t_seconds'].to_numpy(dtype=float)
            t_start = float(source_times.min())
            t_end = float(source_times.max())
            if multiunits is not None:
                pf_multiunits, pf_rtc_time = build_multiunits_from_array(multiunits, rtc_time)
            else:
                pf_multiunits, pf_rtc_time = build_multiunits_from_session(sess, clusterless_params.clusterless_sampling_frequency_hz, t_start, t_end, spikes_df=pf.filtered_spikes_df.copy())
            decoder = ClusterlessRTCPositionDecoder(pf=pf, sampling_frequency_hz=clusterless_params.clusterless_sampling_frequency_hz, multiunits=pf_multiunits, rtc_time=pf_rtc_time, clusterless_params=clusterless_params, setup_on_init=True, post_load_on_init=False, debug_print=False)
            decoder.compute_all()
            return decoder

        computation_result.computed_data['pf1D_ClusterlessDecoder'] = _build_decoder_for_pf(computation_result.computed_data['pf1D'])
        if ('pf2D' in computation_result.computed_data) and (computation_result.computed_data.get('pf2D', None) is not None):
            computation_result.computed_data['pf2D_ClusterlessDecoder'] = _build_decoder_for_pf(computation_result.computed_data['pf2D'])
        else:
            computation_result.computed_data['pf2D_ClusterlessDecoder'] = None
        return computation_result
    

    @function_attributes(short_name='position_decoding_two_step', tags=['decoding', 'position', 'two-step'],
                          input_requires=["computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"], output_provides=["computation_result.computed_data['pf1D_TwoStepDecoder']", "computation_result.computed_data['pf2D_TwoStepDecoder']"],
                          uses=['_compute_avg_speed_at_each_position_bin'], used_by=[], creation_date='2023-09-12 17:32', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D_TwoStepDecoder'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D_TwoStepDecoder']), is_global=False)
    def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False, ndim: int=2, **kwargs):
        """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields
        """

        def _subfn_compute_two_step_decoder(active_xbins, active_ybins, prev_one_step_bayesian_decoder, pos_df, computation_config, debug_print=False):
            """ captures debug_print 

            pos_df = computation_result.sess.position.to_dataframe()
            computation_config = computation_result.computation_config
            """
            
            avg_speed_per_pos = _compute_avg_speed_at_each_position_bin(pos_df, computation_config, prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers, debug_print=debug_print)
                    
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
            two_step_decoder_result = DynamicParameters.init_from_dict({'xbin':active_xbins, 'ybin':active_ybins,
                                                                    'avg_speed_per_pos': avg_speed_per_pos,
                                                                    'K':K, 'V':V,
                                                                    'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
            })
            
            two_step_decoder_result['C'] = 1.0
            two_step_decoder_result['k'] = 1.0

            if (prev_one_step_bayesian_decoder.ndim < 2):
                assert prev_one_step_bayesian_decoder.ndim == 1, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {prev_one_step_bayesian_decoder.ndim}"
                print(f'prev_one_step_bayesian_decoder.ndim == 1, so using [0.0] as active_ybin_centers.')
                active_ybin_centers = np.array([0.0])

            else:
                # 2D Case:
                assert prev_one_step_bayesian_decoder.ndim == 2, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {prev_one_step_bayesian_decoder.ndim}"
                active_ybin_centers = prev_one_step_bayesian_decoder.ybin_centers

            # pre-allocate outputs:
            two_step_decoder_result['all_x'], two_step_decoder_result['flat_all_x'], original_data_shape = Zhang_Two_Step.build_all_positions_matrix(prev_one_step_bayesian_decoder.xbin_centers, active_ybin_centers) # all_x: (64, 29, 2), flat_all_x: (1856, 2)
            two_step_decoder_result['original_all_x_shape'] = original_data_shape # add the original data shape to the computed data
    
            # Pre-allocate output:
            two_step_decoder_result['flat_p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.flat_p_x_given_n, 0.0) # fill with NaNs. 
            two_step_decoder_result['p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.p_x_given_n, 0.0) # fill with NaNs. Pre-allocate output
            two_step_decoder_result['most_likely_position_indicies'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows), dtype=int) # (2, 85841)
            two_step_decoder_result['most_likely_positions'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows)) # (2, 85841)
                    
            if debug_print:
                print(f'np.shape(prev_one_step_bayesian_decoder.p_x_given_n): {np.shape(prev_one_step_bayesian_decoder.p_x_given_n)}')
            
            two_step_decoder_result['all_scaling_factors_k'] = Zhang_Two_Step.compute_scaling_factor_k(prev_one_step_bayesian_decoder.flat_p_x_given_n)
            
            # TODO: Efficiency: This will be inefficient, but do a slow iteration. 
            for time_window_bin_idx in np.arange(prev_one_step_bayesian_decoder.num_time_windows):
                curr_flat_p_x_given_n = prev_one_step_bayesian_decoder.flat_p_x_given_n[:, time_window_bin_idx] # this gets the specific n_t for this time window                
                # previous positions as determined by the two-step decoder: this uses the two_step previous position instead of the one_step previous position:
                prev_x_position = two_step_decoder_result['most_likely_positions'][:, time_window_bin_idx-1] # TODO: is this okay for 1D as well?
                active_k = two_step_decoder_result['all_scaling_factors_k'][time_window_bin_idx] # get the specific k value
                # active_k = two_step_decoder_result['k']
                if debug_print:
                    print(f'np.shape(prev_x_position): {np.shape(prev_x_position)}')
                            
                # Flat version:
                two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx] = Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep(curr_flat_p_x_given_n, prev_x_position, two_step_decoder_result['flat_all_x'], two_step_decoder_result['flat_sigma_t_all'], two_step_decoder_result['C'], active_k) # output shape (1856, )            
                

                if (prev_one_step_bayesian_decoder.ndim < 2):
                    # 1D case:
                    two_step_decoder_result['p_x_given_n_and_x_prev'][:,time_window_bin_idx] = two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx] # used to be (original_data_shape[0], original_data_shape[1])
                else:
                    # 2D:
                    two_step_decoder_result['p_x_given_n_and_x_prev'][:,:,time_window_bin_idx] = np.reshape(two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx], (original_data_shape[0], original_data_shape[1]))
                

            # POST-hoc most-likely computations: Compute the most-likely positions from the p_x_given_n_and_x_prev:
            # # np.shape(self.most_likely_position_indicies) # (2, 85841)
            """ Computes the most likely positions at each timestep from flat_p_x_given_n_and_x_prev """
            two_step_decoder_result['most_likely_position_flat_indicies'] = np.argmax(two_step_decoder_result['flat_p_x_given_n_and_x_prev'], axis=0)
            
            # Adds `most_likely_position_flat_max_likelihood_values`:
            # Same as np.amax(x, axis=axis_idx)
            # axis_idx = 0
            # x = two_step_decoder_result['flat_p_x_given_n_and_x_prev']
            # index_array = two_step_decoder_result['most_likely_position_flat_indicies']
            # two_step_decoder_result['most_likely_position_flat_max_likelihood_values'] = np.take_along_axis(x, np.expand_dims(index_array, axis=axis_idx), axis=axis_idx).squeeze(axis=axis_idx)
            two_step_decoder_result['most_likely_position_flat_max_likelihood_values'] = np.take_along_axis(two_step_decoder_result['flat_p_x_given_n_and_x_prev'], np.expand_dims(two_step_decoder_result['most_likely_position_flat_indicies'], axis=0), axis=0).squeeze(axis=0) # get the flat maximum values
            print(f"{two_step_decoder_result['most_likely_position_flat_max_likelihood_values'].shape = }")
            # Reshape the maximum values:
            # two_step_decoder_result['most_likely_position_max_likelihood_values'] = np.array(np.unravel_index(two_step_decoder_result['most_likely_position_flat_max_likelihood_values'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array

            # np.shape(self.most_likely_position_flat_indicies) # (85841,)
            two_step_decoder_result['most_likely_position_indicies'] = np.array(np.unravel_index(two_step_decoder_result['most_likely_position_flat_indicies'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array
            # np.shape(self.most_likely_position_indicies) # (2, 85841)
            two_step_decoder_result['most_likely_positions'][0, :] = prev_one_step_bayesian_decoder.xbin_centers[two_step_decoder_result['most_likely_position_indicies'][0, :]]

            if prev_one_step_bayesian_decoder.ndim > 1:
                two_step_decoder_result['most_likely_positions'][1, :] = prev_one_step_bayesian_decoder.ybin_centers[two_step_decoder_result['most_likely_position_indicies'][1, :]]
            # two_step_decoder_result['sigma_t_all'] = sigma_t_all # set sigma_t_all                

            ## For some reason we set up the two-step decoder's most_likely_positions with the tranposed shape compared to the one-step decoder:
            two_step_decoder_result['most_likely_positions'] = two_step_decoder_result['most_likely_positions'].T
            
            ## Once done, compute marginals for the two-step:
            curr_unit_marginal_x, curr_unit_marginal_y = prev_one_step_bayesian_decoder.perform_build_marginals(two_step_decoder_result['p_x_given_n_and_x_prev'], two_step_decoder_result['most_likely_positions'], debug_print=debug_print)
            two_step_decoder_result['marginal'] = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
            
            return two_step_decoder_result


        if ndim is None:
            ndim = 2 # add the 2D version if no alterantive is passed in.

        if ndim == 1:
            one_step_decoder_key = 'pf1D_Decoder'
            two_step_decoder_key = 'pf1D_TwoStepDecoder'
        elif ndim == 2:
            one_step_decoder_key = 'pf2D_Decoder'
            two_step_decoder_key = 'pf2D_TwoStepDecoder'
        else:
            raise NotImplementedError # dimensionality must be 1 or 2

        # Get the one-step decoder:
        prev_one_step_bayesian_decoder = computation_result.computed_data[one_step_decoder_key]
        ## New 2022-09-15 direct neuropy.utils.mixins.binning_helpers.build_df_discretized_binned_position_columns version:
        computation_result.sess.position.df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(computation_result.sess.position.df, 
            bin_values=(prev_one_step_bayesian_decoder.xbin, prev_one_step_bayesian_decoder.ybin),
            active_computation_config=computation_result.computation_config.pf_params, force_recompute=False, debug_print=debug_print)
        active_xbins = xbin
        active_ybins = ybin

        pos_df = deepcopy(computation_result.sess.position.df).position.drop_dimensions_above(desired_ndim=ndim, inplace=False)  ## drops excess dimensions in-place. Might not fix the prev decode results though:

        computation_result.computed_data[two_step_decoder_key] = _subfn_compute_two_step_decoder(active_xbins, active_ybins, prev_one_step_bayesian_decoder,
                                                                                                  pos_df=pos_df, # computation_result.sess.position.df,
                                                                                                  computation_config=computation_result.computation_config, debug_print=debug_print)
        ## In this new mode we'll add the two-step properties to the original one-step decoder:
        ## Adds the directly accessible properties to the active_one_step_decoder after they're computed in the active_two_step_decoder so that they can be plotted with the same functions/etc.
        prev_one_step_bayesian_decoder.add_two_step_decoder_results(computation_result.computed_data[two_step_decoder_key])

        return computation_result


    @function_attributes(short_name='recursive_latent_pf_decoding', tags=['decoding', 'recursive', 'latent'],
                          input_requires=["computation_result.computation_config.pf_params", "computation_result.computed_data['pf1D']", "computation_result.computed_data['pf2D']", "computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"],
                          output_provides=["computation_result.computed_data['pf1D_RecursiveLatent']", "computation_result.computed_data['pf2D_RecursiveLatent']"],
                           uses=[], used_by=[], creation_date='2023-09-12 17:34', related_items=[],
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D_RecursiveLatent'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D_RecursiveLatent']), is_global=False)
    def _perform_recursive_latent_placefield_decoding(computation_result: ComputationResult, **kwargs):
        """ note that currently the pf1D_Decoders are not built or used. 

        """
        ### For _perform_recursive_latent_placefield_decoding
        from neuropy.utils import position_util
        from neuropy.core import Position
        from neuropy.analyses.placefields import perform_compute_placefields

        def _subfn_build_recurrsive_placefields(active_one_step_decoder, next_order_computation_config, spikes_df=None, pos_df=None, pos_linearization_method='isomap'):
            """ This subfunction is called to produce a specific depth level of placefields. 
            """
            if spikes_df is None:
                spikes_df = active_one_step_decoder.spikes_df
            if pos_df is None:
                pos_df = active_one_step_decoder.pf.filtered_pos_df
            
            def _prepare_pos_df_for_recurrsive_decoding(active_one_step_decoder, pos_df, pos_linearization_method='isomap'):
                """ duplicates pos_df and builds a new pseudo-pos_df for building second-order placefields/decoder 
                pos_df comes in with columns: ['t', 'x', 'y', 'lin_pos', 'speed', 'binned_x', 'binned_y']

                ISSUE/POTENTIAL BUG: 'lin_pos' which is computed for the second-order pos_df seems completely off when compared to the incoming pos_df.
                    - [ ] Actually the 'x' and 'y' values seem pretty off too. Not sure if this is being computed correctly.
                """
                ## Build the new second-order pos_df from the decoded positions:
                active_second_order_pos_df = pd.DataFrame({'t': active_one_step_decoder.active_time_window_centers, 'x': active_one_step_decoder.most_likely_positions[:,0], 'y': active_one_step_decoder.most_likely_positions[:,1]})

                ## Build the linear position for the second-order pos_df:
                _temp_pos_obj = Position(active_second_order_pos_df) # position_util.linearize_position(...) expects a neuropy Position object instead of a raw DataFrame, so build a temporary one to make it happy
                linear_pos = position_util.linearize_position(_temp_pos_obj, method=pos_linearization_method)
                active_second_order_pos_df['lin_pos'] = linear_pos.x
                return active_second_order_pos_df

            def _prepare_spikes_df_for_recurrsive_decoding(active_one_step_decoder, spikes_df):
                """ duplicates spikes_df and builds a new pseudo-spikes_df for building second-order placefields/decoder """
                active_second_order_spikes_df = deepcopy(spikes_df)
                # TODO: figure it out instead of hacking -- Currently just dropping the last time bin because something is off 
                # invalid_timestamp = np.nanmax(active_second_order_spikes_df['binned_time'].astype(int).to_numpy()) # 11881
                # active_second_order_spikes_df = active_second_order_spikes_df[active_second_order_spikes_df['binned_time'].astype(int) < invalid_timestamp] # drop the last time-bin as a workaround
                # backup measured columns because they will be replaced by the new values:
                active_second_order_spikes_df['x_measured'] = active_second_order_spikes_df['x'].copy()
                active_second_order_spikes_df['y_measured'] = active_second_order_spikes_df['y'].copy()
                # spike_binned_time_idx = (active_second_order_spikes_df['binned_time'].astype(int)-1) # subtract one to get to a zero-based index
                spike_binned_time_idx = active_second_order_spikes_df['binned_time'].astype(int) # already a zero-based index
                # replace the x and y measured positions with the most-likely decoded ones for the next round of decoding
                active_second_order_spikes_df['x'] = active_one_step_decoder.most_likely_positions[spike_binned_time_idx.to_numpy(),0] # x-pos
                active_second_order_spikes_df['y'] = active_one_step_decoder.most_likely_positions[spike_binned_time_idx.to_numpy(),1] # y-pos
                return active_second_order_spikes_df

            def _next_order_decode(active_pf_1D, active_pf_2D, pf_computation_config):
                """ performs the actual decoding given the"""
                ## 1D Decoder
                new_decoder_pf1D = active_pf_1D
                new_1D_decoder_spikes_df = new_decoder_pf1D.filtered_spikes_df.copy()
                new_1D_decoder = BayesianPlacemapPositionDecoder(time_bin_size=pf_computation_config.time_bin_size, pf=new_decoder_pf1D, spikes_df=new_1D_decoder_spikes_df, debug_print=False) 
                new_1D_decoder.compute_all() #  --> TODO: NOTE: 1D .compute_all() has just been recently added due to a previous error in ffill

                ## Custom Manual 2D Decoder:
                new_decoder_pf2D = active_pf_2D # 
                new_decoder_spikes_df = new_decoder_pf2D.filtered_spikes_df.copy()
                new_2D_decoder = BayesianPlacemapPositionDecoder(time_bin_size=pf_computation_config.time_bin_size, pf=new_decoder_pf2D, spikes_df=new_decoder_spikes_df, debug_print=False)
                new_2D_decoder.compute_all() #  --> n = self.
                
                return new_1D_decoder, new_2D_decoder

            active_second_order_spikes_df = _prepare_spikes_df_for_recurrsive_decoding(active_one_step_decoder, spikes_df)
            active_second_order_pos_df = _prepare_pos_df_for_recurrsive_decoding(active_one_step_decoder, pos_df)
            active_second_order_pf_1D, active_second_order_pf_2D = perform_compute_placefields(active_second_order_spikes_df, Position(active_second_order_pos_df),
                                                                                            next_order_computation_config.pf_params, None, None, included_epochs=None, should_force_recompute_placefields=True)
            # build the second_order decoders:
            active_second_order_1D_decoder, active_second_order_2D_decoder = _next_order_decode(active_second_order_pf_1D, active_second_order_pf_2D, next_order_computation_config.pf_params) 
            
            return active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder

        pos_linearization_method='isomap'
        prev_one_step_bayesian_decoder = computation_result.computed_data['pf2D_Decoder']

        ## Builds a duplicate of the current computation config but sets the speed_thresh to 0.0:
        next_order_computation_config = deepcopy(computation_result.computation_config) # make a deepcopy of the active computation config
        next_order_computation_config.pf_params.speed_thresh = 0.0 # no speed thresholding because the speeds aren't real for the second-order fields

        enable_pf1D = True # Enabled on 2022-12-15 afte fixing bug in implementaion of ffill
        enable_pf2D = True

        # Start with empty lists, which will accumulate the different levels of recurrsive depth:
        computation_result.computed_data['pf1D_RecursiveLatent'] = []
        computation_result.computed_data['pf2D_RecursiveLatent'] = []

        # 1st Order:
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':computation_result.computed_data['pf1D'], 'pf1D_Decoder':computation_result.computed_data.get('pf1D_Decoder', {})}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':computation_result.computed_data['pf2D'], 'pf2D_Decoder':computation_result.computed_data['pf2D_Decoder']}))

        # 2nd Order:
        active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder = _subfn_build_recurrsive_placefields(prev_one_step_bayesian_decoder, next_order_computation_config=next_order_computation_config, spikes_df=prev_one_step_bayesian_decoder.spikes_df, pos_df=prev_one_step_bayesian_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_second_order_pf_1D, 'pf1D_Decoder':active_second_order_1D_decoder}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_second_order_pf_2D, 'pf2D_Decoder':active_second_order_2D_decoder}))

        # # 3rd Order:
        active_third_order_pf_1D, active_third_order_pf_2D, active_third_order_1D_decoder, active_third_order_2D_decoder = _subfn_build_recurrsive_placefields(active_second_order_2D_decoder, next_order_computation_config=next_order_computation_config, spikes_df=active_second_order_2D_decoder.spikes_df, pos_df=active_second_order_2D_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_third_order_pf_1D, 'pf1D_Decoder':active_third_order_1D_decoder}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_third_order_pf_2D, 'pf2D_Decoder':active_third_order_2D_decoder}))

        return computation_result


    @function_attributes(short_name='_perform_specific_epochs_decoding', tags=['BasePositionDecoder', 'computation', 'decoder', 'epoch'],
                          input_requires=[ "computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"], output_provides=["computation_result.computed_data['specific_epochs_decoding']"],
                          uses=[], used_by=[], creation_date='2023-04-07 02:16',
        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['specific_epochs_decoding']), is_global=False)
    def _perform_specific_epochs_decoding(computation_result: ComputationResult, active_config, decoder_ndim:int=2, filter_epochs='ripple', decoding_time_bin_size=0.02, **kwargs):
        """ TODO: meant to be used by `_display_plot_decoded_epoch_slices` but needs a smarter way to cache the computations and etc. 
        Eventually to replace `pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError._compute_specific_decoded_epochs`

        Usage:
            ## Test _perform_specific_epochs_decoding
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import DefaultComputationFunctions
            computation_result = curr_active_pipeline.computation_results['maze1_PYR']
            computation_result = DefaultComputationFunctions._perform_specific_epochs_decoding(computation_result, curr_active_pipeline.active_configs['maze1_PYR'], filter_epochs='ripple', decoding_time_bin_size=0.02)
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = computation_result.computed_data['specific_epochs_decoding'][('Ripples', 0.02)]

        """
        ## BEGIN_FUNCTION_BODY _perform_specific_epochs_decoding:
        ## Check for previous computations:
        needs_compute = True # default to needing to recompute.
        computation_tuple_key = (filter_epochs, decoding_time_bin_size, decoder_ndim) # used to be (default_figure_name, decoding_time_bin_size) only

        curr_result = computation_result.computed_data.get('specific_epochs_decoding', {})
        found_result = curr_result.get(computation_tuple_key, None)
        if found_result is not None:
            # Unwrap and reuse the result:
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = found_result # computation_result.computed_data['specific_epochs_decoding'][('Laps', decoding_time_bin_size)]
            needs_compute = False # we don't need to recompute

        if needs_compute:
            ## Do the computation:
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, decoder_ndim=decoder_ndim)

            ## Cache the computation result via the tuple key: (default_figure_name, decoding_time_bin_size) e.g. ('Laps', 0.02) or ('Ripples', 0.02)
            curr_result[computation_tuple_key] = (filter_epochs_decoder_result, active_filter_epochs, default_figure_name)

        computation_result.computed_data['specific_epochs_decoding'] = curr_result
        return computation_result

    # @function_attributes(short_name='', tags=['radon_transform','epoch','replay','decoding','UNFINISHED'], input_requires=[], output_provides=[], uses=['compute_radon_transforms'], used_by=[], creation_date='2023-05-31 12:25')
    # def _perform_decoded_replay_fit_best_line_computation(computation_result: ComputationResult, **kwargs):
    #     """ Radon Transform
    #     """
    #     # TODO: does this need to be a global function since there aren't decodings specifically for the epochs in a given session?
    #     epochs_linear_fit_df, *extra_outputs = compute_radon_transforms(long_results_obj.original_1D_decoder, long_results_obj.all_included_filter_epochs_decoder_result)
    #     epochs_linear_fit_df
        
    #     ## TODO UNFINISHED 2023-05-31: need to add the result to the computation result:
        
    #     return computation_result


# ==================================================================================================================== #
# Private Methods                                                                                                      #
# ==================================================================================================================== #
#TODO 2025-02-18 09:21: - [ ] Moved {'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.KnownFilterEpochs': 'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.KnownFilterEpochs'}

@function_attributes(short_name='_subfn_compute_decoded_epochs', tags=['BasePositionDecoder'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-07 02:15')
def _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs='ripple', decoding_time_bin_size=0.02, decoder_ndim:int=2):
    """ compuites a plot with the 1D Marginals either (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. 
    
    It determines which epochs are being referred to (enabling specifying them by a simple string identifier, like 'ripple', 'pbe', or 'laps') and then gets the coresponding data that's needed to recompute the decoded data for them.
    This decoding is done by calling:
        active_decoder.decode_specific_epochs(...) which returns a result that can then be plotted.

    Used by: _perform_specific_epochs_decoding

    # TODO: 2022-12-20: Need to convert to work with 1D
    
    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import KnownFilterEpochs
    
    default_figure_name = 'stacked_epoch_slices_matplotlib_subplots'
    min_epoch_included_duration = decoding_time_bin_size * float(2) # 0.06666 # all epochs shorter than min_epoch_included_duration will be excluded from analysis
    active_filter_epochs, default_figure_name, epoch_description_list = KnownFilterEpochs.process_functionList(sess=computation_result.sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration, default_figure_name=default_figure_name)

    ## BEGIN_FUNCTION_BODY _subfn_compute_decoded_epochs:
    if decoder_ndim is None:
        decoder_ndim = 2 # add the 2D version if no alterantive is passed in.
    if decoder_ndim == 1:
        one_step_decoder_key = 'pf1D_Decoder'
        # two_step_decoder_key = 'pf1D_TwoStepDecoder'

    elif decoder_ndim == 2:
        one_step_decoder_key = 'pf2D_Decoder'
        # two_step_decoder_key = 'pf2D_TwoStepDecoder'
    else:
        raise NotImplementedError # dimensionality must be 1 or 2
   
    active_decoder = computation_result.computed_data[one_step_decoder_key]
    filter_epochs_decoder_result = active_decoder.decode_specific_epochs(computation_result.sess.spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
    filter_epochs_decoder_result.epoch_description_list = epoch_description_list
    return filter_epochs_decoder_result, active_filter_epochs, default_figure_name


def compute_radon_transforms(decoder: "BasePositionDecoder", decoder_result: "DecodedFilterEpochsResult", nlines:int=8192, margin=16, jump_stat=None, n_jobs:int=1) -> pd.DataFrame:
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import perform_compute_radon_transforms

    """
    active_posterior = decoder_result.p_x_given_n_list # one for each epoch
    
    xbin_centers = deepcopy(decoder.xbin_centers) # the same for all xbins
    t_bin_centers: List[NDArray] = deepcopy(decoder_result.time_window_centers) # list of the time_bin_centers for each decoded time bin in each decoded epoch (list of length n_epochs)
    t0s_list: List[float] = [float(a_t_bin_centers[0]) for a_t_bin_centers in t_bin_centers] # the first time for each decoded posterior.


    # the size of the x_bin in [cm]
    if decoder.pf.bin_info is not None:
        pos_bin_size = float(decoder.pf.bin_info['xstep'])
    else:
        ## if the bin_info is for some reason not accessible, just average the distance between the bin centers.
        pos_bin_size = np.diff(decoder.pf.xbin_centers).mean()

    return perform_compute_radon_transforms(active_posterior=active_posterior, x0=float(xbin_centers[0]), t0=t0s_list, decoding_time_bin_duration=decoder_result.decoding_time_bin_size, pos_bin_size=pos_bin_size, nlines=nlines, margin=margin, jump_stat=jump_stat, n_jobs=n_jobs)



@function_attributes(short_name=None, tags=['radon-transform','decoder','line','fit','velocity','speed'], input_requires=[], output_provides=[], uses=['get_radon_transform'], used_by=['_perform_decoded_replay_fit_best_line_computation'], creation_date='2023-05-31 19:55', related_items=[])
def perform_compute_radon_transforms(active_posterior, x0, t0, decoding_time_bin_duration: float, pos_bin_size:float, nlines=8192, margin=16, jump_stat=None, n_jobs=4, enable_return_neighbors_arr=False) -> pd.DataFrame:
    """ 2023-05-25 - Computes the line of best fit (which gives the velocity) for the 1D Posteriors for each replay epoch using the Radon Transform approch.
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import compute_radon_transforms
        epochs_linear_fit_df = compute_radon_transforms(long_results_obj.original_1D_decoder, long_results_obj.all_included_filter_epochs_decoder_result)
        
        a_directional_laps_filter_epochs_decoder_result = a_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
        laps_radon_transform_df = compute_radon_transforms(a_directional_pf1D_Decoder, a_directional_laps_filter_epochs_decoder_result)

        Columns:         ['score', 'velocity', 'intercept', 'speed']
    """
    assert isinstance(decoding_time_bin_duration, (float, int)), f"second argument should be the decoding_time_bin_duration (as a float, in seconds). Did you mean to call the `compute_radon_transforms(decoder, decoder_result, ...)` version?"
    from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import get_radon_transform
    ## compute the Radon transform to get the lines of best fit
    extra_outputs = []
    score, velocity, intercept, *extra_outputs = get_radon_transform(active_posterior, decoding_time_bin_duration=decoding_time_bin_duration, pos_bin_size=pos_bin_size, posteriors=None, nlines=nlines, margin=margin, jump_stat=jump_stat, n_jobs=n_jobs, enable_return_neighbors_arr=enable_return_neighbors_arr, x0=x0, t0=t0)
    epochs_linear_fit_df = pd.DataFrame({'score': score, 'velocity': velocity, 'intercept': intercept, 'speed': np.abs(velocity)})
    return epochs_linear_fit_df, *extra_outputs




# ==================================================================================================================== #
# Average Speed/Acceleration per Position Bin                                                                          #
# ==================================================================================================================== #
def _subfn_compute_group_stats_for_var(active_position_df, xbin, ybin, variable_name:str = 'speed', debug_print=False):
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


def _compute_avg_speed_at_each_position_bin(active_position_df, active_pf_computation_config, xbin, ybin, show_plots=False, debug_print=False):
    """ compute the average speed at each position x. Used only by the two-step decoder above. """
    outputs = dict()
    outputs['speed'] = _subfn_compute_group_stats_for_var(active_position_df, xbin, ybin, 'speed')
    # outputs['velocity_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'velocity_x')
    # outputs['acceleration_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'acceleration_x')
    return outputs['speed']

