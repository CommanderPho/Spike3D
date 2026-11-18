```python
# active_epoch_names = epoch_names
active_epoch_names = ['roam', 'sprinkle' ]
fine_decoding_time_bin_size: float = 0.025
for an_epoch_name in active_epoch_names:    
    try:
        print(f'\ttrying `.masked_container.final_refine_single_decoder_result_masks(...)` for an_epoch_name: "{an_epoch_name}"...')
        if an_epoch_name not in masked_container.debug_computed_dict:
            masked_container.debug_computed_dict[an_epoch_name] = {}
        
        # active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = a_masked_container.final_refine_single_decoder_result_masks(curr_active_pipeline=owning_pipeline_reference, decoding_time_bin_size=time_bin_size, an_epoch_name=an_epoch_name)
        active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container.final_refine_single_decoder_result_masks(curr_active_pipeline=curr_active_pipeline, fine_decoding_t_bin_size=fine_decoding_time_bin_size, a_decoder_name=an_epoch_name)
        masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
    except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
        print(f'\t\tWARN: the `enable_filter_and_final_result_processing` part of `perform_predictive_decoding_analysis(...) failed with error: {e}. Skipping.')
        raise # pass
    
    except Exception as e:
        raise
## END for an_epoch_name in epoch_names...

# step: 0.0004381917184223964
# 68m 40.4s - FAILED

```


```
	trying `.masked_container.final_refine_single_decoder_result_masks(...)` for an_epoch_name: "roam"...
Running in parallel: max_workers=4, use_parallel=True
	finished with for epochs loops doing locality recomputations
step: 0.0243439843567998
```

```
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
Cell In[13], line 11
      8         masked_container.debug_computed_dict[an_epoch_name] = {}
     10     # active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = a_masked_container.final_refine_single_decoder_result_masks(curr_active_pipeline=owning_pipeline_reference, decoding_time_bin_size=time_bin_size, an_epoch_name=an_epoch_name)
---> 11     active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container.final_refine_single_decoder_result_masks(curr_active_pipeline=curr_active_pipeline, fine_decoding_t_bin_size=fine_decoding_time_bin_size, a_decoder_name=an_epoch_name)
     12     masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
     13 except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:3952, in PredictiveDecodingComputationsContainer.final_refine_single_decoder_result_masks(self, curr_active_pipeline, fine_decoding_t_bin_size, a_decoder_name, **kwargs)
   3948 print(f'step: {step}')
   3950 # decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PosteriorPeaksPeakProminence2dResult.init_from_old_PeakProminence2D_result_dict(active_peak_prominence_2d_results=old_prom_2d_result)
-> 3952 decoded_epoch_t_bins_promenence_result_obj: PosteriorPeaksPeakProminence2dResult = PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(p_x_given_n_list=a_masked_result.p_x_given_n_list, 
   3953     **xybin_edges_kwargs,
   3954     **xybin_centers_only_kwargs,
   3955     # xbin_centers=self.predictive_decoding.xbin_centers, 
   3956     # ybin_centers=self.predictive_decoding.ybin_centers,
   3957     step=step, minimum_included_peak_height=None, # 1m 42s - 7m 1s
   3958     # step=1e-2, minimum_included_peak_height=1e-5, # 47.3s
   3959     peak_height_multiplier_probe_levels=slice_level_multipliers,
   3960     should_use_faster_compute_single_slab_implementation=should_use_faster_compute_single_slab_implementation,
   3961     min_considered_promenence=1e-11,
   3962     parallel=use_parallel, max_workers=max_workers,
   3963     # parallel=True, max_workers=4,
   3964     # parallel=True, max_workers=None,    
   3965 )
   3966 ## 55m - step=1e-4, minimum_included_peak_height=1e-5
   3967 ## 11m - step=1e-3, minimum_included_peak_height=1e-5,
   3969 assert decoded_epoch_t_bins_promenence_result_obj is not None

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\peak_prominence2d.py:1974, in PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(cls, p_x_given_n_list, xbin_centers, ybin_centers, xbin, ybin, step, peak_height_multiplier_probe_levels, minimum_included_peak_height, min_considered_promenence, uniform_blur_size, gaussian_blur_sigma, debug_print, parallel, max_workers, should_use_faster_compute_single_slab_implementation)
   1964             fut = ex.submit(
   1965                 active_compute_single_posterior_slab_fn,
   1966                 epoch_idx, t_idx, slab,
   (...)
   1970                 debug_print
   1971             )
   1972             futures.append(fut)
-> 1974         results_list = [fut.result() for fut in as_completed(futures)]
   1975 else:
   1976     # Serial fallback: identical behavior to original implementation, but via helper
   1977     results_list = []

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\peak_prominence2d.py:1974, in <listcomp>(.0)
   1964             fut = ex.submit(
   1965                 active_compute_single_posterior_slab_fn,
   1966                 epoch_idx, t_idx, slab,
   (...)
   1970                 debug_print
   1971             )
   1972             futures.append(fut)
-> 1974         results_list = [fut.result() for fut in as_completed(futures)]
   1975 else:
   1976     # Serial fallback: identical behavior to original implementation, but via helper
   1977     results_list = []

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\concurrent\futures\_base.py:245, in as_completed(fs, timeout)
    240     if wait_timeout < 0:
    241         raise TimeoutError(
    242                 '%d (of %d) futures unfinished' % (
    243                 len(pending), total_futures))
--> 245 waiter.event.wait(wait_timeout)
    247 with waiter.lock:
    248     finished = waiter.finished_futures

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\threading.py:581, in Event.wait(self, timeout)
    579 signaled = self._flag
    580 if not signaled:
--> 581     signaled = self._cond.wait(timeout)
    582 return signaled

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\threading.py:312, in Condition.wait(self, timeout)
    310 try:    # restore state no matter what (e.g., KeyboardInterrupt)
    311     if timeout is None:
--> 312         waiter.acquire()
    313         gotit = True
    314     else:

KeyboardInterrupt:
```