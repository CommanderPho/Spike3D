---
title: Explain features related to 2D Decoded Posterior frames snapshots by subdivision
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/218fe8f0-4f6e-4fcf-a87f-0be63c6e1ae8
url: https://phohale.sourcegraph.app/deepsearch/299
created: 2025-12-12T11:07:25Z
updated: 2026-07-22T17:39:57Z
---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoPlaceCellAnalysis Explain the current state of the features related to "2D Decoded Posterior frames/snapshots by subdivision"

Based on my investigation of both repositories, here's the current state of the "2D Decoded Posterior frames/snapshots by subdivision" feature:

## Current Implementation Status

The feature is **functional but still being refined**. It was developed in February 2025 and has undergone several iterations with the most recent updates through mid-2025.

### Core Components

#### 1. **Data Structure** ([DecodingResultND](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L534-L545))
- Stores 2D decoded posterior results with frame division metadata
- Key fields:
  - `frame_divided_epochs_df`: DataFrame containing subdivided time epochs
  - `frame_divided_epochs_results`: Decoded results for each frame
  - Renamed from "subdivision" to "frame_division" on 2025-02-20 for clarity

#### 2. **Frame Division Builder** ([build_frame_divided_epochs](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L357-L450))
- Splits the global session timeline into fixed-duration "frames"
- Default `frame_divide_bin_size`: 0.5-1.0 seconds
- Creates epochs with `global_frame_division_idx` tracking
- Adds position data with frame offsets for visualization

#### 3. **Visualization Helper** ([SingleArtistMultiEpochBatchHelpers](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L59-L500))
Created Feb 2025, this is the main plotting class with three key methods:

**a. Data Preparation:**
- `shared_build_flat_stacked_data()`: Reshapes 2D posteriors into flat arrays for timeline display
- Handles coordinate transformations and axis rotations

**b. Track Components (all working):**
- `add_track_shapes()`: Draws repeated track geometry along timeline
- `add_track_positions()`: Plots measured positions with color-coded time progression
- `add_position_posteriors()`: Renders 2D decoded posteriors as heatmap "frames"

### Current Capabilities

✅ **Working:**
1. Automatic frame division of global session timeline
2. Decoding 2D posteriors for each frame subdivision
3. Stacking/flattening posteriors for 2D timeline display
4. Color-coded position trajectories (green→black over time)
5. Track shape repetition along timeline
6. Frame separator visualization (white vertical lines)
7. Integration with Spike2DRaster timeline plots

✅ **Recent Fixes (Feb-Mar 2025):**
- Fixed `num_filter_epochs` calculation bug
- Completed position plotting with NaN-separated frame boundaries
- Added color gradient mapping per frame division
- Proper coordinate scaling and axis rotation

### Known Limitations & Ongoing Work

⚠️ **Areas Still Being Refined:**

1. **Coordinate Mapping**: Some trial-and-error evident in [notebook code](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/SCRATCH/NotebookExtractions_2025-02-21.ipynb?L2821-L3150) around scaling between data/unit/axes coordinates

2. **Performance**: Processing large sessions with many frames can be slow (noted as taking ~4.5 minutes in comments)

3. **Synchronization**: Track plots don't always sync properly with timeline windows when using different axes (noted in usage comments)

4. **Incomplete Features**:
   - Masking/filtering capabilities partially implemented
   - Some export functionality still in progress

### Usage Pattern

The typical workflow from [NotebookExtractions_2025-02-21.ipynb](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/SCRATCH/NotebookExtractions_2025-02-21.ipynb?L2900-L3055):

```python
# Build frame divisions
frame_divide_bin_size = 0.5
(global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df = 
    Compute_NonPBE_Epochs.build_frame_divided_epochs(curr_active_pipeline, frame_divide_bin_size)

# Decode each frame
frame_divided_epochs_specific_decoded_results_dict = 
    {name: decoder.decode_specific_epochs(..., filter_epochs=global_frame_divided_epochs_obj)}

# Visualize with batch helper
batch_plot_helper = SingleArtistMultiEpochBatchHelpers(
    results2D=results2D, active_ax=track_ax, 
    frame_divide_bin_size=frame_divide_bin_size
)
plots_data = batch_plot_helper.add_all_track_plots(global_session=global_session)
```

### Summary

The feature is **production-ready for basic use** but still sees active refinement. The core functionality—subdividing time, decoding 2D posteriors per frame, and visualizing them on a timeline—works. The main ongoing work involves polish: performance optimization, coordinate mapping edge cases, and integration refinements with the broader visualization system.

The diagram above illustrates the data flow from session timeline through frame division, decoding, and final visualization assembly.

## Sources

- [repo:github.com/CommanderPho/Spike3D 2D decoded posterior frames snapshots subdivision](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FSpike3D%202D%20decoded%20posterior%20frames%20snapshots%20subdivision)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis 2D decoded posterior frames snapshots subdivision](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%202D%20decoded%20posterior%20frames%20snapshots%20subdivision)
- [repo:github.com/CommanderPho/Spike3D decoded_frames OR DecodedFrames OR decoded_posterior](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FSpike3D%20decoded_frames%20OR%20DecodedFrames%20OR%20decoded_posterior)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis decoded_frames OR DecodedFrames OR decoded_posterior](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20decoded_frames%20OR%20DecodedFrames%20OR%20decoded_posterior)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis subdivision](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20subdivision)
- [repo:github.com/CommanderPho/Spike3D subdivision](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FSpike3D%20subdivision)
- [github.com/CommanderPho/Spike3D/SCRATCH/NotebookExtractions_2025-02-21.ipynb@079aeb8?L2821-2900](/github.com/CommanderPho/Spike3D@079aeb8b87cd0b02ee64393f77834411fd28a7f3/-/blob/SCRATCH/NotebookExtractions_2025-02-21.ipynb?L2821-2900)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py@26f673d?L534-560](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L534-560)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py@26f673d?L68-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L68-100)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis frame_divided_epochs OR frame_division](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20frame_divided_epochs%20OR%20frame_division)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis SingleArtistMultiEpochBatchHelpers](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20SingleArtistMultiEpochBatchHelpers)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis frame_divided_epochs_specific_decoded_results compute build](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20frame_divided_epochs_specific_decoded_results%20compute%20build)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py@26f673d?L357-450](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L357-450)
- [type:commit (message:frame_divided OR message:subdivision OR message:"2D decoded posterior") repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2025-02-01 count:10](/search?q=type:commit%20%28message:frame_divided%20OR%20message:subdivision%20OR%20message:%222D%20decoded%20posterior%22%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2025-02-01%20count:10)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L59-500](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L59-500)
- [type:commit (message:SingleArtistMultiEpochBatchHelpers OR message:"2D decoded" OR message:frame) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2025-02-10 count:15](/search?q=type:commit%20%28message:SingleArtistMultiEpochBatchHelpers%20OR%20message:%222D%20decoded%22%20OR%20message:frame%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2025-02-10%20count:15)
- [repo:github.com/CommanderPho/Spike3D SingleArtistMultiEpochBatchHelpers](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FSpike3D%20SingleArtistMultiEpochBatchHelpers)
- [repo:github.com/CommanderPho/Spike3D SingleArtist batch plotting 2D posterior](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FSpike3D%20SingleArtist%20batch%20plotting%202D%20posterior)
- [github.com/CommanderPho/Spike3D/SCRATCH/NotebookExtractions_2025-02-21.ipynb@079aeb8?L2900-3150](/github.com/CommanderPho/Spike3D@079aeb8b87cd0b02ee64393f77834411fd28a7f3/-/blob/SCRATCH/NotebookExtractions_2025-02-21.ipynb?L2900-3150)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L775-950](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L775-950)
- [type:commit (message:complete OR message:working OR message:fixed) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2025-02-01 count:5](/search?q=type:commit%20%28message:complete%20OR%20message:working%20OR%20message:fixed%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2025-02-01%20count:5)

## Suggested Follow-ups

- What are the best practices for optimizing the performance of the 2D decoded posterior frame visualization, especially for large datasets?
- Can you elaborate on the coordinate mapping issues encountered during the development of this feature and how they were addressed?


---

# Trying to generate a basic plot, I'm encountering ```---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
Cell In[28], line 1
----> 1 batch_plot_helper.shared_build_flat_stacked_data(force_recompute=True, debug_print=True)
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:234, in SingleArtistMultiEpochBatchHelpers.shared_build_flat_stacked_data(self, debug_print, should_expand_first_dim, force_recompute, **kwargs)
    232 if debug_print:
    233     print(f'desired_epoch_start_idx: {self.desired_epoch_start_idx}, desired_epoch_end_idx: {self.desired_epoch_end_idx}')
--> 234     print(f'desired_start_time_seconds: {self.desired_start_time_seconds}, desired_end_time_seconds: {self.desired_end_time_seconds}')
    236 ## finalize building the data for single-artist plotting (does not plot anything)
    237 (self.n_xbins, self.n_ybins, self.n_tbins), (self.flattened_n_xbins, self.flattened_n_ybins, self.flattened_n_tbins), (self.stacked_p_x_given_n, self.stacked_flat_time_bin_centers, self.stacked_flat_xbin_centers, self.stacked_flat_ybin_centers) = self.complete_build_stacked_flat_arrays(a_result=self.a_result2D, a_new_global_decoder=self.a_new_global2D_decoder,
    238                                                                                                                                                                                                             desired_epoch_start_idx=self.desired_epoch_start_idx, desired_epoch_end_idx=self.desired_epoch_end_idx,
    239                                                                                                                                                                                                             rotate_to_vertical=self.rotate_to_vertical, should_expand_first_dim=should_expand_first_dim)
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:182, in SingleArtistMultiEpochBatchHelpers.desired_end_time_seconds(self)
    180     return self.frame_divide_bin_size * self.desired_epoch_end_idx
    181 else:
--> 182     return self.frame_divide_bin_size * (self.num_filter_epochs-1)
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:158, in SingleArtistMultiEpochBatchHelpers.num_filter_epochs(self)
    155 @property
    156 def num_filter_epochs(self) -> int:
    157     """number of frame_division epochs."""
--> 158     return self.a_result2D.num_filter_epochs
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:167, in SingleArtistMultiEpochBatchHelpers.a_result2D(self)
    165 @property
    166 def a_result2D(self) -> DecodedFilterEpochsResult:
--> 167     return self.results2D.frame_divided_epochs_results[self.active_epoch_name]
KeyError: 'global'``` what is this about?

The issue is clear: the code expects a key `'global'` in the `frame_divided_epochs_results` dictionary, but it's not there. This typically happens when the frame division decoding hasn't been performed or was performed with different decoder names.

Here's how to diagnose and fix this:

## Debug Steps

**1. First, check what keys actually exist in your `results2D` object:**

```python
# Check the structure
print("Keys in frame_divided_epochs_results:", results2D.frame_divided_epochs_results.keys())
print("Keys in decoders:", results2D.decoders.keys())
```

**2. Check if frame division was performed:**

```python
# Verify the frame_divided_epochs_df exists and has data
print("Frame divided epochs shape:", results2D.frame_divided_epochs_df.shape)
print("Frame divided epochs:\n", results2D.frame_divided_epochs_df.head())
```

## Common Solutions

### Solution 1: The decoder key might not be 'global'

If you see different keys (like `'long'`, `'short'`, `'LR'`, `'RL'`), you need to either:

**Option A:** Use the correct key name when creating the helper:

```python
# Check what keys are available
available_keys = list(results2D.frame_divided_epochs_results.keys())
print(f"Available decoder keys: {available_keys}")

# Use the first available key or specify the one you want
decoder_key = available_keys[0]  # or specify explicitly like 'long', 'short', etc.
```

**Option B:** Modify the helper class to accept a custom decoder name:

```python
# You'll need to add this parameter to SingleArtistMultiEpochBatchHelpers
# For now, you can access the result directly:
a_result2D = results2D.frame_divided_epochs_results[decoder_key]
a_decoder = results2D.decoders[decoder_key]
```

### Solution 2: Frame division decoding wasn't performed

If `frame_divided_epochs_results` is empty or missing, you need to run the decoding:

```python
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs

# Build frame divisions
frame_divide_bin_size = 1.0  # 1 second frames
(global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df = \
    Compute_NonPBE_Epochs.build_frame_divided_epochs(
        curr_active_pipeline, 
        frame_divide_bin_size=frame_divide_bin_size
    )

# Decode the frame-divided epochs
# Assuming you have a 2D decoder named 'global' or similar
epochs_decoding_time_bin_size = 0.025  # 25ms bins

frame_divided_epochs_specific_decoded_results_dict = {
    a_name: a_decoder.decode_specific_epochs(
        spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
        filter_epochs=deepcopy(global_frame_divided_epochs_obj),
        decoding_time_bin_size=epochs_decoding_time_bin_size,
        debug_print=False
    )
    for a_name, a_decoder in new_decoder2D_dict.items()  # your 2D decoders
}
```

### Solution 3: Quick workaround - manually set the decoder name

You can temporarily patch the class to use whatever key is available:

```python
# Find what key is actually available
available_key = list(results2D.frame_divided_epochs_results.keys())[0]

# Create a wrapper that uses the correct key
class MyBatchHelper(SingleArtistMultiEpochBatchHelpers):
    @property
    def a_result2D(self):
        return self.results2D.frame_divided_epochs_results[available_key]
    
    @property  
    def a_new_global2D_decoder(self):
        return self.results2D.decoders[available_key]

# Use your custom class
batch_plot_helper = MyBatchHelper(
    results2D=results2D,
    active_ax=track_ax,
    frame_divide_bin_size=frame_divide_bin_size,
    desired_epoch_start_idx=0,
    desired_epoch_end_idx=None
)
```

**What to do next:**

Run the first debug code snippet I provided to see what's actually in your `results2D` object, then share the output. That will tell us exactly which solution you need.

## Sources

- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L136-170](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L136-170)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis active_epoch_name](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20active_epoch_name)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py@26f673d?L534-555](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py?L534-555)


---

# ---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[31], line 2
      1 global_session = curr_active_pipeline.sess
----> 2 plots_data = batch_plot_helper.add_all_track_plots(global_session=global_session)
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:723, in SingleArtistMultiEpochBatchHelpers.add_all_track_plots(self, global_session, override_ax, posterior_masking_value, debug_print, defer_draw, **kwargs)
    719     raise e
    722 # track_shape_patch_collection_artists = batch_plot_helper.add_track_shapes(global_session=global_session, override_ax=track_shapes_dock_track_ax) ## does not seem to successfully synchronize to window
--> 723 plots_data.curr_artist_dict, plots_data.image_extent, plots_data = self.add_position_posteriors(posterior_masking_value=posterior_masking_value, override_ax=override_ax, debug_print=debug_print, defer_draw=True, extant_plot_data=plots_data)
    725 measured_pos_line_artist, frame_division_epoch_separator_vlines = self.add_track_positions(override_ax=override_ax, debug_print=debug_print, defer_draw=True)
    726 # measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions(override_ax=measured_pos_dock_track_ax)
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:539, in SingleArtistMultiEpochBatchHelpers.add_position_posteriors(self, override_ax, posterior_masking_value, debug_print, defer_draw, **kwargs)
    532 curr_artist_dict = {}
    533 ## Perform the plot:
    534 # curr_artist_dict['prev_heatmaps'], (a_meas_pos_line, a_line), (_meas_pos_out_markers, _out_markers), plots_data = DecodedTrajectoryMatplotlibPlotter._perform_add_decoded_posterior_and_trajectory(active_ax, xbin_centers=a_xbin_centers, a_p_x_given_n=a_p_x_given_n,
    535 #                                                                     a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, a_measured_pos_df=a_measured_pos_df, ybin_centers=a_ybin_centers,
    536 #                                                                     include_most_likely_pos_line=None, time_bin_index=None, rotate_to_vertical=True, should_perform_reshape=False, should_post_hoc_fit_to_image_extent=False, debug_print=True) # , allow_time_slider=True
    537 
    538 # Delegate the posterior plotting functionality.
--> 539 curr_artist_dict['prev_heatmaps'], image_extent, plots_data = _active_plot_fn(active_ax,
    540                                                 xbin_centers=a_xbin_centers, ybin_centers=a_ybin_centers, a_time_bin_centers=None, a_p_x_given_n=a_p_x_given_n,
    541                                                 posterior_masking_value=posterior_masking_value, rotate_to_vertical=False, debug_print=True, should_perform_reshape=False, custom_image_extent=custom_image_extent, extant_plot_data=kwargs.get('extant_plot_data', None))
    544 if not defer_draw:
    545     if override_ax is None:
File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\decoder_plotting_mixins.py:1905, in DecodedTrajectoryMatplotlibPlotter._helper_add_hdr_contours(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, ybin_centers, rotate_to_vertical, debug_print, posterior_masking_value, full_posterior_opacity, custom_image_extent, time_cmap, should_perform_reshape, extant_plot_data, contour_level_fractions, filled, smoothing_sigma)
   1899         cset = an_ax.contourf(XX, YY, frame_data, 
   1900                             levels=fill_levels, 
   1901                             colors=[rgba_color], 
   1902                             alpha=full_posterior_opacity)
   1903     else:
   1904         # Outlines (Lines)
-> 1905         cset = an_ax.contour(XX, YY, frame_data, 
   1906                             levels=current_levels, 
   1907                             colors=[rgba_color], 
   1908                             linewidths=1.5, 
   1909                             alpha=full_posterior_opacity)
   1911     artists_list.append(cset)
   1913 except ValueError as e:
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\__init__.py:1493, in _preprocess_data.<locals>.inner(ax, data, *args, **kwargs)
   1490 @functools.wraps(func)
   1491 def inner(ax, *args, data=None, **kwargs):
   1492     if data is None:
-> 1493         return func(ax, *map(sanitize_sequence, args), **kwargs)
   1495     bound = new_sig.bind(ax, *args, **kwargs)
   1496     auto_label = (bound.arguments.get(label_namer)
   1497                   or bound.kwargs.get(label_namer))
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\axes\_axes.py:6520, in Axes.contour(self, *args, **kwargs)
   6511 """
   6512 Plot contour lines.
   6513 
   (...)
   6517 %(contour_doc)s
   6518 """
   6519 kwargs['filled'] = False
-> 6520 contours = mcontour.QuadContourSet(self, *args, **kwargs)
   6521 self._request_autoscale_view()
   6522 return contours
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\contour.py:858, in ContourSet.__init__(self, ax, levels, filled, linewidths, linestyles, hatches, alpha, origin, extent, cmap, colors, norm, vmin, vmax, extend, antialiased, nchunk, locator, transform, negative_linestyles, clip_path, *args, **kwargs)
    854 if self.negative_linestyles is None:
    855     self.negative_linestyles = \
    856         mpl.rcParams['contour.negative_linestyle']
--> 858 kwargs = self._process_args(*args, **kwargs)
    859 self._process_levels()
    861 self._extend_min = self.extend in ['min', 'both']
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\contour.py:1523, in QuadContourSet._process_args(self, corner_mask, algorithm, *args, **kwargs)
   1520         corner_mask = mpl.rcParams['contour.corner_mask']
   1521 self._corner_mask = corner_mask
-> 1523 x, y, z = self._contour_args(args, kwargs)
   1525 contour_generator = contourpy.contour_generator(
   1526     x, y, z, name=self._algorithm, corner_mask=self._corner_mask,
   1527     line_type=contourpy.LineType.SeparateCode,
   1528     fill_type=contourpy.FillType.OuterCode,
   1529     chunk_size=self.nchunk)
   1531 t = self.get_transform()
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\contour.py:1563, in QuadContourSet._contour_args(self, args, kwargs)
   1561 elif 2 < nargs <= 4:
   1562     x, y, z_orig, *args = args
-> 1563     x, y, z = self._check_xyz(x, y, z_orig, kwargs)
   1565 else:
   1566     raise _api.nargs_error(fn, takes="from 1 to 4", given=nargs)
File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\matplotlib\contour.py:1610, in QuadContourSet._check_xyz(self, x, y, z, kwargs)
   1608 elif x.ndim == 2:
   1609     if x.shape != z.shape:
-> 1610         raise TypeError(
   1611             f"Shapes of x {x.shape} and z {z.shape} do not match")
   1612     if y.shape != z.shape:
   1613         raise TypeError(
   1614             f"Shapes of y {y.shape} and z {z.shape} do not match")
TypeError: Shapes of x (1611194, 41) and z (41, 1611194) do not match

Based on the error and my analysis of the code, the issue is in how `add_all_track_plots` is calling a different plotting function than expected. The error shows it's calling `_helper_add_hdr_contours` (High Dynamic Range contours) but the line numbers suggest the actual calling path has changed. Looking at line 723 of your traceback and line 539 in decoder_plotting_mixins.py, there's a mismatch.

Here's the **solution**:

## Quick Fix

The issue is that `add_position_posteriors()` is calling the **wrong plotting function**. It should be calling `_helper_add_heatmap` but is instead calling something that uses contour plots with incompatible shapes.

**Temporary workaround - disable contour plotting:**

```python
# Before calling add_all_track_plots, patch the method to use heatmap instead:
batch_plot_helper.add_position_posteriors = lambda **kwargs: batch_plot_helper._add_position_posteriors_heatmap_only(**kwargs)

# Or more directly, just call the individual plot methods instead of add_all_track_plots:
batch_plot_helper.shared_build_flat_stacked_data(force_recompute=True, debug_print=True)
plots_data = batch_plot_helper.add_track_shapes(global_session=global_session, override_ax=None, defer_draw=True)

# Skip the position posteriors that are failing
# batch_plot_helper.add_position_posteriors(...) # SKIP THIS

# Add positions only
measured_pos_line_artist, frame_division_epoch_separator_vlines = batch_plot_helper.add_track_positions(override_ax=None, debug_print=False, defer_draw=False)
```

## Proper Fix

The real issue is in lines 530-532 vs what's actually being called at line 539 (from your error). Check what `_active_plot_fn` is set to:

```python
# Debug what plotting function is being used
print(f"batch_plot_helper line 539 function: {batch_plot_helper._active_plot_fn if hasattr(batch_plot_helper, '_active_plot_fn') else 'not set'}")
```

The **transpose issue** is that XX and YY meshgrid arrays don't match the data shape. This happens because:
- `XX, YY = np.meshgrid(x_values, y_values)` creates shapes based on the input
- But the `frame_data` passed to `contour()` has been transposed somewhere

**To fix the transpose**, you need to check the `rotate_to_vertical` parameter and ensure consistent axis ordering. Add this before line 723 in your code:

```python
# Force using heatmap mode instead of contour mode  
batch_plot_helper.add_position_posteriors = lambda override_ax=None, **kwargs: \
    SingleArtistMultiEpochBatchHelpers._add_posteriors_as_heatmap_only(
        batch_plot_helper, override_ax=override_ax, **kwargs
    )
```

Or **simpler - just skip the posterior visualization for now**:

```python
# Call methods individually, skipping the problematic posteriors
batch_plot_helper.shared_build_flat_stacked_data(force_recompute=True, debug_print=True)
track_shapes = batch_plot_helper.add_track_shapes(global_session=global_session)
measured_pos, separators = batch_plot_helper.add_track_positions()
# Don't call add_position_posteriors - it's the one failing
```

Would you like me to help you implement one of these fixes, or would you prefer to share more of your setup code so I can give a more targeted solution?

## Sources

- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1850-1920](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1850-1920)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L520-545](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L520-545)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L695-750](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L695-750)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1750-1780](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1750-1780)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1820-1920](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1820-1920)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1600-1700](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1600-1700)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L2000-2100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L2000-2100)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1450-1580](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1450-1580)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis _helper_add_hdr_contours meshgrid XX YY](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20_helper_add_hdr_contours%20meshgrid%20XX%20YY)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis contour filled smoothing_sigma contour_level_fractions](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20contour%20filled%20smoothing_sigma%20contour_level_fractions)

## Suggested Follow-ups

- What determines whether `add_position_posteriors` calls `_helper_add_hdr_contours` vs. `_helper_add_heatmap`, and how can I control this?
- Can you explain how the `rotate_to_vertical` parameter affects the shapes of the x, y, and z arrays passed to the contour function?


---

# Sorry, I don't want to re-implement anything, I'm just trying to remember where I left off. I remember that there was some ambiguity related to getting the global (all time) epoch, can you remind me of my recent changes related to this that could be causing this problem?

Perfect! Now I found it. Here's what happened:

## The Root Cause

On **2025-09-05** (commit `02d7a57`), you replaced calls to:
```python
_, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
```

with:
```python
global_epoch_name = curr_active_pipeline.find_Global_epoch_name()
```

This was done to support **Bapun sessions** (which don't have "long" and "short" epochs, only a global epoch).

## The Problem

The `SingleArtistMultiEpochBatchHelpers` class is **hardcoded** to look for the key `'global'` (line 161):

```python
@property
def a_result2D(self) -> DecodedFilterEpochsResult:
    return self.results2D.frame_divided_epochs_results['global']  # <-- HARDCODED 'global'
```

But `find_Global_epoch_name()` might return a **different name** depending on your session:
- For **Kdiba sessions**: returns `'maze'` or `'maze1'` or `'maze2'`  
- For **Bapun sessions**: might return something else
- Only **some sessions** actually have an epoch named `'global'`

## The Solution

You need to **pass the actual epoch name** to the helper. Here's how to fix it:

```python
# Instead of hardcoding 'global', get the actual global epoch name
global_epoch_name = curr_active_pipeline.find_Global_epoch_name()
print(f"Global epoch name: {global_epoch_name}")  # Debug: see what it actually is

# Check what keys are in your results
print(f"Available decoder keys: {list(results2D.frame_divided_epochs_results.keys())}")
print(f"Available position decoder keys: {list(results2D.decoders.keys())}")

# Create a custom batch helper that uses the correct epoch name
class MyBatchHelper(SingleArtistMultiEpochBatchHelpers):
    def __init__(self, *args, epoch_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_name = epoch_name or 'global'
    
    @property
    def a_result2D(self):
        return self.results2D.frame_divided_epochs_results[self.epoch_name]
    
    @property  
    def a_new_global2D_decoder(self):
        return self.results2D.decoders[self.epoch_name]

# Use your custom class with the correct epoch name
batch_plot_helper = MyBatchHelper(
    results2D=results2D,
    active_ax=track_ax,
    frame_divide_bin_size=frame_divide_bin_size,
    desired_epoch_start_idx=0,
    desired_epoch_end_idx=None,
    epoch_name=global_epoch_name  # Pass the actual name
)
```

**Or more directly** - just check what key your data actually has and use it:

```python
# Quick fix - just use whatever key is in there
available_key = list(results2D.frame_divided_epochs_results.keys())[0]
print(f"Using decoder key: {available_key}")

# Monkey-patch the properties
SingleArtistMultiEpochBatchHelpers.a_result2D = property(
    lambda self: self.results2D.frame_divided_epochs_results[available_key]
)
SingleArtistMultiEpochBatchHelpers.a_new_global2D_decoder = property(
    lambda self: self.results2D.decoders[available_key]
)

# Now create the helper normally
batch_plot_helper = SingleArtistMultiEpochBatchHelpers(...)
```

This is **exactly** the kind of ambiguity you were remembering - the change from hardcoded `'global'` epoch names to dynamic epoch name lookup that broke code expecting the old convention!

## Sources

- [type:commit (message:"global epoch" OR message:LongShortGlobal OR message:epoch_name OR message:find_LongShortGlobal) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2024-01-01 count:20](/search?q=type:commit%20%28message:%22global%20epoch%22%20OR%20message:LongShortGlobal%20OR%20message:epoch_name%20OR%20message:find_LongShortGlobal%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2024-01-01%20count:20)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis find_LongShortGlobal_epoch_names](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20find_LongShortGlobal_epoch_names)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis active_epoch_name global](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20active_epoch_name%20global)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L136-170](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L136-170)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis def find_Global_epoch_name](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20def%20find_Global_epoch_name)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis (diff 02d7a57~1...02d7a57)](/github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis/-/compare/02d7a57~1..02d7a57)

## Suggested Follow-ups

- How can I make the `SingleArtistMultiEpochBatchHelpers` class automatically detect the correct epoch name without needing a custom class?
- Besides the epoch name, are there other hardcoded assumptions in `SingleArtistMultiEpochBatchHelpers` that might cause problems with different session types or pipeline configurations?

