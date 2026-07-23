---
title: Exporting SpikeRasters Decoded Epochs Figures in Spike3D
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/b5cd214f-3370-4939-a235-bbe72fcc7857
url: https://phohale.sourcegraph.app/deepsearch/358
created: 2026-03-28T02:07:07Z
updated: 2026-07-22T17:37:41Z
---

# github.com/CommanderPho/Spike3D find how I export spike rasters and decoded PBE figures for the 1D/KDiba-style project

Based on my exploration of your Spike3D codebase, I can now provide you with a comprehensive answer about how you export spike rasters and decoded PBE figures for the 1D/KDiba-style project.

## Exporting Spike Rasters and Decoded PBE Figures

You use two main approaches for exporting these figures:

### 1. **Batch Export via Completion Functions** (Primary Method)

The main export workflow is orchestrated through the [`figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/ProcessBatchOutputs.ipy?L329), which is configured in your batch processing scripts:

**Configuration in [ProcessBatchOutputs.ipy](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/ProcessBatchOutputs.ipy?L328-330):**
```python
'generalized_decode_epochs_dict_and_export_results_completion_function': dict(
    epochs_decoding_time_bin_size=ripple_decoding_time_bin_size, 
    force_recompute=drop_previous_result_and_compute_fresh, 
    debug_print=True
),
'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function': dict(),
```

This function is imported from:
```python
from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function
```

**Usage in notebooks** (e.g., [ReviewOfWork_2025-06-26.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-26.ipynb?L11396-11407)):
```python
_across_session_results_extended_dict = _across_session_results_extended_dict | figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(
    a_dummy, None,
    curr_session_context=complete_session_context,
    curr_session_basedir=curr_active_pipeline.sess.basepath.resolve(), 
    curr_active_pipeline=curr_active_pipeline,
    across_session_results_extended_dict=_across_session_results_extended_dict,
    included_figures_names=['_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay'],
)
```

### 2. **Direct PBE/Ripple Export via `PosteriorExporting`**

For exporting individual decoded PBE (ripple) epochs with spike rasters, you use the `PosteriorExporting._perform_export_current_epoch_marginal_and_raster_images` method:

**Code pattern** (from [ReviewOfWork_2025-06-03.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-03.ipynb?L17180-17192)):
```python
from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting

# Get the directional merged decoders result
directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']

# Set export path
root_export_path: Path = Path(r'K:/scratch/collected_outputs/figures/array_as_image').resolve()
root_export_path.mkdir(exist_ok=True)

# Export ripple/PBE events with rasters
attached_ripple_rasters_widget: RankOrderRastersDebugger = paginated_multi_decoder_decoded_epochs_window.attached_ripple_rasters_widget
_out_ripple_rasters = paginated_multi_decoder_decoded_epochs_window

epoch_specific_folder, (out_image_save_tuple_dict, _out_rasters_save_paths, merged_img_save_path) = PosteriorExporting._perform_export_current_epoch_marginal_and_raster_images(
    _out_ripple_rasters=_out_ripple_rasters, 
    directional_merged_decoders_result=directional_merged_decoders_result, 
    filtered_decoder_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, 
    epoch_id_identifier_str='ripple',
    active_session_context=curr_context, 
    root_export_path=root_export_path,
)
```

### 3. **Image Export Utilities**

The actual image saving is handled by helper functions:
- `save_array_as_image` - for individual images
- `get_array_as_image` - to get PIL Image objects
- `get_array_as_image_stack` - for creating stacked visualizations

These are imported from:
```python
from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image, get_array_as_image, get_array_as_image_stack
```

### Key Export Features

1. **Supports multiple epoch types:** ripple/PBE events and laps
2. **Exports include:**
   - Decoded posterior marginals (position probability over time)
   - Spike rasters aligned to epochs
   - Merged composite images combining posteriors and rasters
3. **Custom export formats** defined via `HeatmapExportConfig` for different visualizations (greyscale, colormapped, raw RGBA)
4. **Session-specific output folders** organized by session context

The typical export path structure follows: `{root_export_path}/{session_context}/{epoch_type}/...`

## Sources

- [repo:^github.com/CommanderPho/Spike3D$ export spike raster figure](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20export%20spike%20raster%20figure)
- [repo:^github.com/CommanderPho/Spike3D$ PBE decode export figure](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20PBE%20decode%20export%20figure)
- [repo:^github.com/CommanderPho/Spike3D$ KDiba 1D export](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20KDiba%201D%20export)
- [repo:^github.com/CommanderPho/Spike3D$ decoded_PBE OR spike_raster export](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20decoded_PBE%20OR%20spike_raster%20export)
- [repo:^github.com/CommanderPho/Spike3D$ figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function)
- [repo:^github.com/CommanderPho/Spike3D$ save figure raster decoded posterior epoch](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20save%20figure%20raster%20decoded%20posterior%20epoch)
- [repo:^github.com/CommanderPho/Spike3D$ plot_raster_plot OR plot_decoded_epoch_slices](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20plot_raster_plot%20OR%20plot_decoded_epoch_slices)
- [github.com/CommanderPho/Spike3D/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-26.ipynb@1c479db?L11250-11450](/github.com/CommanderPho/Spike3D@1c479db460c9ceea4ef7856da8a354dcd346c4e5/-/blob/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-26.ipynb?L11250-11450)
- [repo:^github.com/CommanderPho/Spike3D$ _perform_export_current_epoch_marginal_and_raster_images OR PosteriorExporting](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20_perform_export_current_epoch_marginal_and_raster_images%20OR%20PosteriorExporting)
- [github.com/CommanderPho/Spike3D/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-03.ipynb@1c479db?L17180-17230](/github.com/CommanderPho/Spike3D@1c479db460c9ceea4ef7856da8a354dcd346c4e5/-/blob/EXTERNAL/_DEP_OldNotebooks/ReviewOfWork_2025-06-03.ipynb?L17180-17230)
- [repo:^github.com/CommanderPho/Spike3D$ export_array_as_image OR save_array_as_image](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20export_array_as_image%20OR%20save_array_as_image)

