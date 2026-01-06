---
name: Add peak counts layer to Napari plotter
overview: Add a method to DecodedTrajectoryNapariPlotter that takes a PosteriorPeaksPeakProminence2dResult object and adds the peak_counts.raw as a new separate Napari layer alongside the decoded posteriors.
todos:
  - id: add_method_signature
    content: Add method signature `add_peak_counts_layer` to DecodedTrajectoryNapariPlotter class with PosteriorPeaksPeakProminence2dResult parameter
    status: completed
  - id: extract_peak_counts
    content: Extract peak_counts.raw from the PosteriorPeaksPeakProminence2dResult object
    status: completed
    dependencies:
      - add_method_signature
  - id: broadcast_to_4d
    content: Broadcast the 2D peak_counts.raw array to 4D shape matching posterior_volume (num_epochs, max_num_time_bins, n_xbins, n_ybins)
    status: completed
    dependencies:
      - extract_peak_counts
  - id: add_napari_layer
    content: Add the broadcasted peak counts as a new Napari image layer with appropriate name and colormap
    status: completed
    dependencies:
      - broadcast_to_4d
  - id: handle_edge_cases
    content: Add error handling for missing viewer or posterior_volume, and coordinate alignment checks
    status: completed
    dependencies:
      - add_napari_layer
---

# Add Peak Counts Layer to DecodedTrajectoryNapariPlotter

## Overview

Add a method to `DecodedTrajectoryNapariPlotter` that takes a `PosteriorPeaksPeakProminence2dResult` object and adds the `peak_counts.raw` counter map as a new separate Napari image layer.

## Implementation Details

### Method Signature

Add a new method `add_peak_counts_layer` to the `DecodedTrajectoryNapariPlotter` class in [`pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py`](pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py) (around line 3798, after `build_ui`).The method should:

- Accept a `PosteriorPeaksPeakProminence2dResult` parameter
- Extract `peak_counts.raw` (2D array of shape `(n_xbins, n_ybins)`)
- Broadcast it to match the `posterior_volume` shape `(num_epochs, max_num_time_bins, n_xbins, n_ybins)` by repeating across epochs and time bins
- Add it as a new Napari image layer with appropriate name (e.g., 'peak_counts')
- Return the created layer
- Handle the case where `viewer` or `posterior_volume` hasn't been built yet

### Key Implementation Points

1. **Shape Broadcasting**: The `peak_counts.raw` is 2D `(n_xbins, n_ybins)`, but needs to be 4D `(num_epochs, max_num_time_bins, n_xbins, n_ybins)` to match the posterior volume structure. Use `np.broadcast_to` or `np.tile` to expand dimensions.
2. **Coordinate Alignment**: Verify that the `xbin_centers` and `ybin_centers` from the `PosteriorPeaksPeakProminence2dResult` match those in the plotter, or handle mismatches appropriately.
3. **Layer Configuration**: Use a different colormap (e.g., 'plasma' or 'inferno') and potentially different blending mode to distinguish from the posterior layer.