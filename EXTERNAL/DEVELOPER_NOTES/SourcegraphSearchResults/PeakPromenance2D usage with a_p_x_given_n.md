---
title: PeakPromenance2D usage with a_p_x_given_n
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/ad78b9b9-8272-455c-8e10-e2c050b2757c
url: https://phohale.sourcegraph.app/deepsearch/301
created: 2025-12-12T14:02:11Z
updated: 2026-01-15T10:15:29Z
---

# How to use PeakPromenance2D for a given `a_p_x_given_n`

Based on my search, I can see that **`PeakProminence2D` is used for analyzing placefield ratemaps, not decoder posteriors** (`a_p_x_given_n`). Let me clarify the usage:

The `a_p_x_given_n` variable represents **decoder posteriors** (probability of position given spikes), while `PeakProminence2D` is used for **analyzing 2D placefield tuning curves**. These are two different things in the codebase.

However, let me show you how to use `PeakProminence2D` / `compute_prominence_contours`:

## Using `compute_prominence_contours` (PeakProminence2D)

Based on the code in [peak_prominence2d.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py?L501-L516) and the usage in [PlacefieldDensityAnalysisComputationFunctions.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/PlacefieldDensityAnalysisComputationFunctions.py?L711-L713):

```python
from pyphoplacecellanalysis.External.peak_prominence2d import compute_prominence_contours

# For a 2D placefield ratemap (NOT a_p_x_given_n):
# Get your placefield data
slab = active_pf_2D.ratemap.tuning_curves[neuron_idx].T  # Note: transpose the tuning curve

# Call compute_prominence_contours
xx, yy, slab, peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(
    xbin_centers=active_pf_2D.xbin_centers,  # x-axis bin centers
    ybin_centers=active_pf_2D.ybin_centers,  # y-axis bin centers 
    slab=slab,                                # transposed tuning curve
    step=0.1,                                 # contour interval (smaller = finer)
    min_area=None,                            # optional minimum area filter
    min_depth=0.2,                            # minimum prominence threshold
    include_edge=True,                        # include edge contours
    verbose=False
)

# peaks_dict contains the found peaks with their properties
# Each peak has: 'height', 'prominence', 'center', 'area', 'contours', etc.
```

If you're actually trying to analyze decoder posteriors (`a_p_x_given_n`), that's a different use case and doesn't involve `PeakProminence2D`. Could you clarify which analysis you're trying to perform?

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ a_p_x_given_n](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20a_p_x_given_n)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ PeakProminence2D peak prominence](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20PeakProminence2D%20peak%20prominence)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py@26f673d?L1-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py?L1-100)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ compute_prominence_contours a_p_x_given_n usage example](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20compute_prominence_contours%20a_p_x_given_n%20usage%20example)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ compute_prominence_contours](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20compute_prominence_contours)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py@26f673d?L500-523](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py?L500-523)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/PlacefieldDensityAnalysisComputationFunctions.py@26f673d?L700-750](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/PlacefieldDensityAnalysisComputationFunctions.py?L700-750)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "a_p_x_given_n" "PeakProminence" OR "peak_prominence" OR "prominence"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22a_p_x_given_n%22%20%22PeakProminence%22%20OR%20%22peak_prominence%22%20OR%20%22prominence%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py@26f673d?L150-430](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py?L150-430)

## Suggested Follow-ups

- How can I analyze decoder posteriors (`a_p_x_given_n`) to find the most probable location or track changes in certainty over time if `PeakProminence2D` is not suitable?
- If `PeakProminence2D` is intended for analyzing placefield ratemaps, how can I integrate it with decoder posteriors (`a_p_x_given_n`) to potentially refine location estimates or assess the reliability of placefield activation patterns?


---

# want to very efficiently compute the prominence of at least the top 3 peaks, and get the region around the top peak as a mask. I have `a_p_x_given_n[:, :, a_timestamp_idx]`

Perfect! Now let me create an efficient solution for you:

```python
from pyphoplacecellanalysis.External.peak_prominence2d import compute_prominence_contours
import numpy as np
from matplotlib.path import Path

# Extract your 2D posterior slice at a specific timestamp
posterior_slice = a_p_x_given_n[:, :, a_timestamp_idx]  # Shape: (n_x_bins, n_y_bins) or similar

# Get bin centers (if you have them, otherwise use indices)
# xbin_centers = ... # your x position bin centers
# ybin_centers = ... # your y position bin centers
# If you don't have them:
xbin_centers = np.arange(posterior_slice.shape[1])
ybin_centers = np.arange(posterior_slice.shape[0])

# Compute prominence - note the transpose!
xx, yy, slab, peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(
    xbin_centers=xbin_centers,
    ybin_centers=ybin_centers,
    slab=posterior_slice.T,  # IMPORTANT: transpose the slice
    step=0.05,                # smaller = finer but slower (0.01-0.1 typical)
    min_depth=0.1,            # minimum prominence threshold
    min_area=None,
    include_edge=True,
    verbose=False
)

# Sort peaks by prominence to get top 3
peaks_by_prominence = sorted(peaks_dict.items(), 
                             key=lambda x: x[1]['prominence'], 
                             reverse=True)
top_3_peaks = peaks_by_prominence[:3]

# Get the top peak
top_peak_id, top_peak_info = top_3_peaks[0]

# Create a mask for the region around the top peak
# The outermost contour defines the peak's region
top_peak_contour = top_peak_info['contour']  # or top_peak_info['contours'][-1]

# Create a boolean mask from the contour
mask = np.zeros(posterior_slice.T.shape, dtype=bool)  # Note: transposed shape
yy_grid, xx_grid = np.meshgrid(ybin_centers, xbin_centers, indexing='ij')
points = np.vstack([xx_grid.ravel(), yy_grid.ravel()]).T

# Check which points are inside the contour
inside = top_peak_contour.contains_points(points)
mask = inside.reshape(posterior_slice.T.shape)

# If you want the mask in the original orientation:
mask = mask.T

# Summary of results
print(f"Found {len(peaks_dict)} peaks")
print(f"\nTop 3 peaks by prominence:")
for i, (peak_id, peak_info) in enumerate(top_3_peaks):
    print(f"  {i+1}. Peak {peak_id}: prominence={peak_info['prominence']:.3f}, "
          f"height={peak_info['height']:.3f}, center={peak_info['center']}")

# Access peak properties:
# top_peak_info['center']      - (x, y) center coordinates
# top_peak_info['prominence']  - prominence value
# top_peak_info['height']      - peak height
# top_peak_info['area']        - contour area
# top_peak_info['contours']    - list of contours from peak to col
```

**Efficiency tips:**

1. **`step` parameter**: Larger values (0.1-0.2) are faster but less accurate. For quick analysis, use 0.1. For precision, use 0.01-0.05.

2. **Filter early**: Use `min_depth` to filter out small peaks immediately rather than sorting all of them.

3. **For just top peak** (fastest):
   ```python
   # If you only need the top peak, use aggressive filtering
   xx, yy, slab, peaks_dict, id_map, prominence_map, parent_map = compute_prominence_contours(
       xbin_centers=xbin_centers,
       ybin_centers=ybin_centers, 
       slab=posterior_slice.T,
       step=0.1,           # coarser for speed
       min_depth=0.3,      # higher threshold to get fewer peaks
       include_edge=True,
       verbose=False
   )
   
   # Get top peak directly
   if peaks_dict:
       top_peak_id = max(peaks_dict.keys(), key=lambda k: peaks_dict[k]['prominence'])
       top_peak = peaks_dict[top_peak_id]
   ```

The `mask` variable will be a boolean array the same shape as your posterior slice, with `True` values inside the top peak's region.

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ id_map prominence_map mask contour](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20id_map%20prominence_map%20mask%20contour)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py@26f673d?L430-500](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py?L430-500)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ contains_point mask polygon region peak](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20contains_point%20mask%20polygon%20region%20peak)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "peaks_dict" sort prominence height top](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22peaks_dict%22%20sort%20prominence%20height%20top)

## Suggested Follow-ups

- How can I vectorize the mask creation step to improve performance further?
- Can you provide an example of how to use the resulting mask to filter the original `a_p_x_given_n` array?


---

# Find all other instances in the codebase including github.com/CommanderPho/Spike3D where peak prominence as a concept is used. 

Perfect! Now I have a comprehensive view. Let me create a summary of all peak prominence-related instances:

## Summary: All Peak Prominence Usage in Codebases

Based on my comprehensive search across **github.com/CommanderPho/pyPhoPlaceCellAnalysis** and **github.com/CommanderPho/Spike3D**, here are all instances where peak prominence as a concept is used:

### **Primary Implementation - 2D Topographic Peak Prominence**

#### Core Module
- [**peak_prominence2d.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/External/peak_prominence2d.py) - Complete implementation of 2D peak prominence detection using contour methods
  - `getProminence()` - Main algorithm (lines 159-498)
  - `compute_prominence_contours()` - User-friendly wrapper (lines 501-516)
  - `plot_Prominence()` - Visualization function (line 524)

#### Computation Pipeline
- [**PlacefieldDensityAnalysisComputationFunctions.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/PlacefieldDensityAnalysisComputationFunctions.py)
  - `_perform_pf_find_ratemap_peaks_peak_prominence2d_computation()` (lines 366-793) - Main computation function for placefield peak analysis
  - Analyzes **2D placefield ratemaps** to find peaks, their prominences, contours, and bounding boxes
  - Stores results in: `computation_result.computed_data['RatemapPeaksAnalysis']['PeakProminence2D']`

#### Visualization
- [**EloyAnalysis.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/EloyAnalysis.py)
  - `_display_pf_peak_prominence2d_default_quadrant_plots()` (lines 38-86) - 4-quadrant prominence plots
  - `_display_pf_peak_prominence2d_plots()` (lines 91-173) - Custom width/height analysis plots

- [**peak_prominences.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py)
  - `render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter()` (line 378) - 3D PyVista visualization
  - `_render_peak_prominence_2d_results_on_pyvista_plotter()` (line 256) - Single neuron 3D rendering
  - `_build_pyvista_single_neuron_prominence_result_data()` (lines 23-103) - Data extraction for visualization

### **Alternative Peak Detection Methods**

#### 1. **scipy.signal.find_peaks** 
- [**DirectionalPlacefieldGlobalComputationFunctions.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L240-L261)
  - `get_decoders_tuning_curve_modes()` (line 240) - Detects **unimodal/bimodal/multimodal placefields** using `scipy.signal.find_peaks`
  - Used for **1D tuning curves**, not 2D surfaces
  - Returns peak counts per cell to classify placefield modality

#### 2. **findpeaks Library** (Deprecated)
- [**PlacefieldDensityAnalysisComputationFunctions.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/PlacefieldDensityAnalysisComputationFunctions.py?L322-L362)
  - `_DEP_perform_pf_find_ratemap_peaks_computation()` (line 322) - **DEPRECATED** in favor of peak_prominence2d
  - Used external `findpeaks` library for topology and mask methods
  - Note in code: "Not really used as the `peak_prominence2d` seems to work much better"

#### 3. **Burst Detection Peak Finding**
- [**lib_final_ehv.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/External/burst-detection-master/lib_final_ehv.py?L260-L268)
  - Extended Hill-Valley method for **spike burst detection**
  - Finds peaks/troughs in spike trains, not spatial data
  - Different domain: temporal spike analysis vs. spatial placefield analysis

### **Usage in Analysis Pipelines**

#### Long/Short Track Analysis
- [**LongShortTrackComputations.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py)
  - `add_peak_promenance_pf_peaks()` (line 408) - Adds 2D peak info to neuron stats
  - `add_directional_pf_maximum_peaks()` (line 433) - Adds 1D maximum peak locations
  - `_perform_add_peak_promenance_pf_peaks()` (line 344) - Merges prominence results into dataframes
  - Columns added: `['long_LR_pf2D_peak_x', 'long_LR_pf2D_peak_y', 'long_RL_pf2D_peak_x', 'long_RL_pf2D_peak_y', ...]`

#### Parameter Configuration
- [**SpecificComputationParameterTypes.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/SpecificComputationParameterTypes.py?L542-L546)
  - `ratemap_peaks_prominence2d_Parameters` class (line 543)
  - Default parameters: `step=0.01`, `peak_height_multiplier_probe_levels=(0.5, 0.9)`, `minimum_included_peak_height=0.2`

### **Decoder Posterior Peak Detection** (Separate Concept!)

- [**heuristic_replay_scoring.py**](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L3303-L3315)
  - `get_peaks_mask()` (line 3303) - Finds peak probability position for each time bin in **decoder posteriors**
  - `expand_peaks_mask()` (line 3311) - Expands peak mask using convolution
  - **Different from placefield prominence**: this is for **posterior distributions** (`a_p_x_given_n`), not ratemaps

### **In Spike3D Repository**

- [**peak_prominence2d_testing.ipynb**](https://phohale.sourcegraph.app/github.com/CommanderPho/Spike3D/-/blob/EXTERNAL/TESTING/testing_notebooks/peak_prominence2d_testing.ipynb) - Testing notebook for prominence features
- Multiple references in batch processing scripts and configuration files for the `ratemap_peaks_prominence2d` computation stage

### **Key Distinctions**

| Method | Purpose | Data Type | Dimensionality |
|--------|---------|-----------|----------------|
| **peak_prominence2d** | Placefield peak detection | 2D ratemap (tuning curves) | 2D spatial |
| **scipy.signal.find_peaks** | Placefield modality detection | 1D tuning curves | 1D spatial |
| **get_peaks_mask** | Decoder posterior peaks | Posterior probabilities | 2D (position × time) |
| **findpeaks library** | DEPRECATED | 2D ratemaps | 2D spatial |
| **burst detection** | Spike burst detection | Spike trains | 1D temporal |

The **peak_prominence2d** module is the primary and most sophisticated peak prominence implementation, specifically designed for analyzing 2D spatial placefield ratemaps using topographic prominence concepts.

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ prominence peak](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20prominence%20peak)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ peak detection find peaks topographic](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20peak%20detection%20find%20peaks%20topographic)
- [repo:^github.com/CommanderPho/Spike3D$ prominence peak](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20prominence%20peak)
- [repo:^github.com/CommanderPho/Spike3D$ peak detect topographic](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20peak%20detect%20topographic)
- [repo:^github.com/CommanderPho/Spike3D$ find_peaks scipy signal](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FSpike3D$%20find_peaks%20scipy%20signal)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ scipy.signal.find_peaks](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20scipy.signal.find_peaks)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ from scipy.signal import find_peaks](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20from%20scipy.signal%20import%20find_peaks)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L239-320](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L239-320)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ findpeaks library](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20findpeaks%20library)

## Suggested Follow-ups

- Can you provide a code example of how to use the `get_peaks_mask()` function for decoder posteriors (`a_p_x_given_n`)?
- Could you elaborate on the performance differences between `peak_prominence2d` and the deprecated `findpeaks` library, and why `peak_prominence2d` is preferred?

