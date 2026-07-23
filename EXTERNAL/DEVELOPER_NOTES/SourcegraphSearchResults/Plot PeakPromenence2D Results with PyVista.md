---
title: Plot PeakPromenence2D Results with PyVista in pyPhoPlaceCellAnalysis
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/4f00c71a-6a8b-45f7-9ee9-e93161a62136
url: https://phohale.sourcegraph.app/deepsearch/362
created: 2026-04-01T20:59:31Z
updated: 2026-04-01T21:00:12Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis Plotting PeakPromenence2D results on a pyvista-backed plotter with promenence curves, done within the last year

The key work on plotting PeakProminence2D results on a pyvista-backed plotter with prominence curves was done in two commits on **June 27, 2025**:

## Key Commits

1. **[50b2166](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/commit/50b21669ce6a8cfd9d95f86eb2404595badc064b)** - June 27, 2025
   - Added `LongShort3DPlacefieldsHelpers.render_long_short_all_neuron_peak_prominence_2d_results_on_pyvista_plotter`
   - Extended prominence plotting to support both long and short placefields simultaneously

2. **[7fc97d7](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/commit/7fc97d7c86398de66dda382afd156526eb976280)** - June 27, 2025  
   - Refactored by extracting `_subfn_plot_for_pdata` into `_plot_pyvista_single_neuron_prominence_result`
   - Cleaned up and modularized the prominence plotting code

## Implementation Details

The prominence plotting functionality is in [peak_prominences.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L255-L400):

**Core functions:**
- [`_render_peak_prominence_2d_results_on_pyvista_plotter`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L256-L363) - Renders peak prominence for a single neuron
- [`_plot_pyvista_single_neuron_prominence_result`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L115-L249) - Plots contours, bounding boxes, text labels, and peak points
- [`render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L378-L401) - Renders prominence for all neurons

**What gets plotted:**
- **Prominence contours** at peak levels ([line 145](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L145))
- **Bounding boxes** around prominence regions ([lines 165-171](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L165-L171))
- **Text labels** showing dimensions ([lines 183-216](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L183-L216))
- **Peak point markers** at peak locations ([lines 222-235](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L222-L235))

## Sources

- [type:commit (message:PeakProminence2D OR message:pyvista OR message:prominence) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2025-04-01 count:100 timeout:60s](/search?q=type:commit%20%28message:PeakProminence2D%20OR%20message:pyvista%20OR%20message:prominence%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2025-04-01%20count:100%20timeout:60s)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ PeakProminence2D pyvista prominence plotter](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20PeakProminence2D%20pyvista%20prominence%20plotter)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py@26f673d?L254-400](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L254-400)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis (diff 50b21669ce6a8cfd9d95f86eb2404595badc064b~1...50b21669ce6a8cfd9d95f86eb2404595badc064b)](/github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis/-/compare/50b21669ce6a8cfd9d95f86eb2404595badc064b~1..50b21669ce6a8cfd9d95f86eb2404595badc064b)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis (diff 7fc97d7c86398de66dda382afd156526eb976280~1...7fc97d7c86398de66dda382afd156526eb976280)](/github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis/-/compare/7fc97d7c86398de66dda382afd156526eb976280~1..7fc97d7c86398de66dda382afd156526eb976280)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py@26f673d?L115-255](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho3D/PyVista/peak_prominences.py?L115-255)

