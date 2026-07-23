---
title: PyPhoPlaceCellAnalysis: Position-Likeness Score Across Instances
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/1f968751-61a9-4a1f-9135-c8ee69da56a7
url: https://phohale.sourcegraph.app/deepsearch/360
created: 2026-04-01T20:19:17Z
updated: 2026-04-01T20:21:44Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/CommanderPho/Spike3D github.com/diba-lab/NeuroPy Position-Likeness score all instances

Perfect! Now I have a comprehensive view of the "Position-Likeness score" implementation. Let me compile the findings:

## Position-Likeness Score Implementation

The **Position-Likeness score** (also called PLI - Position-Like Information score) is implemented in the [`PositionLikePosteriorScoring`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1612-1614) class across your repositories.

### Key Locations

**Primary Implementation:**
- [pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1612) - Main class definition
- [Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1612) - Identical implementation

**Usage in Analysis:**
- [pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py?L4584-4586) - Import and usage
- [Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py?L42-43) - Import for filtering

**Notebook References:**
- [Spike3D/InteractivePipelineLoadFromPickle_Bapun_Day4OpenField.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/InteractivePipelineLoadFromPickle_Bapun_Day4OpenField.ipynb) - Multiple references to "2026-01-08_Position-likePosteriorsOnlyFilteredResult"
- [Spike3D/InteractivePipelineLoadFromPickle_Bapun_RatK_D4OpenField.ipynb](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3D/-/blob/InteractivePipelineLoadFromPickle_Bapun_RatK_D4OpenField.ipynb) - Similar filtered result references

### Score Thresholds

```python
high_quality_score_cutoff: float = 0.7
position_like_score_cutoff: float = 0.42
```

### Core Calculation Method

The [`calculate_pli_score`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1810-1864) method computes a composite score based on two components:

1. **Sharpness (Entropy)** - Weight: 0.4 (default)
   - Measures how concentrated/peaked the posterior is
   - Uses normalized entropy: `s_sharpness = 1.0 - (entropy / max_entropy)`

2. **Locality (Spatial Spread)** - Weight: 0.6 (default)
   - Measures spatial compactness using RMS distance from peak
   - Uses exponential decay: `s_locality = exp(-(rms_dist / sigma_tolerated))`

**Composite Score:**
```python
score = (w_sharpness * s_sharpness) + (w_locality * s_locality)
```

### Classification Labels

From [`get_pli_quality_label`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1867-1873):
- Score ≥ 0.7: **"Position-Like HQ"** (limegreen)
- Score ≥ 0.42: **"Position-Like"** (green)
- Score < 0.42: **"Ambiguous"** (red)

### Main Methods

1. **[`compute_and_plot_posterior_stack`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1878-1961)** - Batch processes posteriors and returns scoring DataFrame

2. **[`filter_to_position_like_epochs_only`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1966-2100)** - Filters decoded epochs to keep only position-like time bins

### Purpose

The Position-Likeness score is used to:
- Determine which decoded time bins contain non-local but position-like posteriors
- Filter decoded epochs to include only well-localized position predictions
- Distinguish quality decoded positions from ambiguous/diffuse posteriors during predictive decoding analysis

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/Spike3D OR repo:github.com/diba-lab/NeuroPy Position-Likeness score](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%20Position-Likeness%20score)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/Spike3D OR repo:github.com/diba-lab/NeuroPy) Position Likeness score](/search?q=patterntype:nls%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%29%20Position%20Likeness%20score)
- [(repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis OR repo:github.com/CommanderPho/Spike3D OR repo:github.com/diba-lab/NeuroPy) Position-Likeness](/search?q=patterntype:nls%20%28repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:github.com%2FCommanderPho%2FSpike3D%20OR%20repo:github.com%2Fdiba-lab%2FNeuroPy%29%20Position-Likeness)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L1610-1900](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1610-1900)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L1-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1-100)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis PositionLikePosteriorScoring class definition](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20PositionLikePosteriorScoring%20class%20definition)
- [github.com/CommanderPho/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@eadd004?L1600-1700](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1600-1700)
- [github.com/CommanderPho/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@eadd004?L1800-2100](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1800-2100)


---

# what code produces this image with the title "Position-Likeness..."

Based on the code I've already analyzed, the image with "Position-Likeness..." in the title is produced by the [`compute_and_plot_posterior_stack`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1878-1961) method in the `PositionLikePosteriorScoring` class.

## Code that produces the visualization:

```python
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import PositionLikePosteriorScoring

scoring_results = PositionLikePosteriorScoring.compute_and_plot_posterior_stack(
    posterior_stack,
    x_edges=xbin,
    y_edges=ybin, 
    should_plot_results=True,  # Must be True to generate the plot
    max_n_frames_to_plot=20,    # Number of time bins to display
)
```

### Key plotting code sections:

**Title generation** ([lines 1942-1945](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1942-1945)):
```python
title_str = (
    f"T[{res['t']}]: {res['label']}\n"
    f"Score: {res['score']:.2f} (S: {res['comps']['sharpness']:.2f} | L: {res['comps']['locality']:.2f})"
)
```

**Plot creation** ([lines 1924-1955](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L1924-1955)):
```python
rows = math.ceil(frames_to_process / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.7))
axes = np.atleast_1d(axes).flatten()

for i in range(len(axes)):
    if i < frames_to_process:
        res = scoring_results[i]
        ax = axes[i]
        im = ax.imshow(res['data'].T, origin='lower', cmap='inferno', aspect='auto')
        
        ax.set_title(title_str, color=res['color'], fontweight='bold', fontsize=9)
        ax.set_xticks([])  # Hide ticks for cleaner look
        ax.set_yticks([])
```

### Title format:
The titles appear as:
- **"T[{time_bin}]: Position-Like HQ"** (for high quality, score ≥ 0.7, in limegreen)
- **"T[{time_bin}]: Position-Like"** (for position-like, score ≥ 0.42, in green)  
- **"T[{time_bin}]: Ambiguous"** (for score < 0.42, in red)

Each subtitle shows: `Score: {score} (S: {sharpness} | L: {locality})`

The visualization displays posteriors as heatmaps using the 'inferno' colormap with titles color-coded by quality level.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis Position-Likeness title plot figure](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20Position-Likeness%20title%20plot%20figure)

