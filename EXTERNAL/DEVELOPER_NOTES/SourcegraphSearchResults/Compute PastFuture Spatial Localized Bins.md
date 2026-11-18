---
title: Compute Past/Future Spatial Localized Bins in pyPhoPlaceCellAnalysis
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/9915c697-7638-4543-b319-4d9f1d9eaa0b
url: https://phohale.sourcegraph.app/deepsearch/361
created: 2026-04-01T20:35:21Z
updated: 2026-04-01T20:37:14Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis Find the steps for computing the Past/Future results involving the spatially localized/non-local decoded bins and the trajectories. Provide both code comment references and a professionally PPTX-type summary of the steps

Perfect! Now I have a comprehensive understanding. Let me create a professional summary for you.

Based on my analysis of your codebase, I found the comprehensive heuristic replay scoring pipeline. Here's the professional summary:

---

## **Past/Future Decoded Replay Analysis Pipeline**
### Summary of Steps for Computing Spatially Localized/Non-Local Decoded Bins and Trajectories

---

### **Overview**
The pipeline analyzes decoded posterior probability distributions from Bayesian position decoders to identify and score replay events (ripples/laps) based on spatial trajectory coherence and temporal sequencing.

---

### **STEP 1: Decode Epochs Using Bayesian Position Decoder**

**File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

**Process**:
- **Input**: Spike data during specific epochs (ripples or laps), position bin edges, decoder parameters
- **Computation**: Generate posterior probability distributions `p_x_given_n` for each time bin
- **Output**: `DecodedFilterEpochsResult` containing:
  - `p_x_given_n_list`: List of 2D arrays (n_position_bins × n_time_bins) for each epoch
  - `time_bin_containers`: Time bin centers and edges for each epoch
  - `most_likely_positions_list`: Maximum likelihood position for each time bin

**Key Code Reference**: 
```python
# Line ~986-1015 in reconstruction.py
def compute_radon_transforms(self, pos_bin_size:float, xbin_centers: NDArray, 
                            nlines:int=8192, margin:float=8, jump_stat=None, 
                            n_jobs:int=4) -> pd.DataFrame
```

---

### **STEP 2: Extract Most-Likely Position Trajectory**

**File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py)

**Process**:
- **Extract** `flat_positions`: Most-likely position (in cm) for each decoded time bin
- **Compute** first-order differences between sequential positions
- **Identify** direction changes, jumps, and spatial continuity

**Key Code Reference** (Lines 157-200):
```python
@define(slots=False)
class SubsequencesPartitioningResult(ComputedResult):
    flat_positions: NDArray  # most-likely positions (cm) for each time bin
    pos_bin_edges: NDArray   # position bin boundaries
    max_ignore_bins: int = 2 # max bins to bridge over
    same_thresh: float = 4   # threshold (cm) for "same" position
    max_jump_distance_cm: Optional[float] = None  # max allowed jump
```

---

### **STEP 3: Partition Trajectory into Subsequences**

**File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py)

**Process**:
1. **Identify Direction Changes**: Split where Δposition changes sign or exceeds `max_jump_distance_cm`
2. **Filter Low-Magnitude Changes**: Mark bins with Δposition < `same_thresh` as "repeats"
3. **Create Initial Subsequences**: Partition at direction change points

**Outputs**:
- `split_positions_arrays`: Initial subsequences based on direction changes
- `low_magnitude_change_indicies`: Time bins with position changes < threshold
- `diff_split_indicies`: Indices where subsequence splits occur

**Key Metrics** (Lines 250-450):
```python
@property
def n_flat_position_bins(self) -> int:
    return len(self.flat_positions)

@property  
def longest_sequence_length(self) -> int:
    return int(np.nanmax(self.num_merged_subsequence_bins))
```

---

### **STEP 4: Merge Subsequences ("Bridging")**

**Process**:
- **Bridge Short Intrusions**: Merge subsequences separated by ≤ `max_ignore_bins` "intrusion" bins
- **Identify Main Sequence**: Find longest continuous subsequence after merging
- **Track Intrusion Bins**: Mark bins that were bridged over

**Outputs**:
- `merged_split_positions_arrays`: Subsequences after bridging
- `bridged_intrusion_bin_indicies`: Positions of bridged bins
- `subsequences_df`: DataFrame with properties of each subsequence

**Key Code Reference** (Line ~185):
```python
merged_split_positions_arrays: List[NDArray]  
# Subsequences from split_positions_arrays merged by bridging 
# over intrusive tbins (max length = max_ignore_bins)
```

---

### **STEP 5: Classify Bins as Localized vs. Non-Local**

**Criteria**:
- **Spatially Localized**: Bins in the main contiguous subsequence (longest coherent trajectory)
- **Non-Local (Intrusions)**: 
  - Bins outside main subsequence
  - Bins bridged over during merging
  - Bins with position jumps > `max_jump_distance_cm`

**Classification Storage** (Lines 192-194):
```python
position_bins_info_df: pd.DataFrame  
# Columns: 'is_intrusion', 'subsequence_idx', 'direction_change', etc.
```

---

### **STEP 6: Compute Heuristic Scores**

**File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py)

**Main Function** (Lines 3126-3224):
```python
@classmethod
def compute_all_heuristic_scores(cls, track_templates: TrackTemplates,
                                 a_decoded_filter_epochs_decoder_result_dict: Dict,
                                 max_ignore_bins:float=2, 
                                 same_thresh_cm: float=6.0,
                                 max_jump_distance_cm: float=60.0) -> Tuple
```

**Computed Metrics**:

#### **A. Sequence Length Metrics**
- `mseq_len`: Main subsequence length (# bins)
- `mseq_len_ignoring_intrusions`: Length excluding bridged bins  
- `mseq_len_ignoring_intrusions_and_repeats`: Length excluding intrusions AND repeats
- `mseq_len_ratio_ignoring_intrusions_and_repeats`: Ratio to total bins

#### **B. Spatial Coverage Metrics** (Lines 2545-2560)
- `mseq_tcov`: **Track coverage score** - fraction of track covered by main sequence
- `mseq_dtrav`: **Total distance traveled** along main sequence

#### **C. Jump Distance Metrics** (Lines 2610-2700)
- `max_jump_cm`: Maximum position jump between adjacent bins
- `bin_by_bin_position_jump_distance`: Jump distance for each bin
- `max_jump_cm_per_sec`: Maximum velocity (cm/s) of position jumps

#### **D. Continuity Metrics**
- `continuous_seq_len_ratio_no_repeats`: Ratio of contiguous bins (no repeats)
- `sweep_score`: How well trajectory "sweeps" across track
- `direction_change_bin_ratio`: Fraction of bins with direction changes

---

### **STEP 7: Compute Radon Transform (Velocity Fitting)**

**File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py)

**Process** (Lines 1041-1043):
```python
def get_radon_transform(posterior: NDArray, 
                       decoding_time_bin_duration:float,
                       pos_bin_size:float, 
                       nlines:int=5000, 
                       margin:float=16.0) -> pd.DataFrame:
    """Radon Transform to fit line to decoded replay epoch posteriors. 
    Gives score, velocity, and intercept."""
```

**Outputs**:
- `score`: Quality of linear fit
- `velocity`: Estimated replay velocity (cm/s)
- `intercept`: Starting position of linear trajectory

**Reference**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py) (Lines 6450-6456)

---

### **STEP 8: Compute Weighted Correlation**

**File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)

**Function** (Line 6346):
```python
def compute_weighted_correlations(
    decoder_decoded_epochs_result_dict: Dict[types.DecoderName, DecodedFilterEpochsResult],
    debug_print=False
)
```

**Purpose**: Correlate decoded posterior with expected position tuning curves

**Outputs**:
- `decoder_laps_weighted_corr_df_dict`: Weighted correlations for lap epochs
- `decoder_ripple_weighted_corr_df_dict`: Weighted correlations for ripple epochs

---

### **STEP 9: Aggregate Results into DataFrames**

**Storage Structure** (Lines 3215-3226):
```python
@define(slots=False)
class DecoderDecodedEpochsResult:
    decoder_laps_radon_transform_df_dict: Dict
    decoder_ripple_radon_transform_df_dict: Dict
    laps_weighted_corr_merged_df: pd.DataFrame
    ripple_weighted_corr_merged_df: pd.DataFrame
    decoder_laps_weighted_corr_df_dict: Dict
    decoder_ripple_weighted_corr_df_dict: Dict
```

**Final DataFrame Columns**:
- Sequence metrics: `mseq_len`, `mseq_tcov`, `mseq_dtrav`
- Heuristic scores: `longest_sequence_length_ratio`, `direction_change_bin_ratio`
- Radon transform: `score`, `velocity`, `intercept`
- Weighted correlation: correlation coefficients per decoder
- Timing: `start`, `stop`, `delta_aligned_start_t`

---

### **Key Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `same_thresh_cm` | 6.0 | Position difference threshold for "repeats" (cm) |
| `max_ignore_bins` | 2 | Max bins to bridge during subsequence merging |
| `max_jump_distance_cm` | 60.0 | Max allowed jump between adjacent bins (cm) |
| `nlines` | 8192 | Number of lines tested in Radon transform |
| `margin` | 8-16 | Margin for Radon transform computation |

---

### **Main Computation Entry Point**

**File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)

**Function** (Lines 7435-7524):
```python
@function_attributes(short_name='directional_decoders_epoch_heuristic_scoring')
def _decoded_epochs_heuristic_scoring(
    computation_result: ComputationResult,
    same_thresh_fraction_of_track: float=0.05,
    max_ignore_bins: float=2,
    max_jump_distance_cm: float=60.0,
    use_bin_units_instead_of_realworld: bool=False,
    **kwargs
) -> ComputationResult
```

---

### **Workflow Summary Diagram**

```
┌─────────────────────────────┐
│ 1. Decode Epochs            │ → p_x_given_n posteriors
│    (Bayesian Decoder)       │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 2. Extract Most-Likely      │ → flat_positions array
│    Position Trajectory      │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 3. Partition into           │ → split_positions_arrays
│    Subsequences (by dir)    │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 4. Merge Subsequences       │ → merged_split_positions_arrays
│    (Bridge Intrusions)      │ → bridged_intrusion_bin_indicies
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 5. Classify Bins            │ → is_intrusion flags
│    (Localized vs Non-Local) │ → position_bins_info_df
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 6. Compute Heuristic Scores │ → mseq_len, mseq_tcov, jumps
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 7. Radon Transform          │ → velocity, score, intercept
│    (Linear Fit)             │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 8. Weighted Correlation     │ → correlation coefficients
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│ 9. Aggregate into           │ → Final scored epochs DataFrame
│    Result DataFrames        │
└─────────────────────────────┘
```

---

### **Output Data Products**

**Per-Epoch Metrics DataFrame** includes:
- `mseq_len`, `mseq_len_ignoring_intrusions`, `mseq_len_ratio_ignoring_intrusions_and_repeats`
- `mseq_tcov` (track coverage), `mseq_dtrav` (distance traveled)
- `max_jump_cm`, `sweep_score`, `direction_change_bin_ratio`
- `score` (Radon), `velocity` (Radon), `intercept` (Radon)
- Weighted correlation coefficients

**Classification Arrays**:
- `position_bins_info_df['is_intrusion']`: Boolean array marking non-local bins
- `position_bins_info_df['subsequence_idx']`: Which subsequence each bin belongs to
- `subsequences_df['is_main']`: Identifies the main (longest) coherent sequence

---

This pipeline enables quantification of replay quality by distinguishing coherent spatial trajectories (localized bins) from fragmented or non-contiguous decoded positions (non-local bins), providing a comprehensive heuristic-based scoring framework for hippocampal replay analysis.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis past future decoded bins trajectories](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20past%20future%20decoded%20bins%20trajectories)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis radon transform posterior localized non-local](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20radon%20transform%20posterior%20localized%20non-local)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis radon](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20radon)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "nonlocal"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22nonlocal%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis spatial localized position bins classify trajectory](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20spatial%20localized%20position%20bins%20classify%20trajectory)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "P_Long" "P_Short"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22P_Long%22%20%22P_Short%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis time bin most likely position trajectory weighted correlation](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20time%20bin%20most%20likely%20position%20trajectory%20weighted%20correlation)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "weighted_corr"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22weighted_corr%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "actual_position" "most_likely"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22actual_position%22%20%22most_likely%22)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis compare measured position decoded posterior distance spatial](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20compare%20measured%20position%20decoded%20posterior%20distance%20spatial)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis "sequence" "heuristic" "decode"](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20%22sequence%22%20%22heuristic%22%20%22decode%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L3126-3250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L3126-3250)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L1900-2100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1900-2100)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L1-200](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L1-200)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L2500-2700](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L2500-2700)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py@26f673d?L250-450](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py?L250-450)

