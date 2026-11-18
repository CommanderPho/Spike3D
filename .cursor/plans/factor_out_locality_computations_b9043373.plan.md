---
name: Factor out locality computations
overview: Extract the computation logic from lines 616-719 into a standalone @classmethod that takes `a_p_x_given_n` and required parameters, and returns a dictionary of computed measures.
todos: []
---

# Factor Out Locality Computations into Independent Classmethod

## Overview

Extract the computation logic from `compute_locality_measures` method (lines 616-719) into a completely independent `@classmethod` that can be called on `a_p_x_given_n` without requiring an instance of `DecodingLocalityMeasures`.

## Implementation Details

### 1. Create New Classmethod

Add a new `@classmethod` to the `DecodingLocalityMeasures` class in [`pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py):**Method Signature:**

```python
@classmethod
def compute_locality_measures_for_posterior(cls, a_p_x_given_n, gaussian_volume, xbin_centers, ybin_centers, n_total_pos_bins=None, min_val_epsilon=1e-9, enable_debug_outputs=True, earthmovers_fn=None):
```

**Parameters:**

- `a_p_x_given_n`: NDArray with shape `(N_X_BINS, N_Y_BINS, N_TIME_BINS)` - the posterior probability distribution
- `gaussian_volume`: NDArray with shape `(N_X_BINS, N_Y_BINS, N_TIME_BINS)` - the gaussian spread volume
- `xbin_centers`: NDArray - x-axis bin centers
- `ybin_centers`: NDArray - y-axis bin centers  
- `n_total_pos_bins`: Optional[int] - total number of position bins (computed from xbin_centers/ybin_centers if None)
- `min_val_epsilon`: float - minimum value threshold (default: 1e-9)
- `enable_debug_outputs`: bool - whether to compute and return debug outputs (default: True)
- `earthmovers_fn`: Optional[Callable] - optional function for computing earthmovers distance (default: None)

**Returns:**

- `Dict[str, Any]` containing:
- `'mask_overlap'`: NDArray
- `'peak_prom'`: NDArray
- `'peak_prom_num_bins'`: NDArray
- `'peak_prom_Focality'`: NDArray
- `'peak_prom_Peakiness'`: NDArray
- `'peak_prom_num_peaks'`: NDArray (if computation succeeds)
- `'dist_to_highest_peak'`: NDArray
- `'earthmovers'`: NDArray (only if `earthmovers_fn` is provided)
- `'debug'`: Dict[str, Any] (only if `enable_debug_outputs=True`)

### 2. Include Helper Functions

The classmethod will include:

- `safe_nanmax` function (from line 324)
- `_subfn_pdf_spatial_distances` function (from line 515, modified to not require `_obj`)

### 3. Required Imports

The classmethod will need these imports:

- `from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence`
- Standard numpy operations

### 4. Update Original Method

Modify `compute_locality_measures` (starting at line 616) to call the new classmethod:

```python
computation_results = cls.compute_locality_measures_for_posterior(
    a_p_x_given_n=a_p_x_given_n,
    gaussian_volume=self.gaussian_volume,
    xbin_centers=self.xbin_centers,
    ybin_centers=self.ybin_centers,
    n_total_pos_bins=self.n_total_pos_bins,
    min_val_epsilon=min_val_epsilon,
    enable_debug_outputs=enable_debug_outputs,
    earthmovers_fn=active_subfn_compute_earthmovers_fn
)

# Extract results into self.locality_measures_dict_dict[an_epoch_name]
for key, value in computation_results.items():
    if key != 'debug':
        self.locality_measures_dict_dict[an_epoch_name][key] = value

if enable_debug_outputs and 'debug' in computation_results:
    self.debugging_dict_dict[an_epoch_name].update(computation_results['debug'])
```



### 5. Handle Edge Cases