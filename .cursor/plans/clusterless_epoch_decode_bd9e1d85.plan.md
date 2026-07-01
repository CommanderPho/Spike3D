---
name: Clusterless Epoch Decode
overview: Implement `ClusterlessRTCPositionDecoder.decode_specific_epochs` by ignoring `spikes_df`, slicing the decoder's prebuilt clusterless `multiunits`/`rtc_time` over requested epochs, and decoding each epoch with the already fitted RTC classifier path.
todos:
  - id: add-tests
    content: Add failing tests for clusterless decode_specific_epochs behavior, ignored spikes_df, and single-bin epochs.
    status: completed
  - id: implement-method
    content: Implement clusterless epoch slicing and DecodedFilterEpochsResult assembly in rtc_clusterless_decoder.py.
    status: completed
  - id: verify
    content: Run focused pytest file with uv and fix any introduced lint/test issues.
    status: completed
isProject: false
---

# Clusterless Epoch Decode Plan

## Design

Update [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\rtc_clusterless_decoder.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\rtc_clusterless_decoder.py) so `decode_specific_epochs(...)` no longer raises. It will ignore `spikes_df`, validate `self.multiunits` and `self.rtc_time`, convert `filter_epochs` to a DataFrame, and decode each epoch by passing the epoch's sliced `multiunits` and `rtc_time` into `self.decode(...)`.

This matches the RTC “Predicting Position” flow, where a fitted clusterless model predicts with `decoder.predict(multiunits[time_ind], time=time[time_ind])` ([RTC docs](https://replay-trajectory-classification.readthedocs.io/en/latest/_copied_over/notebooks/03-Decoding_with_Clusterless_Spikes.html#predicting-position)). In this codebase, `self.decode(...)` already calls `_predict_clusterless_posterior(...)`, and `_ensure_fitted_classifier(...)` reuses `self.classifier` if it exists.

## Implementation Details

- Keep the public signature unchanged:
  `def decode_specific_epochs(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby: Optional[float] = None, debug_print=False) -> "DecodedFilterEpochsResult":`
- Build per-epoch time windows from epoch `start`/`stop` and `self.rtc_time`. `decoding_time_bin_size` and `slideby` will be preserved on the result for API compatibility, but the actual decoded samples come from the RTC clock already stored in `self.rtc_time`.
- Support `use_single_time_bin_per_epoch=True` by selecting the nearest RTC sample to the epoch midpoint, producing one decoded bin for that epoch.
- For normal mode, include RTC samples whose bin centers fall inside `[start, stop]`. If an epoch is shorter than the RTC step and contains no center, select the nearest center within the epoch span when possible, otherwise return an empty epoch entry.
- Assemble a `DecodedFilterEpochsResult` directly with the same field layout used by `BasePositionDecoder._perform_decoding_specific_epochs`: `most_likely_positions_list`, `p_x_given_n_list`, `most_likely_position_indicies_list`, `marginal_x_list`, `marginal_y_list`, `marginal_z_list`, `spkcount`, `nbins`, `time_bin_containers`, `time_bin_edges`, `filter_epochs`, `decoding_time_bin_size`, `slideby`, and `pos_bin_edges`.
- Use `BasePositionDecoder.perform_build_marginals(...)` for each epoch so 1D/2D marginal behavior stays consistent with existing decoded epoch consumers.

## Tests

Modify [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_rtc_clusterless_decoder.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_rtc_clusterless_decoder.py):

- Replace the existing `NotImplementedError` test with a behavior test that creates simulated clusterless `multiunits`, builds a few epoch intervals, patches training data as the current decode test does, and asserts `decode_specific_epochs(...)` returns a `DecodedFilterEpochsResult` with matching epoch count, nonzero `nbins`, aligned time-bin centers, posterior list entries, and most-likely position entries.
- Add a test proving `spikes_df` is ignored by passing an empty DataFrame while decoding succeeds from `self.multiunits`.
- Add a small test for `use_single_time_bin_per_epoch=True` asserting each epoch has exactly one bin.

## Verification

Run the focused test file:

`uv run pytest h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_rtc_clusterless_decoder.py -q`

If imports or environment setup require running from the package directory, run the same command with working directory [`h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis).