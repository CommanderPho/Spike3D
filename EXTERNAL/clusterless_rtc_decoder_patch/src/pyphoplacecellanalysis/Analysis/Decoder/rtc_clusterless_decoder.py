from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from neuropy.utils.mixins.binning_helpers import BinningContainer, compute_spanning_bins
from replay_trajectory_classification import ClusterlessClassifier

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import (
    ClusterlessDecodingParameters,
    build_clusterless_training_data_from_pfnd,
    build_rtc_environment_from_pfnd,
    most_likely_positions_from_posterior,
    rtc_posterior_to_p_x_given_n,
)
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field


@custom_define(slots=False, eq=False)
class ClusterlessRTCPositionDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder):
    """Clusterless position decoder using replay_trajectory_classification on PfND spatial grids."""

    sampling_frequency_hz: float = 1000.0
    multiunits: np.ndarray = None
    rtc_time: np.ndarray = None
    clusterless_params: ClusterlessDecodingParameters = None
    classifier: ClusterlessClassifier = non_serialized_field(default=None, repr=False)
    rtc_results: xr.Dataset = non_serialized_field(default=None, repr=False)
    p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    flat_p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    revised_most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_position_flat_indicies: np.ndarray = non_serialized_field(default=None, repr=False)
    time_binning_container: BinningContainer = non_serialized_field(default=None, repr=False)
    is_training_mask: np.ndarray = non_serialized_field(default=None, repr=False)


    @property
    def time_bin_size(self) -> float:
        return 1.0 / float(self.sampling_frequency_hz)


    @property
    def time_window_centers(self) -> np.ndarray:
        return self.rtc_time


    @property
    def time_window_edges(self) -> np.ndarray:
        half_bin = self.time_bin_size / 2.0
        return np.concatenate([[self.rtc_time[0] - half_bin], self.rtc_time + half_bin])


    @property
    def num_time_windows(self) -> int:
        return len(self.rtc_time) if self.rtc_time is not None else 0


    @property
    def total_spike_counts_per_window(self) -> np.ndarray:
        if self.multiunits is None:
            return None
        return np.sum(np.any(np.isfinite(self.multiunits), axis=1), axis=1)


    @property
    def is_non_firing_time_bin(self) -> np.ndarray:
        spike_counts = self.total_spike_counts_per_window
        if spike_counts is None:
            return np.zeros(self.num_time_windows, dtype=bool)
        return spike_counts == 0


    def compute_all(self, debug_print: bool = False) -> None:
        if self.multiunits is None or self.rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time before compute_all().")
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        environment = build_rtc_environment_from_pfnd(self.pf, environment_name=params.rtc_environment_name, place_bin_size_override=params.rtc_place_bin_size_override)
        self.classifier = ClusterlessClassifier(environments=[environment], clusterless_algorithm="multiunit_likelihood", clusterless_algorithm_params={"mark_std": params.rtc_mark_std, "position_std": params.rtc_position_std})
        position_train, multiunits_train, is_training = build_clusterless_training_data_from_pfnd(self.pf, self.multiunits, self.rtc_time, self.sampling_frequency_hz)
        self.is_training_mask = is_training
        self.classifier.fit(position_train, multiunits_train, is_training=is_training)
        self.rtc_results = self.classifier.predict(multiunits_train, time=self.rtc_time[:len(multiunits_train)], is_compute_acausal=True, use_gpu=False)
        self.p_x_given_n = rtc_posterior_to_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior)
        self.flat_p_x_given_n = self.p_x_given_n.reshape(self.flat_position_size, self.num_time_windows) if self.p_x_given_n.ndim > 2 else self.p_x_given_n
        self.most_likely_positions = most_likely_positions_from_posterior(self.p_x_given_n, self.pf)
        self.revised_most_likely_positions = self.most_likely_positions.copy()
        self.most_likely_position_flat_indicies = np.argmax(self.p_x_given_n, axis=0)
        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(self.rtc_time, bin_size=self.time_bin_size)
        self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        if debug_print or self.debug_print:
            print(f"ClusterlessRTCPositionDecoder.compute_all(): p_x_given_n.shape={self.p_x_given_n.shape}, most_likely_positions.shape={np.shape(self.most_likely_positions)}")
