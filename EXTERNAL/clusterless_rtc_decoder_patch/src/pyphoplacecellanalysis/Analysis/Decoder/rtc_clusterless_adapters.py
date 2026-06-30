"""Adapters between NeuroPy PfND / session data and replay_trajectory_classification clusterless decoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.analyses.placefields import PfND
from replay_trajectory_classification.environments import Environment


@dataclass
class ClusterlessDecodingParameters:
    clusterless_sampling_frequency_hz: float = 1000.0
    rtc_place_bin_size_override: Optional[float] = None
    rtc_mark_std: float = 24.0
    rtc_position_std: float = 6.0
    rtc_environment_name: str = ""
    state_index_for_posterior: Optional[int] = None


def _pfnd_place_bin_size(pf: PfND, place_bin_size_override: Optional[float] = None) -> float:
    if place_bin_size_override is not None:
        return float(place_bin_size_override)
    pos_bin_size = pf.pos_bin_size
    if isinstance(pos_bin_size, (tuple, list, np.ndarray)):
        return float(np.mean(pos_bin_size))
    return float(pos_bin_size)


def _pfnd_position_range(pf: PfND) -> Optional[np.ndarray]:
    grid_bin_bounds = pf.config.grid_bin_bounds
    if grid_bin_bounds is None:
        return None
    if pf.ndim == 1:
        bounds_1d = grid_bin_bounds[0] if isinstance(grid_bin_bounds[0], (tuple, list, np.ndarray)) else grid_bin_bounds
        return np.array([bounds_1d, None], dtype=object)
    x_bounds = grid_bin_bounds[0]
    y_bounds = grid_bin_bounds[1]
    return np.array([x_bounds, y_bounds], dtype=object)


def position_array_from_pfnd(pf: PfND) -> np.ndarray:
    pos_df = pf.filtered_pos_df
    if pf.ndim == 1:
        if 'lin_pos' in pos_df.columns:
            return pos_df['lin_pos'].to_numpy(dtype=float)[:, np.newaxis]
        return pos_df['x'].to_numpy(dtype=float)[:, np.newaxis]
    return np.column_stack([pos_df['x'].to_numpy(dtype=float), pos_df['y'].to_numpy(dtype=float)])


def build_rtc_environment_from_pfnd(pf: PfND, environment_name: str = "", place_bin_size_override: Optional[float] = None) -> Environment:
    return Environment(environment_name=environment_name, place_bin_size=_pfnd_place_bin_size(pf, place_bin_size_override=place_bin_size_override), position_range=_pfnd_position_range(pf), infer_track_interior=True)


def resample_position_to_rtc_clock(position_array: np.ndarray, source_times: np.ndarray, t_start: float, t_end: float, sampling_frequency_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    position_array = np.asarray(position_array, dtype=float)
    if position_array.ndim == 1:
        position_array = position_array[:, np.newaxis]
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    resampled_position = np.empty((len(rtc_time), position_array.shape[1]), dtype=float)
    for dim_idx in range(position_array.shape[1]):
        resampled_position[:, dim_idx] = np.interp(rtc_time, source_times, position_array[:, dim_idx])
    return rtc_time, resampled_position


def build_is_training_mask_from_pfnd(pf: PfND, rtc_time: np.ndarray, source_times: np.ndarray, source_speed: np.ndarray) -> np.ndarray:
    speed_at_rtc = np.interp(rtc_time, source_times, source_speed)
    return speed_at_rtc >= float(pf.config.speed_thresh)


def build_multiunits_from_array(multiunits: np.ndarray, time: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    multiunits = np.asarray(multiunits, dtype=float)
    if multiunits.ndim != 3:
        raise ValueError(f"multiunits must have shape (n_time, n_marks, n_electrodes); got {multiunits.shape}")
    return multiunits, time


def build_multiunits_from_rtc_simulation(n_runs: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from replay_trajectory_classification.clusterless_simulation import make_simulated_run_data
    time, position, _sampling_frequency, multiunits, _multiunits_spikes = make_simulated_run_data(n_runs=n_runs, **kwargs)
    return time, position[:, np.newaxis], multiunits, position


def _assign_spike_marks_to_multiunits(multiunits: np.ndarray, time_bin_indices: np.ndarray, electrode_indices: np.ndarray, mark_features: np.ndarray) -> None:
    n_marks = multiunits.shape[1]
    mark_features = np.asarray(mark_features, dtype=float)
    if mark_features.ndim == 1:
        mark_features = mark_features[np.newaxis, :]
    n_features = min(n_marks, mark_features.shape[-1])
    for spike_idx in range(len(time_bin_indices)):
        t_idx = int(time_bin_indices[spike_idx])
        e_idx = int(electrode_indices[spike_idx])
        if 0 <= t_idx < multiunits.shape[0] and 0 <= e_idx < multiunits.shape[2]:
            multiunits[t_idx, :n_features, e_idx] = mark_features[spike_idx, :n_features]


def build_multiunits_from_session(sess, sampling_frequency_hz: float, t_start: float, t_end: float, spikes_df: Optional[pd.DataFrame] = None, n_mark_dims: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    neurons = sess.neurons
    if neurons.waveforms is None:
        raise ValueError("Clusterless decoding requires session.neurons.waveforms to be populated with per-neuron waveform mark features, or pass a pre-built multiunits array via the pipeline multiunits= kwarg. Expected RTC tensor shape: (n_time, n_marks, n_electrodes) with NaN for no spike and non-NaN mark features for spikes. See replay_trajectory_classification notebook 03-Decoding_with_Clusterless_Spikes.")
    if spikes_df is None:
        spikes_df = sess.spikes_df
    spikes_df = spikes_df.copy()
    time_variable_name = sess.spikes.time_variable_name if hasattr(sess, 'spikes') else 't'
    if time_variable_name not in spikes_df.columns and 't_seconds' in spikes_df.columns:
        time_variable_name = 't_seconds'
    spike_times = spikes_df[time_variable_name].to_numpy(dtype=float)
    valid_spikes = (spike_times >= t_start) & (spike_times <= t_end)
    spike_times = spike_times[valid_spikes]
    spikes_df = spikes_df.loc[valid_spikes].reset_index(drop=True)
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    if neurons.shank_ids is not None:
        n_electrodes = int(np.max(np.asarray(neurons.shank_ids, dtype=int))) + 1
    elif neurons.peak_channels is not None:
        n_electrodes = int(np.max(np.asarray(neurons.peak_channels, dtype=int))) + 1
    else:
        n_electrodes = int(neurons.n_neurons)
    multiunits = np.full((len(rtc_time), n_mark_dims, n_electrodes), np.nan, dtype=float)
    aclu_column = 'aclu' if 'aclu' in spikes_df.columns else ('cluster' if 'cluster' in spikes_df.columns else None)
    if aclu_column is None:
        raise ValueError("spikes_df must contain 'aclu' or 'cluster' to map spikes to waveform marks.")
    reverse_map = neurons.reverse_cellID_index_map
    time_bin_indices = np.clip(np.searchsorted(rtc_time, spike_times), 0, len(rtc_time) - 1)
    electrode_indices = []
    mark_features = []
    waveforms = np.asarray(neurons.waveforms)
    for spike_row_idx, aclu in enumerate(spikes_df[aclu_column].to_numpy()):
        if aclu not in reverse_map:
            continue
        neuron_idx = reverse_map[int(aclu)]
        neuron_waveform = waveforms[neuron_idx]
        if neuron_waveform.ndim == 1:
            feature_vector = neuron_waveform
        elif neuron_waveform.ndim == 2:
            feature_vector = neuron_waveform[0] if neuron_waveform.shape[0] == 1 else neuron_waveform.mean(axis=0)
        else:
            feature_vector = neuron_waveform.reshape(-1)
        if neurons.shank_ids is not None:
            electrode_idx = int(neurons.shank_ids[neuron_idx])
        elif neurons.peak_channels is not None:
            electrode_idx = int(neurons.peak_channels[neuron_idx])
        else:
            electrode_idx = int(neuron_idx)
        electrode_indices.append(electrode_idx)
        mark_features.append(feature_vector)
    if len(mark_features) == 0:
        raise ValueError("No spikes could be mapped to waveform marks for clusterless decoding.")
    _assign_spike_marks_to_multiunits(multiunits, time_bin_indices[:len(mark_features)], np.asarray(electrode_indices, dtype=int), np.asarray(mark_features, dtype=float))
    return multiunits, rtc_time


def rtc_posterior_to_p_x_given_n(rtc_results: xr.Dataset, pf: PfND, state_index: Optional[int] = None) -> np.ndarray:
    posterior = rtc_results.acausal_posterior.values
    if posterior.ndim == 4:
        posterior = posterior[..., 0]
    if state_index is None:
        posterior_time_bins = posterior.sum(axis=1)
    else:
        posterior_time_bins = posterior[:, state_index, :]
    p_x_given_n_rtc = posterior_time_bins.T
    n_pf_bins = int(np.prod(np.shape(pf.occupancy)))
    n_rtc_bins = p_x_given_n_rtc.shape[0]
    if n_rtc_bins == n_pf_bins:
        return p_x_given_n_rtc
    n_time = p_x_given_n_rtc.shape[1]
    if n_rtc_bins > n_pf_bins:
        return p_x_given_n_rtc[:n_pf_bins, :]
    padded = np.zeros((n_pf_bins, n_time), dtype=float)
    padded[:n_rtc_bins, :] = p_x_given_n_rtc
    return padded


def most_likely_positions_from_posterior(p_x_given_n: np.ndarray, pf: PfND) -> np.ndarray:
    most_likely_flat_indices = np.argmax(p_x_given_n, axis=0)
    if pf.ndim == 1:
        x_centers = pf.xbin_centers
        return x_centers[np.clip(most_likely_flat_indices, 0, len(x_centers) - 1)]
    x_centers = pf.xbin_centers
    y_centers = pf.ybin_centers
    n_x = len(x_centers)
    x_idx = most_likely_flat_indices // len(y_centers)
    y_idx = most_likely_flat_indices % len(y_centers)
    return np.column_stack([x_centers[np.clip(x_idx, 0, n_x - 1)], y_centers[np.clip(y_idx, 0, len(y_centers) - 1)]])


def build_clusterless_training_data_from_pfnd(pf: PfND, multiunits: np.ndarray, rtc_time: np.ndarray, sampling_frequency_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_pos_df = pf.filtered_pos_df
    source_times = source_pos_df['t'].to_numpy(dtype=float) if 't' in source_pos_df.columns else source_pos_df['t_seconds'].to_numpy(dtype=float)
    t_start = float(rtc_time[0] - 0.5 / sampling_frequency_hz)
    t_end = float(rtc_time[-1] + 0.5 / sampling_frequency_hz)
    position_source = position_array_from_pfnd(pf)
    _, resampled_position = resample_position_to_rtc_clock(position_source, source_times, t_start, t_end, sampling_frequency_hz)
    if len(resampled_position) != len(rtc_time):
        min_len = min(len(resampled_position), len(rtc_time))
        resampled_position = resampled_position[:min_len]
        multiunits = multiunits[:min_len]
        rtc_time = rtc_time[:min_len]
    source_speed = pf.speed
    is_training = build_is_training_mask_from_pfnd(pf, rtc_time, source_times, source_speed)
    return resampled_position, multiunits, is_training
