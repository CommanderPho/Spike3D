"""Export helpers for 2D posterior-guessing NPZ bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from pho.posterior_guessing.bundle_io import filter_active_time_bins, metadata_to_json, save_bundle


PathLike = Union[str, Path]


def export_from_arrays(*, neuron_ids, tuning_curves, occupancy, P_x, spike_counts, p_x_given_n, time_bin_centers, xbin, ybin, time_bin_size: float, out_path: PathLike, metadata: Optional[Mapping[str, Any]] = None, filter_empty_bins: bool = True, min_total_spikes: int = 1) -> Path:
    """Build and save a guessing bundle from explicit arrays. Returns output path."""
    data: Dict[str, Any] = {
        'neuron_ids': np.asarray(neuron_ids),
        'tuning_curves': np.asarray(tuning_curves, dtype=np.float64),
        'occupancy': np.asarray(occupancy, dtype=np.float64),
        'P_x': np.asarray(P_x, dtype=np.float64),
        'spike_counts': np.asarray(spike_counts),
        'p_x_given_n': np.asarray(p_x_given_n, dtype=np.float64),
        'time_bin_centers': np.asarray(time_bin_centers, dtype=np.float64),
        'xbin': np.asarray(xbin, dtype=np.float64),
        'ybin': np.asarray(ybin, dtype=np.float64),
        'time_bin_size': np.asarray(float(time_bin_size), dtype=np.float64),
        'metadata_json': metadata_to_json(metadata),
    }
    if filter_empty_bins:
        data = filter_active_time_bins(data, min_total_spikes=min_total_spikes)
    return save_bundle(out_path, data, validate=True)


def _reshape_flat_spatial(flat: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    """Reshape flat position axis of length n_x*n_y into (n_x, n_y, ...)."""
    flat = np.asarray(flat)
    n_flat = n_x * n_y
    if flat.shape[0] != n_flat:
        raise ValueError(f'Expected leading axis length {n_flat} (n_x*n_y); got {flat.shape}')
    trailing = flat.shape[1:]
    return flat.reshape((n_x, n_y) + trailing)


def export_from_decoder(decoder: Any, out_path: PathLike, *, metadata: Optional[Mapping[str, Any]] = None, filter_empty_bins: bool = True, min_total_spikes: int = 1, session_id: Optional[str] = None) -> Path:
    """Export from a BayesianPlacemapPositionDecoder-like object when sibling packages are available.

    Expects 2D placefields: `decoder.pf.ratemap.tuning_curves` shaped (n_neurons, n_x, n_y)
    and `decoder.p_x_given_n` shaped (n_x, n_y, n_time) or flat (n_x*n_y, n_time).
    """
    pf = getattr(decoder, 'pf', None)
    if pf is None:
        raise ValueError('decoder has no .pf placefield attribute')
    ratemap = getattr(pf, 'ratemap', None) or getattr(pf, '_ratemap', None)
    if ratemap is None:
        raise ValueError('decoder.pf has no ratemap')

    tuning_curves = np.asarray(ratemap.tuning_curves, dtype=np.float64)
    if tuning_curves.ndim != 3:
        raise ValueError(f'Expected 2D tuning_curves (n_neurons, n_x, n_y); got {tuning_curves.shape}')
    n_neurons, n_x, n_y = tuning_curves.shape

    occupancy = np.asarray(ratemap.occupancy, dtype=np.float64)
    xbin = np.asarray(getattr(ratemap, 'xbin', None) if getattr(ratemap, 'xbin', None) is not None else pf.xbin, dtype=np.float64)
    ybin = np.asarray(getattr(ratemap, 'ybin', None) if getattr(ratemap, 'ybin', None) is not None else pf.ybin, dtype=np.float64)
    if ybin is None or np.asarray(ybin).size == 0:
        raise ValueError('2D export requires ybin edges')

    neuron_ids = np.asarray(getattr(decoder, 'neuron_IDs', getattr(ratemap, 'neuron_ids', np.arange(n_neurons))))
    if neuron_ids.shape != (n_neurons,):
        neuron_ids = np.asarray(list(neuron_ids))[:n_neurons]

    P_x = np.asarray(decoder.P_x, dtype=np.float64)
    if P_x.ndim == 2 and P_x.shape[1] == 1:
        P_x = _reshape_flat_spatial(P_x[:, 0], n_x, n_y)
    elif P_x.ndim == 1:
        P_x = _reshape_flat_spatial(P_x, n_x, n_y)
    elif P_x.shape != (n_x, n_y):
        raise ValueError(f'Could not reshape P_x with shape {P_x.shape} to ({n_x}, {n_y})')

    spike_counts = np.asarray(decoder.unit_specific_time_binned_spike_counts)
    p_x_given_n = np.asarray(decoder.p_x_given_n, dtype=np.float64)
    if p_x_given_n.ndim == 2:
        p_x_given_n = _reshape_flat_spatial(p_x_given_n, n_x, n_y)
    elif p_x_given_n.shape[:2] != (n_x, n_y):
        raise ValueError(f'p_x_given_n shape {p_x_given_n.shape} incompatible with ({n_x}, {n_y}, n_time)')

    # Align time axes if spike counts and posterior lengths differ (common in docs)
    n_time_spikes = spike_counts.shape[1]
    n_time_post = p_x_given_n.shape[2]
    n_time = min(n_time_spikes, n_time_post)
    spike_counts = spike_counts[:, :n_time]
    p_x_given_n = p_x_given_n[:, :, :n_time]

    tbc = getattr(getattr(decoder, 'time_binning_container', None), 'centers', None)
    if tbc is None:
        time_bin_centers = np.arange(n_time, dtype=np.float64) * float(decoder.time_bin_size)
    else:
        time_bin_centers = np.asarray(tbc, dtype=np.float64)[:n_time]

    meta = dict(metadata or {})
    if session_id is not None:
        meta['session_id'] = session_id
    meta.setdefault('source', 'export_from_decoder')
    meta.setdefault('ndim', 2)

    return export_from_arrays(
        neuron_ids=neuron_ids,
        tuning_curves=tuning_curves,
        occupancy=occupancy,
        P_x=P_x,
        spike_counts=spike_counts,
        p_x_given_n=p_x_given_n,
        time_bin_centers=time_bin_centers,
        xbin=xbin,
        ybin=ybin,
        time_bin_size=float(decoder.time_bin_size),
        out_path=out_path,
        metadata=meta,
        filter_empty_bins=filter_empty_bins,
        min_total_spikes=min_total_spikes,
    )
