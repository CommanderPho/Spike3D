"""NPZ schema load/save/validate for 2D posterior-guessing bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import numpy as np


BUNDLE_REQUIRED_KEYS = (
    'neuron_ids',
    'tuning_curves',
    'occupancy',
    'P_x',
    'spike_counts',
    'p_x_given_n',
    'time_bin_centers',
    'xbin',
    'ybin',
    'time_bin_size',
    'metadata_json',
)


PathLike = Union[str, Path]


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def validate_bundle(data: Mapping[str, Any], *, require_positive_mass: bool = True) -> Dict[str, Any]:
    """Validate array shapes/dtypes for a guessing bundle; return normalized metadata dict.

    Raises
    ------
    KeyError
        Missing required keys.
    ValueError
        Inconsistent shapes or invalid probability mass.
    """
    missing = [k for k in BUNDLE_REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f'Missing required bundle keys: {missing}')

    neuron_ids = np.asarray(data['neuron_ids'])
    tuning_curves = np.asarray(data['tuning_curves'], dtype=np.float64)
    occupancy = np.asarray(data['occupancy'], dtype=np.float64)
    P_x = np.asarray(data['P_x'], dtype=np.float64)
    spike_counts = np.asarray(data['spike_counts'])
    p_x_given_n = np.asarray(data['p_x_given_n'], dtype=np.float64)
    time_bin_centers = np.asarray(data['time_bin_centers'], dtype=np.float64)
    xbin = np.asarray(data['xbin'], dtype=np.float64)
    ybin = np.asarray(data['ybin'], dtype=np.float64)
    time_bin_size = float(np.asarray(data['time_bin_size']).reshape(()))

    if tuning_curves.ndim != 3:
        raise ValueError(f'tuning_curves must be (n_neurons, n_x, n_y); got shape {tuning_curves.shape}')
    n_neurons, n_x, n_y = tuning_curves.shape
    if neuron_ids.shape != (n_neurons,):
        raise ValueError(f'neuron_ids shape {neuron_ids.shape} != ({n_neurons},)')
    if occupancy.shape != (n_x, n_y):
        raise ValueError(f'occupancy shape {occupancy.shape} != ({n_x}, {n_y})')
    if P_x.shape != (n_x, n_y):
        raise ValueError(f'P_x shape {P_x.shape} != ({n_x}, {n_y})')
    if spike_counts.ndim != 2 or spike_counts.shape[0] != n_neurons:
        raise ValueError(f'spike_counts must be (n_neurons, n_time); got {spike_counts.shape}')
    n_time = spike_counts.shape[1]
    if p_x_given_n.shape != (n_x, n_y, n_time):
        raise ValueError(f'p_x_given_n shape {p_x_given_n.shape} != ({n_x}, {n_y}, {n_time})')
    if time_bin_centers.shape != (n_time,):
        raise ValueError(f'time_bin_centers shape {time_bin_centers.shape} != ({n_time},)')
    if xbin.ndim != 1 or xbin.size != n_x + 1:
        raise ValueError(f'xbin must be length n_x+1={n_x + 1}; got shape {xbin.shape}')
    if ybin.ndim != 1 or ybin.size != n_y + 1:
        raise ValueError(f'ybin must be length n_y+1={n_y + 1}; got shape {ybin.shape}')
    if time_bin_size <= 0:
        raise ValueError(f'time_bin_size must be > 0; got {time_bin_size}')

    if require_positive_mass:
        prior_sum = float(np.sum(P_x))
        if not np.isfinite(prior_sum) or prior_sum <= 0:
            raise ValueError('P_x must have positive finite mass')
        # Check a sample of posterior columns for finite mass
        col_sums = np.sum(p_x_given_n.reshape(n_x * n_y, n_time), axis=0)
        if not np.all(np.isfinite(col_sums)):
            raise ValueError('p_x_given_n contains non-finite values')
        if np.any(col_sums <= 0):
            raise ValueError('each p_x_given_n[..., t] must have positive mass')

    metadata = parse_metadata(data['metadata_json'])
    return metadata


def parse_metadata(metadata_json: Any) -> Dict[str, Any]:
    """Parse metadata_json which may be str, bytes, or already a dict."""
    if isinstance(metadata_json, dict):
        return dict(metadata_json)
    if isinstance(metadata_json, (bytes, bytearray, np.bytes_)):
        metadata_json = bytes(metadata_json).decode('utf-8')
    if isinstance(metadata_json, np.ndarray):
        if metadata_json.shape == ():
            metadata_json = metadata_json.item()
        else:
            metadata_json = str(metadata_json)
    if not isinstance(metadata_json, str):
        metadata_json = str(metadata_json)
    metadata_json = metadata_json.strip()
    if not metadata_json:
        return {}
    return json.loads(metadata_json)


def metadata_to_json(metadata: Optional[Mapping[str, Any]] = None) -> str:
    return json.dumps(dict(metadata or {}), sort_keys=True)


def save_bundle(path: PathLike, data: Mapping[str, Any], *, validate: bool = True) -> Path:
    """Write a compressed NPZ bundle. Returns the resolved output path."""
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: MutableMapping[str, Any] = {k: data[k] for k in BUNDLE_REQUIRED_KEYS if k in data}
    # Allow callers to pass metadata as dict
    if 'metadata_json' in payload and not isinstance(payload['metadata_json'], (str, bytes, np.ndarray)):
        payload['metadata_json'] = metadata_to_json(payload['metadata_json'])
    elif 'metadata' in data and 'metadata_json' not in payload:
        payload['metadata_json'] = metadata_to_json(data['metadata'])

    if validate:
        validate_bundle(payload)

    # Ensure metadata is stored as a unicode string for NPZ
    payload['metadata_json'] = metadata_to_json(parse_metadata(payload['metadata_json']))
    np.savez_compressed(out_path, **payload)
    return out_path


def load_bundle(path: PathLike, *, validate: bool = True) -> Dict[str, Any]:
    """Load an NPZ guessing bundle into a plain dict of arrays (+ parsed metadata)."""
    in_path = _as_path(path)
    with np.load(in_path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}

    metadata = parse_metadata(data.get('metadata_json', '{}'))
    if validate:
        validate_bundle(data)
    data['metadata'] = metadata
    data['metadata_json'] = metadata_to_json(metadata)
    data['path'] = str(in_path)
    data['bundle_id'] = metadata.get('session_id') or in_path.stem
    return data


def filter_active_time_bins(data: Mapping[str, Any], *, min_total_spikes: int = 1) -> Dict[str, Any]:
    """Return a shallow-copied bundle keeping only time bins with enough spikes."""
    spike_counts = np.asarray(data['spike_counts'])
    keep = np.sum(spike_counts, axis=0) >= min_total_spikes
    if not np.any(keep):
        raise ValueError('No time bins meet the spike filter')

    out = dict(data)
    out['spike_counts'] = spike_counts[:, keep]
    out['p_x_given_n'] = np.asarray(data['p_x_given_n'])[:, :, keep]
    out['time_bin_centers'] = np.asarray(data['time_bin_centers'])[keep]
    metadata = parse_metadata(data.get('metadata_json', data.get('metadata', {})))
    metadata = dict(metadata)
    metadata['filtered_active_bins'] = True
    metadata['min_total_spikes'] = int(min_total_spikes)
    metadata['n_time_original'] = int(spike_counts.shape[1])
    metadata['n_time_kept'] = int(np.sum(keep))
    out['metadata'] = metadata
    out['metadata_json'] = metadata_to_json(metadata)
    return out
