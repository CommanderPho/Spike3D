"""Synthetic 2D placefield / decoder bundle for demos and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.special import factorial

from pho.posterior_guessing.export import export_from_arrays


PathLike = Union[str, Path]


def _gaussian_2d(xx: np.ndarray, yy: np.ndarray, mu_x: float, mu_y: float, sigma: float, peak_rate: float) -> np.ndarray:
    return peak_rate * np.exp(-0.5 * (((xx - mu_x) / sigma) ** 2 + ((yy - mu_y) / sigma) ** 2))


def zhang_posterior_2d(tau: float, P_x: np.ndarray, tuning_curves: np.ndarray, spike_counts_t: np.ndarray) -> np.ndarray:
    """Compute P(x|n) for one time bin from 2D tuning curves.

    Parameters
    ----------
    tau:
        Time bin size.
    P_x:
        Prior over (n_x, n_y), positive mass.
    tuning_curves:
        (n_neurons, n_x, n_y) firing rates.
    spike_counts_t:
        (n_neurons,) spike counts in the bin.
    """
    n_neurons, n_x, n_y = tuning_curves.shape
    assert spike_counts_t.shape == (n_neurons,)

    # Work in flat spatial space for the product
    F_flat = tuning_curves.reshape(n_neurons, n_x * n_y).T  # (n_pos, n_neurons)
    prior = np.asarray(P_x, dtype=np.float64).reshape(n_x * n_y)
    prior = prior / np.sum(prior)

    cell_prob = np.ones(n_x * n_y, dtype=np.float64)
    for cell in range(n_neurons):
        n_i = float(spike_counts_t[cell])
        f_i = F_flat[:, cell]
        coeff = 1.0 / float(factorial(n_i))
        cell_prob *= ((tau * f_i) ** n_i) * coeff * np.exp(-tau * f_i)
    ## END for cell in range(n_neurons)...

    posterior = prior * cell_prob
    total = float(np.sum(posterior))
    if total <= 0 or not np.isfinite(total):
        posterior = prior.copy()
    else:
        posterior = posterior / total
    return posterior.reshape(n_x, n_y)


def make_synthetic_bundle_arrays(*, n_neurons: int = 8, n_x: int = 20, n_y: int = 16, n_time: int = 40, time_bin_size: float = 0.25, seed: int = 0, arena_xy: tuple = ((0.0, 100.0), (0.0, 80.0))) -> Dict[str, Any]:
    """Create in-memory synthetic 2D decoding arrays (no disk write)."""
    rng = np.random.default_rng(seed)
    (x_min, x_max), (y_min, y_max) = arena_xy
    xbin = np.linspace(x_min, x_max, n_x + 1)
    ybin = np.linspace(y_min, y_max, n_y + 1)
    xcent = 0.5 * (xbin[:-1] + xbin[1:])
    ycent = 0.5 * (ybin[:-1] + ybin[1:])
    xx, yy = np.meshgrid(xcent, ycent, indexing='ij')

    neuron_ids = np.arange(1, n_neurons + 1)
    tuning_curves = np.zeros((n_neurons, n_x, n_y), dtype=np.float64)
    for i in range(n_neurons):
        mu_x = rng.uniform(x_min + 10, x_max - 10)
        mu_y = rng.uniform(y_min + 10, y_max - 10)
        sigma = rng.uniform(8.0, 18.0)
        peak = rng.uniform(4.0, 18.0)
        tuning_curves[i] = _gaussian_2d(xx, yy, mu_x, mu_y, sigma, peak) + 0.05
    ## END for i in range(n_neurons)...

    # Soft occupancy biased toward center
    occupancy = np.exp(-0.5 * (((xx - xx.mean()) / (0.35 * (x_max - x_min))) ** 2 + ((yy - yy.mean()) / (0.35 * (y_max - y_min))) ** 2))
    occupancy = occupancy * 10.0 + 0.5
    P_x = occupancy / np.sum(occupancy)

    # Simulate animal trajectory as random walk on grid, sample Poisson spikes
    ix = n_x // 2
    iy = n_y // 2
    spike_counts = np.zeros((n_neurons, n_time), dtype=np.int64)
    p_x_given_n = np.zeros((n_x, n_y, n_time), dtype=np.float64)
    for t in range(n_time):
        ix = int(np.clip(ix + rng.integers(-2, 3), 0, n_x - 1))
        iy = int(np.clip(iy + rng.integers(-2, 3), 0, n_y - 1))
        local_rates = tuning_curves[:, ix, iy]
        counts = rng.poisson(local_rates * time_bin_size)
        # Ensure a few bins are empty and most have activity
        if t % 11 == 0:
            counts = np.zeros_like(counts)
        spike_counts[:, t] = counts
        p_x_given_n[:, :, t] = zhang_posterior_2d(time_bin_size, P_x, tuning_curves, counts.astype(np.float64))
    ## END for t in range(n_time)...

    time_bin_centers = (np.arange(n_time, dtype=np.float64) + 0.5) * time_bin_size
    return {
        'neuron_ids': neuron_ids,
        'tuning_curves': tuning_curves,
        'occupancy': occupancy,
        'P_x': P_x,
        'spike_counts': spike_counts,
        'p_x_given_n': p_x_given_n,
        'time_bin_centers': time_bin_centers,
        'xbin': xbin,
        'ybin': ybin,
        'time_bin_size': float(time_bin_size),
        'metadata': {
            'session_id': 'synthetic_demo',
            'source': 'make_synthetic_bundle',
            'seed': int(seed),
            'ndim': 2,
        },
    }


def make_synthetic_bundle(out_path: Optional[PathLike] = None, *, filter_empty_bins: bool = True, **kwargs) -> Path:
    """Create and optionally save a synthetic guessing bundle.

    Default out_path: data/posterior_guessing/synthetic_demo_pf2d_guessing.npz
    """
    arrays = make_synthetic_bundle_arrays(**kwargs)
    if out_path is None:
        out_path = Path('data/posterior_guessing/synthetic_demo_pf2d_guessing.npz')
    return export_from_arrays(
        neuron_ids=arrays['neuron_ids'],
        tuning_curves=arrays['tuning_curves'],
        occupancy=arrays['occupancy'],
        P_x=arrays['P_x'],
        spike_counts=arrays['spike_counts'],
        p_x_given_n=arrays['p_x_given_n'],
        time_bin_centers=arrays['time_bin_centers'],
        xbin=arrays['xbin'],
        ybin=arrays['ybin'],
        time_bin_size=arrays['time_bin_size'],
        out_path=out_path,
        metadata=arrays['metadata'],
        filter_empty_bins=filter_empty_bins,
    )
