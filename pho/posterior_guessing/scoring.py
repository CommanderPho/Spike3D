"""Scoring helpers for painted vs true 2D posteriors."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def renormalize(weights: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Return a non-negative map renormalized to sum to 1.

    Parameters
    ----------
    weights:
        Raw painted weights (any shape). Negative values are clipped to 0.
    eps:
        If total mass <= eps, returns a uniform distribution over the same shape.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= eps:
        return np.full(w.shape, 1.0 / w.size, dtype=np.float64)
    return w / total


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance H(p, q) in [0, 1] for discrete distributions."""
    p_n = renormalize(p)
    q_n = renormalize(q)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_n) - np.sqrt(q_n)) ** 2)))


def hellinger_affinity(p: np.ndarray, q: np.ndarray) -> float:
    """Primary score: 1 - H(p, q), so identical maps score 1."""
    return float(1.0 - hellinger_distance(p, q))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Secondary score: cosine similarity of flattened renormalized maps."""
    p_n = renormalize(p).ravel()
    q_n = renormalize(q).ravel()
    denom = float(np.linalg.norm(p_n) * np.linalg.norm(q_n))
    if denom <= 0:
        return 0.0
    return float(np.dot(p_n, q_n) / denom)


def score_prediction(user_weights: np.ndarray, true_posterior: np.ndarray) -> Dict[str, float]:
    """Return hellinger_affinity + cosine_similarity for a user paint vs truth."""
    user_p = renormalize(user_weights)
    true_p = renormalize(true_posterior)
    return {
        'hellinger_affinity': hellinger_affinity(user_p, true_p),
        'hellinger_distance': hellinger_distance(user_p, true_p),
        'cosine_similarity': cosine_similarity(user_p, true_p),
    }


def assert_compatible_maps(user_weights: np.ndarray, true_posterior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    user = np.asarray(user_weights, dtype=np.float64)
    truth = np.asarray(true_posterior, dtype=np.float64)
    if user.shape != truth.shape:
        raise ValueError(f'shape mismatch: user {user.shape} vs true {truth.shape}')
    return user, truth
