"""Persist user posterior guesses for later analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from pho.posterior_guessing.scoring import renormalize, score_prediction


PathLike = Union[str, Path]


def default_predictions_dir(bundle_id: str, root: PathLike = 'outputs/posterior_guessing') -> Path:
    return Path(root).expanduser().resolve() / str(bundle_id)


def save_prediction(*, bundle_id: str, time_bin_index: int, user_weights: np.ndarray, true_posterior: np.ndarray, scores: Optional[Mapping[str, float]] = None, root: PathLike = 'outputs/posterior_guessing', extra: Optional[Mapping[str, Any]] = None, save_npz: bool = True) -> Dict[str, Any]:
    """Score (if needed), append JSONL record, and optionally save per-bin NPZ.

    Returns the record dict that was appended.
    """
    user = np.asarray(user_weights, dtype=np.float64)
    truth = np.asarray(true_posterior, dtype=np.float64)
    if user.shape != truth.shape:
        raise ValueError(f'shape mismatch: user {user.shape} vs true {truth.shape}')

    user_p = renormalize(user)
    if scores is None:
        scores = score_prediction(user_p, truth)

    out_dir = default_predictions_dir(bundle_id, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    record: Dict[str, Any] = {
        'bundle_id': bundle_id,
        'time_bin_index': int(time_bin_index),
        'timestamp': timestamp,
        'hellinger_affinity': float(scores['hellinger_affinity']),
        'hellinger_distance': float(scores.get('hellinger_distance', 1.0 - float(scores['hellinger_affinity']))),
        'cosine_similarity': float(scores['cosine_similarity']),
        'user_mass': float(np.sum(np.clip(user, 0.0, None))),
        'shape': list(user.shape),
    }
    if extra:
        record.update(dict(extra))

    jsonl_path = out_dir / 'predictions.jsonl'
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    if save_npz:
        npz_path = out_dir / f'bin_{int(time_bin_index):05d}_{timestamp.replace(":", "").replace("-", "")}.npz'
        np.savez_compressed(
            npz_path,
            user_weights=user,
            user_posterior=user_p,
            true_posterior=renormalize(truth),
            time_bin_index=np.asarray(int(time_bin_index)),
            hellinger_affinity=np.asarray(float(scores['hellinger_affinity'])),
            cosine_similarity=np.asarray(float(scores['cosine_similarity'])),
            metadata_json=np.asarray(json.dumps(record)),
        )
        record['npz_path'] = str(npz_path)

    record['jsonl_path'] = str(jsonl_path)
    return record
