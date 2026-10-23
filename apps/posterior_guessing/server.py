"""FastAPI server for the 2D posterior-guessing webapp."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pho.posterior_guessing.bundle_io import load_bundle
from pho.posterior_guessing.persistence import save_prediction
from pho.posterior_guessing.scoring import renormalize, score_prediction


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / 'static'
DEFAULT_BUNDLE_DIR = REPO_ROOT / 'data' / 'posterior_guessing'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'outputs' / 'posterior_guessing'

app = FastAPI(title='2D Posterior Guessing', version='0.1.0')
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

_BUNDLE_CACHE: Dict[str, Dict[str, Any]] = {}


class GuessRequest(BaseModel):
    user_weights: List[List[float]] = Field(..., description='2D painted weights (n_x, n_y)')
    save: bool = True


def _bundle_dir() -> Path:
    return Path(DEFAULT_BUNDLE_DIR)


def _list_bundle_paths() -> List[Path]:
    d = _bundle_dir()
    if not d.exists():
        return []
    return sorted(d.glob('*_pf2d_guessing.npz'))


def _get_bundle(bundle_id: str) -> Dict[str, Any]:
    if bundle_id in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[bundle_id]

    # Prefer exact stem match under data/posterior_guessing
    candidates = []
    for p in _list_bundle_paths():
        data = None
        # Match by filename stem or metadata session_id without loading all eagerly if name matches
        if p.stem == bundle_id or p.stem.replace('_pf2d_guessing', '') == bundle_id:
            candidates.append(p)
    ## END for p in _list_bundle_paths()...

    if not candidates:
        # Fallback: load each to check metadata session_id
        for p in _list_bundle_paths():
            loaded = load_bundle(p, validate=True)
            if loaded['bundle_id'] == bundle_id or p.stem == bundle_id:
                _BUNDLE_CACHE[bundle_id] = loaded
                _BUNDLE_CACHE[loaded['bundle_id']] = loaded
                return loaded
        ## END for p in _list_bundle_paths()...
        raise HTTPException(status_code=404, detail=f'Bundle not found: {bundle_id}')

    loaded = load_bundle(candidates[0], validate=True)
    _BUNDLE_CACHE[bundle_id] = loaded
    _BUNDLE_CACHE[loaded['bundle_id']] = loaded
    _BUNDLE_CACHE[Path(loaded['path']).stem] = loaded
    return loaded


@app.get('/')
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/api/health')
def health() -> Dict[str, Any]:
    return {'ok': True, 'bundle_dir': str(_bundle_dir())}


@app.get('/api/bundles')
def list_bundles() -> Dict[str, Any]:
    items = []
    for p in _list_bundle_paths():
        try:
            data = load_bundle(p, validate=True)
            items.append({
                'bundle_id': data['bundle_id'],
                'path': data['path'],
                'n_neurons': int(np.asarray(data['neuron_ids']).shape[0]),
                'n_x': int(data['tuning_curves'].shape[1]),
                'n_y': int(data['tuning_curves'].shape[2]),
                'n_time': int(data['spike_counts'].shape[1]),
                'time_bin_size': float(np.asarray(data['time_bin_size']).reshape(())),
                'metadata': data.get('metadata', {}),
            })
            _BUNDLE_CACHE[data['bundle_id']] = data
        except Exception as exc:
            items.append({'path': str(p), 'error': str(exc)})
    ## END for p in _list_bundle_paths()...
    return {'bundles': items}


@app.get('/api/bundles/{bundle_id}')
def bundle_summary(bundle_id: str) -> Dict[str, Any]:
    data = _get_bundle(bundle_id)
    n_x, n_y, n_time = data['p_x_given_n'].shape
    return {
        'bundle_id': data['bundle_id'],
        'path': data['path'],
        'n_neurons': int(data['neuron_ids'].shape[0]),
        'n_x': n_x,
        'n_y': n_y,
        'n_time': n_time,
        'time_bin_size': float(np.asarray(data['time_bin_size']).reshape(())),
        'time_bin_centers': np.asarray(data['time_bin_centers'], dtype=float).tolist(),
        'xbin': np.asarray(data['xbin'], dtype=float).tolist(),
        'ybin': np.asarray(data['ybin'], dtype=float).tolist(),
        'neuron_ids': np.asarray(data['neuron_ids']).tolist(),
        'occupancy': np.asarray(data['occupancy'], dtype=float).tolist(),
        'metadata': data.get('metadata', {}),
    }


@app.get('/api/bundles/{bundle_id}/bins/{bin_index}')
def get_bin(bundle_id: str, bin_index: int) -> Dict[str, Any]:
    """Return bin payload WITHOUT the true posterior (hidden until reveal)."""
    data = _get_bundle(bundle_id)
    n_time = int(data['spike_counts'].shape[1])
    if bin_index < 0 or bin_index >= n_time:
        raise HTTPException(status_code=404, detail=f'bin_index out of range 0..{n_time - 1}')

    counts = np.asarray(data['spike_counts'])[:, bin_index]
    active_mask = counts > 0
    active_idxs = np.where(active_mask)[0]
    neuron_ids = np.asarray(data['neuron_ids'])
    tuning = np.asarray(data['tuning_curves'], dtype=float)

    active_cells = []
    for i in active_idxs.tolist():
        active_cells.append({
            'neuron_id': neuron_ids[i].item() if hasattr(neuron_ids[i], 'item') else neuron_ids[i],
            'neuron_index': int(i),
            'spike_count': int(counts[i]),
            'tuning_curve': tuning[i].tolist(),
        })
    ## END for i in active_idxs.tolist()...

    n_x, n_y = tuning.shape[1], tuning.shape[2]
    return {
        'bundle_id': data['bundle_id'],
        'bin_index': int(bin_index),
        'time_bin_center': float(np.asarray(data['time_bin_centers'])[bin_index]),
        'time_bin_size': float(np.asarray(data['time_bin_size']).reshape(())),
        'n_x': n_x,
        'n_y': n_y,
        'xbin': np.asarray(data['xbin'], dtype=float).tolist(),
        'ybin': np.asarray(data['ybin'], dtype=float).tolist(),
        'total_spikes': int(np.sum(counts)),
        'active_cells': active_cells,
        # Intentionally omit p_x_given_n
    }


@app.post('/api/bundles/{bundle_id}/bins/{bin_index}/reveal')
def reveal_bin(bundle_id: str, bin_index: int, body: GuessRequest) -> Dict[str, Any]:
    data = _get_bundle(bundle_id)
    n_x, n_y, n_time = data['p_x_given_n'].shape
    if bin_index < 0 or bin_index >= n_time:
        raise HTTPException(status_code=404, detail=f'bin_index out of range 0..{n_time - 1}')

    user = np.asarray(body.user_weights, dtype=np.float64)
    if user.shape != (n_x, n_y):
        raise HTTPException(status_code=400, detail=f'user_weights must have shape ({n_x}, {n_y}); got {user.shape}')
    if float(np.sum(np.clip(user, 0.0, None))) <= 0:
        raise HTTPException(status_code=400, detail='Paint a non-zero posterior before revealing')

    truth = np.asarray(data['p_x_given_n'][:, :, bin_index], dtype=np.float64)
    scores = score_prediction(user, truth)
    user_p = renormalize(user)
    truth_p = renormalize(truth)

    saved = None
    if body.save:
        saved = save_prediction(
            bundle_id=data['bundle_id'],
            time_bin_index=bin_index,
            user_weights=user,
            true_posterior=truth,
            scores=scores,
            root=DEFAULT_OUTPUT_ROOT,
            extra={'time_bin_center': float(np.asarray(data['time_bin_centers'])[bin_index])},
            save_npz=True,
        )

    return {
        'bundle_id': data['bundle_id'],
        'bin_index': int(bin_index),
        'scores': scores,
        'user_posterior': user_p.tolist(),
        'true_posterior': truth_p.tolist(),
        'saved': saved,
    }
