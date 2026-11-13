"""Tests for posterior guessing bundle IO, scoring, and API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pho.posterior_guessing.bundle_io import load_bundle, save_bundle, validate_bundle
from pho.posterior_guessing.export import export_from_arrays
from pho.posterior_guessing.persistence import save_prediction
from pho.posterior_guessing.scoring import cosine_similarity, hellinger_affinity, renormalize, score_prediction
from pho.posterior_guessing.synthetic import make_synthetic_bundle, make_synthetic_bundle_arrays


@pytest.fixture()
def synthetic_arrays():
    return make_synthetic_bundle_arrays(n_neurons=5, n_x=12, n_y=10, n_time=15, seed=7)


def test_validate_and_roundtrip(tmp_path, synthetic_arrays):
    out = tmp_path / 'demo_pf2d_guessing.npz'
    path = export_from_arrays(
        neuron_ids=synthetic_arrays['neuron_ids'],
        tuning_curves=synthetic_arrays['tuning_curves'],
        occupancy=synthetic_arrays['occupancy'],
        P_x=synthetic_arrays['P_x'],
        spike_counts=synthetic_arrays['spike_counts'],
        p_x_given_n=synthetic_arrays['p_x_given_n'],
        time_bin_centers=synthetic_arrays['time_bin_centers'],
        xbin=synthetic_arrays['xbin'],
        ybin=synthetic_arrays['ybin'],
        time_bin_size=synthetic_arrays['time_bin_size'],
        out_path=out,
        metadata=synthetic_arrays['metadata'],
        filter_empty_bins=True,
    )
    loaded = load_bundle(path)
    validate_bundle(loaded)
    assert loaded['tuning_curves'].ndim == 3
    assert loaded['p_x_given_n'].shape[2] == loaded['spike_counts'].shape[1]
    assert loaded['bundle_id'] == 'synthetic_demo'


def test_make_synthetic_bundle_writes(tmp_path):
    path = make_synthetic_bundle(out_path=tmp_path / 'synthetic_demo_pf2d_guessing.npz', n_time=20, seed=1)
    assert path.exists()
    data = load_bundle(path)
    assert data['spike_counts'].shape[1] == data['p_x_given_n'].shape[2]


def test_scoring_identical_and_disjoint():
    p = renormalize(np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert hellinger_affinity(p, p) == pytest.approx(1.0, abs=1e-9)
    assert cosine_similarity(p, p) == pytest.approx(1.0, abs=1e-9)

    q = renormalize(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert hellinger_affinity(p, q) == pytest.approx(0.0, abs=1e-9)
    scores = score_prediction(p, q)
    assert scores['hellinger_affinity'] == pytest.approx(0.0, abs=1e-9)


def test_save_prediction(tmp_path, synthetic_arrays):
    truth = synthetic_arrays['p_x_given_n'][:, :, 1]
    user = truth.copy()
    record = save_prediction(
        bundle_id='unit_test',
        time_bin_index=1,
        user_weights=user,
        true_posterior=truth,
        root=tmp_path,
        save_npz=True,
    )
    assert record['hellinger_affinity'] == pytest.approx(1.0, abs=1e-6)
    jsonl = Path(record['jsonl_path'])
    assert jsonl.exists()
    line = jsonl.read_text(encoding='utf-8').strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed['bundle_id'] == 'unit_test'
    assert Path(record['npz_path']).exists()


@pytest.fixture()
def api_client(tmp_path, monkeypatch, synthetic_arrays):
    bundle_dir = tmp_path / 'bundles'
    bundle_dir.mkdir()
    out_root = tmp_path / 'outputs'
    path = export_from_arrays(
        neuron_ids=synthetic_arrays['neuron_ids'],
        tuning_curves=synthetic_arrays['tuning_curves'],
        occupancy=synthetic_arrays['occupancy'],
        P_x=synthetic_arrays['P_x'],
        spike_counts=synthetic_arrays['spike_counts'],
        p_x_given_n=synthetic_arrays['p_x_given_n'],
        time_bin_centers=synthetic_arrays['time_bin_centers'],
        xbin=synthetic_arrays['xbin'],
        ybin=synthetic_arrays['ybin'],
        time_bin_size=synthetic_arrays['time_bin_size'],
        out_path=bundle_dir / 'synthetic_demo_pf2d_guessing.npz',
        metadata=synthetic_arrays['metadata'],
        filter_empty_bins=True,
    )
    assert path.exists()

    import apps.posterior_guessing.server as server

    monkeypatch.setattr(server, 'DEFAULT_BUNDLE_DIR', bundle_dir)
    monkeypatch.setattr(server, 'DEFAULT_OUTPUT_ROOT', out_root)
    server._BUNDLE_CACHE.clear()
    return TestClient(server.app), server


def test_api_hides_posterior_until_reveal(api_client):
    client, server = api_client
    listing = client.get('/api/bundles')
    assert listing.status_code == 200
    bundles = listing.json()['bundles']
    assert len(bundles) == 1
    bundle_id = bundles[0]['bundle_id']

    bin_payload = client.get(f'/api/bundles/{bundle_id}/bins/0')
    assert bin_payload.status_code == 200
    body = bin_payload.json()
    assert 'p_x_given_n' not in body
    assert 'true_posterior' not in body
    assert 'active_cells' in body
    n_x, n_y = body['n_x'], body['n_y']

    # Empty paint rejected
    bad = client.post(f'/api/bundles/{bundle_id}/bins/0/reveal', json={'user_weights': np.zeros((n_x, n_y)).tolist(), 'save': False})
    assert bad.status_code == 400

    data = load_bundle(bundles[0]['path'])
    truth = np.asarray(data['p_x_given_n'][:, :, 0])
    # Use a noisy version of truth as the paint
    rng = np.random.default_rng(0)
    paint = np.clip(truth + 0.01 * rng.random(truth.shape), 0, None)

    revealed = client.post(f'/api/bundles/{bundle_id}/bins/0/reveal', json={'user_weights': paint.tolist(), 'save': True})
    assert revealed.status_code == 200
    result = revealed.json()
    assert 'true_posterior' in result
    assert 'scores' in result
    assert result['scores']['hellinger_affinity'] > 0.5
    assert result['saved'] is not None
