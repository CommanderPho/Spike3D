import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from replay_trajectory_classification import ClusterlessClassifier
from replay_trajectory_classification.environments import Environment

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import build_multiunits_from_rtc_simulation, rtc_posterior_to_p_x_given_n


class _MockPfND:
    def __init__(self, n_bins: int, ndim: int = 1):
        self.ndim = ndim
        self.occupancy = np.ones(n_bins if ndim == 1 else (n_bins, n_bins))
        self.xbin_centers = np.arange(n_bins, dtype=float)
        self.ybin_centers = np.arange(n_bins, dtype=float) if ndim == 2 else None


def test_build_multiunits_from_rtc_simulation_shapes():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    assert len(time) == multiunits.shape[0]
    assert position.shape[0] == multiunits.shape[0]
    assert multiunits.ndim == 3


def test_rtc_clusterless_classifier_simulation_roundtrip():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    classifier = ClusterlessClassifier(environments=[Environment(place_bin_size=6.0)])
    classifier.fit(position, multiunits)
    results = classifier.predict(multiunits)
    posterior = results.acausal_posterior.values
    assert posterior.ndim >= 2
    assert np.isfinite(posterior).any()


def test_rtc_posterior_to_p_x_given_n_shape():
    n_time, n_states, n_bins = 20, 2, 10
    mock_pf = _MockPfND(n_bins=n_bins, ndim=1)
    posterior = np.random.rand(n_time, n_states, n_bins)
    posterior /= posterior.sum(axis=-1, keepdims=True)
    rtc_results = xr.Dataset({"acausal_posterior": (("time", "state", "position"), posterior)})
    p_x_given_n = rtc_posterior_to_p_x_given_n(rtc_results, mock_pf)
    assert p_x_given_n.shape == (n_bins, n_time)


def test_position_decoding_clusterless_registered_in_default_computation_functions():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "General" / "Pipeline" / "Stages" / "ComputationFunctions" / "DefaultComputationFunctions.py"
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    method_names = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "DefaultComputationFunctions":
            method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
    assert "_perform_clusterless_position_decoding_computation" in method_names
    assert "position_decoding_clusterless" in source_text
