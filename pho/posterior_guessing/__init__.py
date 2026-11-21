"""Portable 2D placefield / decoder bundles and posterior-guessing helpers."""

from pho.posterior_guessing.bundle_io import BUNDLE_REQUIRED_KEYS, load_bundle, save_bundle, validate_bundle
from pho.posterior_guessing.export import export_from_arrays, export_from_decoder
from pho.posterior_guessing.persistence import save_prediction
from pho.posterior_guessing.scoring import cosine_similarity, hellinger_affinity, renormalize, score_prediction
from pho.posterior_guessing.synthetic import make_synthetic_bundle


__all__ = [
    'BUNDLE_REQUIRED_KEYS',
    'cosine_similarity',
    'export_from_arrays',
    'export_from_decoder',
    'hellinger_affinity',
    'load_bundle',
    'make_synthetic_bundle',
    'renormalize',
    'save_bundle',
    'save_prediction',
    'score_prediction',
    'validate_bundle',
]
