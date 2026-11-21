#!/usr/bin/env python3
"""Export a portable 2D posterior-guessing NPZ bundle.

Examples
--------
  uv run python scripts/export_posterior_guessing_bundle.py --synthetic
  uv run python scripts/export_posterior_guessing_bundle.py --from-pkl /path/to/decoder_or_pipeline.pkl --out data/posterior_guessing/session_pf2d_guessing.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_object_from_pickle(pkl_path: Path):
    try:
        import dill as pickle  # type: ignore
    except ImportError:
        import pickle

    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def _resolve_decoder(obj):
    """Best-effort extraction of a 2D decoder from a pipeline or decoder pickle."""
    # Direct decoder-like
    if hasattr(obj, 'p_x_given_n') and hasattr(obj, 'pf'):
        return obj

    # Common pipeline attribute paths
    candidates = []
    if hasattr(obj, 'computation_results'):
        try:
            for _ctx, result in dict(obj.computation_results).items():
                computed = getattr(result, 'computed_data', None)
                if computed is None:
                    continue
                for key in ('pf2D_Decoder', 'pf2D_decoder', 'decoder', 'pf2D'):
                    if hasattr(computed, key):
                        candidates.append(getattr(computed, key))
                    elif isinstance(computed, dict) and key in computed:
                        candidates.append(computed[key])
                ## END for key in (...)...
            ## END for _ctx, result in dict(obj.computation_results).items()...
        except Exception:
            pass

    for cand in candidates:
        if hasattr(cand, 'p_x_given_n') and hasattr(cand, 'pf'):
            return cand
    ## END for cand in candidates...

    raise ValueError('Could not locate a 2D decoder with .pf and .p_x_given_n in the pickle')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export 2D posterior-guessing NPZ bundle')
    parser.add_argument('--synthetic', action='store_true', help='Write the synthetic demo bundle')
    parser.add_argument('--from-pkl', type=Path, default=None, help='Path to decoder or pipeline pickle')
    parser.add_argument('--out', type=Path, default=None, help='Output NPZ path')
    parser.add_argument('--session-id', type=str, default=None, help='Session id stored in metadata')
    parser.add_argument('--no-filter-empty', action='store_true', help='Keep time bins with zero spikes')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for --synthetic')
    parser.add_argument('--n-time', type=int, default=40, help='Number of synthetic time bins')
    args = parser.parse_args(argv)

    if not args.synthetic and args.from_pkl is None:
        parser.error('Specify --synthetic and/or --from-pkl')

    filter_empty_bins = not args.no_filter_empty

    if args.synthetic:
        from pho.posterior_guessing.synthetic import make_synthetic_bundle

        out = args.out or (REPO_ROOT / 'data' / 'posterior_guessing' / 'synthetic_demo_pf2d_guessing.npz')
        path = make_synthetic_bundle(out_path=out, filter_empty_bins=filter_empty_bins, seed=args.seed, n_time=args.n_time)
        print(f'Wrote synthetic bundle: {path}')
        return 0

    from pho.posterior_guessing.export import export_from_decoder

    pkl_path = Path(args.from_pkl).expanduser().resolve()
    obj = _load_object_from_pickle(pkl_path)
    decoder = _resolve_decoder(obj)
    out = args.out or (REPO_ROOT / 'data' / 'posterior_guessing' / f'{(args.session_id or pkl_path.stem)}_pf2d_guessing.npz')
    path = export_from_decoder(
        decoder,
        out,
        metadata={'source_pkl': str(pkl_path)},
        filter_empty_bins=filter_empty_bins,
        session_id=args.session_id or pkl_path.stem,
    )
    print(f'Wrote bundle from pickle: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
