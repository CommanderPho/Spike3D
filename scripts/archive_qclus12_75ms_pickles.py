#!/usr/bin/env python3
"""One-off archive of current (non-tbin-suffixed) qclus12 pipeline pickles before a 25ms clean_run.

Moves (does not copy) canonical pickles aside with suffix ``2026-09-23_75ms`` so the
upcoming batch can rewrite the cleared canonical paths.

Sessions match ProcessBatchOutputs_qclus12_Only.ipy (active included_session_contexts).
Pickle parameter suffix matches override_custom_pickle_suffix:
  ``_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0``

Usage (on remote, e.g. GreatLakes)::

    # Preview (default)
    python archive_qclus12_75ms_pickles.py
    python archive_qclus12_75ms_pickles.py --dry-run

    # Apply moves
    python archive_qclus12_75ms_pickles.py --execute

    # Explicit data root
    python archive_qclus12_75ms_pickles.py --execute --data-root /nfs/turbo/umms-kdiba/Data
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Matches ProcessBatchOutputs_qclus12_Only.ipy active_phase_dict['override_custom_pickle_suffix']
PARAMETER_SPECIFIER: str = "_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0"
ARCHIVE_SUFFIX: str = "-tbin_75ms"
# ARCHIVE_SUFFIX: str = "2026-09-23-tbin_75ms"

# Relative session dirs under data root (KDIBA layout), matching included_session_contexts
SESSION_REL_PATHS: List[str] = [
    "KDIBA/gor01/one/2006-6-08_14-26-15",
    "KDIBA/gor01/one/2006-6-09_1-22-43",
    "KDIBA/gor01/one/2006-6-12_15-55-31",
    "KDIBA/gor01/two/2006-6-07_16-40-19",
    "KDIBA/gor01/two/2006-6-08_21-16-25",
    "KDIBA/gor01/two/2006-6-09_22-24-40",
    "KDIBA/gor01/two/2006-6-12_16-53-46",
    "KDIBA/vvp01/two/2006-4-09_16-40-54",
    "KDIBA/vvp01/two/2006-4-10_12-58-3",
    "KDIBA/pin01/one/fet11-01_12-58-54",
    "KDIBA/pin01/one/11-02_17-46-44",
    "KDIBA/pin01/one/11-03_12-3-25",
]

KNOWN_DATA_ROOT_CANDIDATES: List[str] = [
    "/nfs/turbo/umms-kdiba/Data",
    "/home/halechr/FastData",
    "/media/halechr/BETAMAX/Data",
    "/media/halechr/MAX/Data",
    "/Volumes/MoverNew/data",
    "W:/Data",
    "/Users/pho/data",
]


def find_data_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not root.exists():
            raise FileNotFoundError(f"--data-root does not exist: {root}")
        return root
    for candidate in KNOWN_DATA_ROOT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "No data root found. Pass --data-root explicitly. Tried:\n  "
        + "\n  ".join(KNOWN_DATA_ROOT_CANDIDATES)
    )


def archive_dest_path(src: Path, archive_suffix: str) -> Path:
    """Append archive_suffix to stem: foo.pkl -> foo_{archive_suffix}.pkl"""
    return src.with_stem(f"{src.stem}_{archive_suffix}")


def files_for_session(session_dir: Path, parameter_specifier: str) -> List[Path]:
    """Canonical (non-tbin) paths for this parameter specifier.

    Names match python_template.py.j2 / _get_custom_filenames_from_computation_metadata:
      loadedSessPickle{suffix}.pkl
      output/global_computation_results{suffix}.pkl
      pipeline{suffix}.h5  (custom h5; may be absent)
    """
    return [
        session_dir / f"loadedSessPickle{parameter_specifier}.pkl",
        session_dir / "output" / f"global_computation_results{parameter_specifier}.pkl",
        session_dir / "output" / f"pipeline{parameter_specifier}.h5",
    ]


def try_move(src: Path, dest: Path, *, execute: bool) -> str:
    """Return status: moved | would_move | skipped_missing | skipped_dest_exists | error:..."""
    if not src.exists():
        return "skipped_missing"
    if dest.exists():
        return "skipped_dest_exists"
    if not execute:
        return "would_move"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return "moved"
    except Exception as e:
        return f"error:{type(e).__name__}: {e}"


def process_sessions(
    data_root: Path,
    session_rel_paths: Sequence[str],
    *,
    parameter_specifier: str,
    archive_suffix: str,
    execute: bool,
) -> Tuple[List[Tuple[str, Path, Path, str]], dict]:
    results: List[Tuple[str, Path, Path, str]] = []
    counts = {
        "moved": 0,
        "would_move": 0,
        "skipped_missing": 0,
        "skipped_dest_exists": 0,
        "error": 0,
        "session_missing": 0,
    }

    for rel in session_rel_paths:
        session_dir = data_root / rel
        if not session_dir.exists():
            print(f"[SESSION MISSING] {session_dir}")
            counts["session_missing"] += 1
            continue
        ## END if not session_dir.exists()...

        for src in files_for_session(session_dir, parameter_specifier):
            dest = archive_dest_path(src, archive_suffix)
            status = try_move(src, dest, execute=execute)
            results.append((rel, src, dest, status))
            if status.startswith("error:"):
                counts["error"] += 1
            else:
                counts[status] = counts.get(status, 0) + 1
            print(f"[{status}] {src}  ->  {dest}")
        ## END for src in files_for_session(...)...
    ## END for rel in session_rel_paths...

    return results, counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move qclus12 canonical pipeline pickles aside with a 75ms archive suffix."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print planned moves only (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually shutil.move files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root (otherwise first extant known path).",
    )
    parser.add_argument(
        "--archive-suffix",
        type=str,
        default=ARCHIVE_SUFFIX,
        help=f"Stem suffix for archived files (default: {ARCHIVE_SUFFIX}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    execute: bool = bool(args.execute)
    data_root = find_data_root(args.data_root)

    print(f"data_root: {data_root}")
    print(f"parameter_specifier: {PARAMETER_SPECIFIER}")
    print(f"archive_suffix: {args.archive_suffix}")
    print(f"mode: {'EXECUTE (move)' if execute else 'DRY-RUN (no changes)'}")
    print(f"n_sessions: {len(SESSION_REL_PATHS)}")
    print("-" * 80)

    _results, counts = process_sessions(
        data_root,
        SESSION_REL_PATHS,
        parameter_specifier=PARAMETER_SPECIFIER,
        archive_suffix=args.archive_suffix,
        execute=execute,
    )

    print("-" * 80)
    print("SUMMARY:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if not execute:
        print("\nRe-run with --execute to apply moves.")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
