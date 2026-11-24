#!/usr/bin/env python3
"""Discover pipeline pickles under given folders and report progress metadata.

Filesystem-only mode (``--meta-only``) never unpickles. Progress mode loads each
pickle read-only, one at a time, extracts stage / computation times / computed
keys, then discards the object before the next file. Never writes pickles.

Usage::

    # Paths + size + mtime only (no unpickle)
    python scripts/inventory_pipeline_pickle_progress.py /path/to/Data --meta-only

    # Full progress (local + paired global when present)
    python scripts/inventory_pipeline_pickle_progress.py W:/Data/KDIBA/gor01

    # Local loadedSessPickle only (skip global_computation_results)
    python scripts/inventory_pipeline_pickle_progress.py W:/Data --local-only --csv progress.csv

Progress mode (default) requires the Spike3D environment (``loadData`` / ``NeuropyPipeline``).
``--meta-only`` is stdlib-only and needs no project imports.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_LOCAL_GLOB: str = "*loadedSessPickle*.pkl"
DEFAULT_GLOBAL_GLOB: str = "output/*global_computation_results*.pkl"

LOCAL_STEM_PREFIX: str = "loadedSessPickle"
GLOBAL_STEM_PREFIX: str = "global_computation_results"

ROW_FIELDNAMES: List[str] = [
    "kind",
    "session_basedir",
    "path",
    "paired_global_path",
    "size_bytes",
    "mtime",
    "last_completed_stage",
    "any_most_recent_computation_time",
    "local_epoch_names",
    "local_computed_fn_names",
    "local_computed_data_keys",
    "global_computed_fn_names",
    "global_computed_data_keys",
    "incomplete_epochs",
    "failed_computations_summary",
    "error",
]


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def _size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


def _filesystem_row(kind: str, path: Path, *, session_basedir: Optional[Path] = None, paired_global_path: Optional[Path] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in ROW_FIELDNAMES}
    row["kind"] = kind
    row["path"] = str(path.resolve())
    row["session_basedir"] = str((session_basedir or path.parent).resolve())
    row["size_bytes"] = _size_bytes(path)
    row["mtime"] = _mtime_iso(path)
    if paired_global_path is not None:
        row["paired_global_path"] = str(paired_global_path.resolve())
    return row


def local_pickle_suffix(path: Path) -> str:
    """Suffix after ``loadedSessPickle`` in the stem (may be empty)."""
    stem: str = path.stem
    if stem.startswith(LOCAL_STEM_PREFIX):
        return stem[len(LOCAL_STEM_PREFIX):]
    return ""


def global_pickle_suffix(path: Path) -> str:
    """Suffix after ``global_computation_results`` in the stem (may be empty)."""
    stem: str = path.stem
    if stem.startswith(GLOBAL_STEM_PREFIX):
        return stem[len(GLOBAL_STEM_PREFIX):]
    return ""


def session_basedir_for_local(path: Path) -> Path:
    return path.parent.resolve()


def session_basedir_for_global(path: Path) -> Path:
    # .../session/output/global_....pkl -> session
    if path.parent.name == "output":
        return path.parent.parent.resolve()
    return path.parent.resolve()


def discover_data_files(basedir: Path, glob_pattern: str, recursive: bool = True) -> List[Path]:
    """Stdlib equivalent of ``pyphocorehelpers.Filesystem.path_helpers.discover_data_files``."""
    basedir = Path(basedir)
    pattern: str = f"**/{glob_pattern}" if recursive else glob_pattern
    return sorted(basedir.glob(pattern))


def discover_pickles(folders: Sequence[Path], *, local_glob: str, global_glob: str, recursive: bool, include_global: bool) -> Tuple[List[Path], List[Path]]:
    local_paths: List[Path] = []
    global_paths: List[Path] = []
    for folder in folders:
        folder = folder.resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        local_paths.extend(discover_data_files(folder, glob_pattern=local_glob, recursive=recursive))
        if include_global:
            global_paths.extend(discover_data_files(folder, glob_pattern=global_glob, recursive=recursive))
        ## END if include_global...
    ## END for folder in folders...

    # Deduplicate while preserving order
    def _dedupe(paths: List[Path]) -> List[Path]:
        seen: Set[Path] = set()
        out: List[Path] = []
        for p in paths:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)
        ## END for p in paths...
        return out

    return _dedupe(local_paths), _dedupe(global_paths)


def pair_local_to_global(local_paths: Sequence[Path], global_paths: Sequence[Path]) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Path]]:
    """Return ([(local, paired_global_or_None), ...], unpaired_globals)."""
    globals_by_session: Dict[Path, List[Path]] = {}
    for g in global_paths:
        session = session_basedir_for_global(g)
        globals_by_session.setdefault(session, []).append(g)
    ## END for g in global_paths...

    used_globals: Set[Path] = set()
    pairs: List[Tuple[Path, Optional[Path]]] = []

    for local in local_paths:
        session = session_basedir_for_local(local)
        local_suffix: str = local_pickle_suffix(local)
        candidates: List[Path] = list(globals_by_session.get(session, []))
        exact: Optional[Path] = None
        prefix_matches: List[Path] = []
        for g in candidates:
            g_suffix: str = global_pickle_suffix(g)
            if g_suffix == local_suffix:
                exact = g
                break
            if g_suffix.startswith(local_suffix):
                prefix_matches.append(g)
            ## END if g_suffix.startswith(local_suffix)...
        ## END for g in candidates...

        chosen: Optional[Path] = exact
        if chosen is None and prefix_matches:
            # Prefer newest mtime among suffix-prefix matches (e.g. -tbin_ variants)
            chosen = max(prefix_matches, key=lambda p: p.stat().st_mtime)
        ## END if chosen is None and prefix_matches...

        if chosen is not None:
            used_globals.add(chosen.resolve())
        pairs.append((local, chosen))
    ## END for local in local_paths...

    unpaired: List[Path] = [g for g in global_paths if g.resolve() not in used_globals]
    return pairs, unpaired


def _keys_of(container: Any) -> List[str]:
    if container is None:
        return []
    try:
        return [str(k) for k in list(container.keys())]
    except Exception:
        return []


def _join_names(names: Iterable[str]) -> str:
    return ";".join(sorted(set(str(n) for n in names)))


def _datetime_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value == datetime.min:
            return ""
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def unwrap_pipeline(loaded: Any) -> Any:
    """Return a NeuropyPipeline from loadData output (object or dict wrapper)."""
    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline

    if isinstance(loaded, NeuropyPipeline):
        return loaded
    if isinstance(loaded, dict):
        for key in ("curr_active_pipeline", "pipeline", "db"):
            cand = loaded.get(key)
            if isinstance(cand, NeuropyPipeline):
                return cand
            ## END if isinstance(cand, NeuropyPipeline)...
        ## END for key in (...)...
        # Sometimes the dict itself is a ComputationResult-like mapping for globals
        return loaded
    return loaded


def extract_from_computation_result_like(obj: Any) -> Tuple[List[str], List[str]]:
    """Return (computed_data_keys, computation_times_fn_names) from a ComputationResult or dict."""
    if obj is None:
        return [], []
    computed_data = None
    computation_times = None
    if isinstance(obj, dict):
        computed_data = obj.get("computed_data")
        computation_times = obj.get("computation_times")
    else:
        computed_data = getattr(obj, "computed_data", None)
        computation_times = getattr(obj, "computation_times", None)
    return _keys_of(computed_data), _keys_of(computation_times)


def extract_pipeline_progress(pipeline: Any) -> Dict[str, Any]:
    """Extract progress fields from a loaded NeuropyPipeline."""
    out: Dict[str, Any] = {
        "last_completed_stage": "",
        "any_most_recent_computation_time": "",
        "local_epoch_names": "",
        "local_computed_fn_names": "",
        "local_computed_data_keys": "",
        "global_computed_fn_names": "",
        "global_computed_data_keys": "",
        "incomplete_epochs": "",
        "failed_computations_summary": "",
    }

    stage = getattr(pipeline, "last_completed_stage", None)
    if stage is not None:
        out["last_completed_stage"] = getattr(stage, "name", str(stage))

    computation_results = getattr(pipeline, "computation_results", None) or {}
    out["local_epoch_names"] = _join_names(_keys_of(computation_results))

    local_fn_names: List[str] = []
    local_data_keys: List[str] = []
    for _epoch_name, result in (computation_results.items() if hasattr(computation_results, "items") else []):
        if result is None:
            continue
        data_keys, fn_names = extract_from_computation_result_like(result)
        local_data_keys.extend(data_keys)
        local_fn_names.extend(fn_names)
    ## END for _epoch_name, result in ...
    out["local_computed_fn_names"] = _join_names(local_fn_names)
    out["local_computed_data_keys"] = _join_names(local_data_keys)

    global_results = getattr(pipeline, "global_computation_results", None)
    g_data_keys, g_fn_names = extract_from_computation_result_like(global_results)
    out["global_computed_data_keys"] = _join_names(g_data_keys)
    out["global_computed_fn_names"] = _join_names(g_fn_names)

    get_computation_times = getattr(pipeline, "get_computation_times", None)
    if callable(get_computation_times):
        try:
            any_most_recent, _each_epoch_latest, _each_epoch_each, (_g_latest, _g_times) = get_computation_times(debug_print=False)
            out["any_most_recent_computation_time"] = _datetime_str(any_most_recent)
        except Exception as e:
            out["any_most_recent_computation_time"] = f"<error: {type(e).__name__}: {e}>"
        ## END try get_computation_times...
    ## END if callable(get_computation_times)...

    stage_obj = getattr(pipeline, "stage", None)
    progress_fn = getattr(stage_obj, "_get_computation_results_progress", None) if stage_obj is not None else None
    if callable(progress_fn):
        try:
            _complete, incomplete = progress_fn(debug_print=False)
            if incomplete:
                out["incomplete_epochs"] = _join_names([f"{k}:{v}" for k, v in incomplete.items()])
            ## END if incomplete...
        except Exception:
            pass
        ## END try progress_fn...
    ## END if callable(progress_fn)...

    get_failed = getattr(pipeline, "get_failed_computations", None)
    if callable(get_failed):
        try:
            failed = get_failed()
            if failed:
                bits: List[str] = []
                for epoch_name, fn_map in failed.items():
                    if not fn_map:
                        continue
                    fn_part = ",".join(str(k) for k in fn_map.keys())
                    bits.append(f"{epoch_name}:[{fn_part}]")
                ## END for epoch_name, fn_map in failed.items()...
                out["failed_computations_summary"] = ";".join(bits)
            ## END if failed...
        except Exception:
            pass
        ## END try get_failed...
    ## END if callable(get_failed)...

    return out


def extract_global_dict_progress(loaded: Any) -> Dict[str, Any]:
    """Extract progress from a standalone global_computation_results pickle (dict or ComputationResult)."""
    out: Dict[str, Any] = {
        "last_completed_stage": "",
        "any_most_recent_computation_time": "",
        "local_epoch_names": "",
        "local_computed_fn_names": "",
        "local_computed_data_keys": "",
        "global_computed_fn_names": "",
        "global_computed_data_keys": "",
        "incomplete_epochs": "",
        "failed_computations_summary": "",
    }
    data_keys, fn_names = extract_from_computation_result_like(loaded)
    out["global_computed_data_keys"] = _join_names(data_keys)
    out["global_computed_fn_names"] = _join_names(fn_names)
    # Prefer max computation_times value when present
    times = None
    if isinstance(loaded, dict):
        times = loaded.get("computation_times")
    else:
        times = getattr(loaded, "computation_times", None)
    if times:
        try:
            values = list(times.values())
            dt_values = [v for v in values if isinstance(v, datetime)]
            if dt_values:
                out["any_most_recent_computation_time"] = _datetime_str(max(dt_values))
            ## END if dt_values...
        except Exception:
            pass
        ## END try times...
    ## END if times...
    return out


def inventory_meta_only(local_paths: Sequence[Path], global_paths: Sequence[Path], *, include_global: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pairs, unpaired = pair_local_to_global(local_paths, global_paths if include_global else [])
    for local, paired_global in pairs:
        rows.append(_filesystem_row("local", local, session_basedir=session_basedir_for_local(local), paired_global_path=paired_global))
    ## END for local, paired_global in pairs...
    if include_global:
        for g in unpaired:
            rows.append(_filesystem_row("global", g, session_basedir=session_basedir_for_global(g)))
        ## END for g in unpaired...
    ## END if include_global...
    return rows


def inventory_with_load(local_paths: Sequence[Path], global_paths: Sequence[Path], *, include_global: bool) -> List[Dict[str, Any]]:
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData
    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline

    rows: List[Dict[str, Any]] = []
    pairs, unpaired = pair_local_to_global(local_paths, global_paths if include_global else [])

    for local, paired_global in pairs:
        row = _filesystem_row("local", local, session_basedir=session_basedir_for_local(local), paired_global_path=paired_global)
        pipeline = None
        loaded = None
        try:
            print(f"[LOAD local] {local}")
            loaded = loadData(local, debug_print=False)
            pipeline = unwrap_pipeline(loaded)
            if not isinstance(pipeline, NeuropyPipeline):
                raise TypeError(f"Expected NeuropyPipeline from {local}, got {type(pipeline)!r}")
            ## END if not isinstance(pipeline, NeuropyPipeline)...

            if include_global and paired_global is not None and paired_global.exists():
                print(f"[LOAD global into pipeline] {paired_global}")
                try:
                    pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=paired_global, allow_overwrite_existing=True, debug_print=False)
                except Exception as ge:
                    # Still report local progress; note global merge failure
                    row["error"] = f"global_merge:{type(ge).__name__}: {ge}"
                ## END try load global into pipeline...
            ## END if include_global and paired_global...

            progress = extract_pipeline_progress(pipeline)
            row.update(progress)
        except Exception as e:
            existing_err = row.get("error") or ""
            msg = f"{type(e).__name__}: {e}"
            row["error"] = f"{existing_err}; {msg}".strip("; ") if existing_err else msg
            print(f"[ERROR local] {local}: {row['error']}")
        finally:
            del pipeline
            del loaded
            gc.collect()
        ## END try/finally local...
        rows.append(row)
    ## END for local, paired_global in pairs...

    if include_global:
        for g in unpaired:
            row = _filesystem_row("global", g, session_basedir=session_basedir_for_global(g))
            loaded = None
            try:
                print(f"[LOAD unpaired global] {g}")
                loaded = loadData(g, debug_print=False)
                progress = extract_global_dict_progress(loaded)
                row.update(progress)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[ERROR global] {g}: {row['error']}")
            finally:
                del loaded
                gc.collect()
            ## END try/finally unpaired global...
            rows.append(row)
        ## END for g in unpaired...
    ## END if include_global...

    return rows


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("(no pickles found)")
        return
    ## END if not rows...

    display_cols: List[str] = [
        "kind",
        "session_basedir",
        "path",
        "size_bytes",
        "mtime",
        "last_completed_stage",
        "any_most_recent_computation_time",
        "local_computed_fn_names",
        "global_computed_data_keys",
        "error",
    ]
    # Compact path display: basename for path column in printout
    for i, row in enumerate(rows, start=1):
        print("-" * 80)
        print(f"[{i}/{len(rows)}] kind={row.get('kind')} stage={row.get('last_completed_stage')}")
        print(f"  session: {row.get('session_basedir')}")
        print(f"  path:    {row.get('path')}")
        if row.get("paired_global_path"):
            print(f"  global:  {row.get('paired_global_path')}")
        print(f"  size={row.get('size_bytes')} mtime={row.get('mtime')}")
        print(f"  any_most_recent_computation_time={row.get('any_most_recent_computation_time')}")
        print(f"  local_epochs={row.get('local_epoch_names')}")
        print(f"  local_fn={row.get('local_computed_fn_names')}")
        print(f"  local_data_keys={row.get('local_computed_data_keys')}")
        print(f"  global_fn={row.get('global_computed_fn_names')}")
        print(f"  global_data_keys={row.get('global_computed_data_keys')}")
        if row.get("incomplete_epochs"):
            print(f"  incomplete={row.get('incomplete_epochs')}")
        if row.get("failed_computations_summary"):
            print(f"  failed={row.get('failed_computations_summary')}")
        if row.get("error"):
            print(f"  ERROR={row.get('error')}")
    ## END for i, row in enumerate(rows)...
    print("-" * 80)
    print(f"Total rows: {len(rows)}")
    _ = display_cols  # kept for potential future columnar layout


def write_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ROW_FIELDNAMES})
        ## END for row in rows...
    ## END with csv_path.open...
    print(f"Wrote CSV: {csv_path} ({len(rows)} rows)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory pipeline pickle progress under given folders (read-only).")
    parser.add_argument("folders", nargs="+", type=Path, help="One or more root folders to search")
    parser.add_argument("--meta-only", action="store_true", help="Filesystem metadata only; do not unpickle")
    parser.add_argument("--local-only", action="store_true", help="Skip output/global_computation_results*.pkl")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--glob-local", type=str, default=DEFAULT_LOCAL_GLOB, help=f"Local pickle glob (default: {DEFAULT_LOCAL_GLOB})")
    parser.add_argument("--glob-global", type=str, default=DEFAULT_GLOBAL_GLOB, help=f"Global pickle glob (default: {DEFAULT_GLOBAL_GLOB})")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Recursive discovery (default)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive discovery")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    include_global: bool = not bool(args.local_only)

    try:
        local_paths, global_paths = discover_pickles(args.folders, local_glob=args.glob_local, global_glob=args.glob_global, recursive=bool(args.recursive), include_global=include_global)
    except NotADirectoryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    ## END try discover...

    print(f"Found {len(local_paths)} local pickle(s)" + (f", {len(global_paths)} global pickle(s)" if include_global else " (local-only)"))

    if args.meta_only:
        rows = inventory_meta_only(local_paths, global_paths, include_global=include_global)
    else:
        rows = inventory_with_load(local_paths, global_paths, include_global=include_global)
    ## END if args.meta_only...

    print_table(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
    ## END if args.csv...
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
