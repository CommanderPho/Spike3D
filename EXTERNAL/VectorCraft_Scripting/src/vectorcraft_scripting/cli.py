"""Locate ``vectorcraft-cli`` and run subprocesses without a shell."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

from .errors import CliNotFoundError, VectorCraftError


ENV_CLI = "VECTORCRAFT_CLI"

_WINDOWS_DEFAULTS = (
    Path(r"C:\Program Files\VectorCraft\vectorcraft-cli.exe"),
    Path(r"C:\Program Files (x86)\VectorCraft\vectorcraft-cli.exe"),
)


def resolve_cli(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the ``vectorcraft-cli`` binary path.

    Search order:
      1. ``explicit`` argument
      2. ``VECTORCRAFT_CLI`` environment variable
      3. ``PATH`` (``vectorcraft-cli`` / ``vectorcraft-cli.exe``)
      4. Windows default install locations under Program Files
    """
    searched: List[str] = []

    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_value = os.environ.get(ENV_CLI)
    if env_value:
        candidates.append(Path(env_value))
    for name in ("vectorcraft-cli", "vectorcraft-cli.exe"):
        which = shutil.which(name)
        if which:
            candidates.append(Path(which))
    ## END for name in ("vectorcraft-cli", "vectorcraft-cli.exe")...

    if sys.platform == "win32":
        candidates.extend(_WINDOWS_DEFAULTS)

    for candidate in candidates:
        searched.append(str(candidate))
        try:
            resolved = candidate.expanduser().resolve(strict=False)
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    ## END for candidate in candidates...

    raise CliNotFoundError(searched)


def abs_path(path: Union[str, Path]) -> str:
    """Return an absolute filesystem path string suitable for CLI args / JSON params."""
    return str(Path(path).expanduser().resolve())


def run_cli(args: Sequence[str], *, cli: Optional[Union[str, Path]] = None, check: bool = True, timeout: Optional[float] = None) -> subprocess.CompletedProcess[str]:
    """Run ``vectorcraft-cli`` with ``args`` (no shell). Raise ``VectorCraftError`` on failure when ``check``."""
    exe = resolve_cli(cli)
    cmd = [str(exe), *args]
    # UTF-8: the command catalogue and docs contain non-cp1252 glyphs; Windows defaults break decoding.
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise VectorCraftError(
            f"vectorcraft-cli failed (exit {completed.returncode}): {detail or 'no output'}",
            returncode=completed.returncode,
            stderr=completed.stderr or "",
            stdout=completed.stdout or "",
        )
    return completed


def parse_json_lines(stdout: str) -> List[Any]:
    """Parse newline-delimited JSON values from CLI stdout (blank lines skipped)."""
    results: List[Any] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        results.append(json.loads(stripped))
    ## END for line in stdout.splitlines()...

    return results


def parse_json_document(stdout: str) -> Any:
    """Parse a single JSON document from CLI stdout (``commands``, ``info``, …)."""
    text = stdout.strip()
    if not text:
        raise VectorCraftError("vectorcraft-cli produced empty stdout", stdout=stdout)
    return json.loads(text)


def dumps_params(params: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Serialize command params for ``--params``, or ``None`` when empty / omitted."""
    if not params:
        return None
    return json.dumps(params, separators=(",", ":"))
