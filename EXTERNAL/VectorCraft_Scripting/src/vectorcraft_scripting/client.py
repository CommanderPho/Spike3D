"""High-level Python client for ``vectorcraft-cli``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cli import abs_path, dumps_params, parse_json_document, parse_json_lines, resolve_cli, run_cli
from .errors import VectorCraftError

CommandSpec = Union[str, Tuple[str, Mapping[str, Any]]]


class VectorCraftClient:
    """Thin wrapper around VectorCraft's headless CLI (``run``, ``commands``, ``convert``, ``info``)."""

    def __init__(self, cli: Optional[Union[str, Path]] = None, *, timeout: Optional[float] = None):
        self.cli_path = resolve_cli(cli)
        self.timeout = timeout


    def run(self, *, input: Optional[Union[str, Path]] = None, commands: Sequence[CommandSpec] = (), exports: Sequence[Union[str, Path]] = (), scale: float = 1.0) -> List[Any]:
        """Headless batch: open ``input`` (or a new document), run commands, export files.

        Each item in ``commands`` is either a command id string or ``(id, params_dict)``.
        Returns one parsed JSON result per CLI step (open / cmd / export).
        """
        args: List[str] = ["run"]
        if input is not None:
            args.extend(["--in", abs_path(input)])
        for command in commands:
            if isinstance(command, str):
                cmd_id, params = command, None
            else:
                cmd_id, params = command[0], command[1]
            args.extend(["--cmd", cmd_id])
            params_json = dumps_params(params)
            if params_json is not None:
                args.extend(["--params", params_json])
        ## END for command in commands...

        for export_path in exports:
            args.extend(["--export", abs_path(export_path)])
        ## END for export_path in exports...

        if scale != 1.0:
            args.extend(["--scale", str(scale)])
        completed = run_cli(args, cli=self.cli_path, timeout=self.timeout)
        return parse_json_lines(completed.stdout)


    def commands(self) -> List[Dict[str, Any]]:
        """Return the engine command catalogue as a list of dicts."""
        completed = run_cli(["commands"], cli=self.cli_path, timeout=self.timeout)
        data = parse_json_document(completed.stdout)
        if not isinstance(data, list):
            raise VectorCraftError(f"expected a JSON array from `commands`, got {type(data).__name__}", stdout=completed.stdout)
        return data


    def convert(self, src: Union[str, Path], dst: Union[str, Path], *, scale: float = 1.0, artboard: Optional[int] = None, range: Optional[str] = None, outline_text: bool = False, decimals: Optional[int] = None, replace_lossy: bool = False, editing_data: bool = False, ai_version: Optional[str] = None, hidden_objects: Optional[str] = None, transparent_images: Optional[str] = None) -> Any:
        """Open ``src`` and export ``dst`` (format from the destination extension)."""
        args: List[str] = ["convert", abs_path(src), abs_path(dst)]
        if scale != 1.0:
            args.extend(["--scale", str(scale)])
        if artboard is not None:
            args.extend(["--artboard", str(artboard)])
        if range is not None:
            args.extend(["--range", range])
        if outline_text:
            args.append("--outline-text")
        if decimals is not None:
            args.extend(["--decimals", str(decimals)])
        if replace_lossy:
            args.append("--replace-lossy")
        if editing_data:
            args.append("--editing-data")
        if ai_version is not None:
            args.extend(["--ai-version", ai_version])
        if hidden_objects is not None:
            args.extend(["--hidden-objects", hidden_objects])
        if transparent_images is not None:
            args.extend(["--transparent-images", transparent_images])
        completed = run_cli(args, cli=self.cli_path, timeout=self.timeout)
        return parse_json_document(completed.stdout)


    def info(self, path: Union[str, Path]) -> Any:
        """Return a JSON summary of ``path`` (warnings, artboards, object counts, fonts)."""
        completed = run_cli(["info", abs_path(path)], cli=self.cli_path, timeout=self.timeout)
        return parse_json_document(completed.stdout)
