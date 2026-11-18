"""Discover running VectorCraft apps that expose the live control protocol."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING

from .control import DEFAULT_CONTROL_PORT, DEFAULT_HOST, probe_control
from .errors import NoInstanceError

if TYPE_CHECKING:
    from .live import LiveVectorCraft

ENV_CONTROL_PORT = "VECTORCRAFT_CONTROL_PORT"
ENV_CONTROL_PORTS = "VECTORCRAFT_CONTROL_PORTS"


@dataclass
class LiveInstance:
    """A VectorCraft process answering the control protocol on loopback."""

    host: str
    port: int
    pid: Optional[int] = None
    ui_inspect: Dict[str, Any] = field(default_factory=dict)


    def connect(self, **kwargs) -> "LiveVectorCraft":
        from .live import LiveVectorCraft
        return LiveVectorCraft(host=self.host, port=self.port, **kwargs)


def _parse_port_list(value: str) -> List[int]:
    ports: List[int] = []
    for part in value.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        ports.append(int(token))
    ## END for part in value.replace(";", ",").split(",")...

    return ports


def _env_candidate_ports() -> List[int]:
    ports: List[int] = []
    single = os.environ.get(ENV_CONTROL_PORT)
    if single:
        ports.extend(_parse_port_list(single))
    multi = os.environ.get(ENV_CONTROL_PORTS)
    if multi:
        ports.extend(_parse_port_list(multi))
    return ports


def _windows_vectorcraft_listen_ports() -> List[tuple[int, Optional[int]]]:
    """Return ``(port, pid)`` for loopback listeners owned by ``vectorcraft`` processes."""
    if sys.platform != "win32":
        return []
    ps = r"""
$names = @('vectorcraft','VectorCraft','vectorcraft-cli')
$procs = Get-Process -ErrorAction SilentlyContinue | Where-Object { $names -contains $_.ProcessName }
if (-not $procs) { exit 0 }
$pids = $procs | Select-Object -ExpandProperty Id
Get-NetTCPConnection -State Listen -LocalAddress 127.0.0.1 -ErrorAction SilentlyContinue |
  Where-Object { $pids -contains $_.OwningProcess } |
  ForEach-Object { '{0}`t{1}' -f $_.LocalPort, $_.OwningProcess }
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found: List[tuple[int, Optional[int]]] = []
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t")
        try:
            port = int(parts[0])
            pid = int(parts[1]) if len(parts) > 1 and parts[1] else None
        except ValueError:
            continue
        found.append((port, pid))
    ## END for line in (completed.stdout or "").splitlines()...

    return found


def _dedupe_ports(ports: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    ordered: List[int] = []
    for port in ports:
        if port in seen:
            continue
        seen.add(port)
        ordered.append(int(port))
    ## END for port in ports...

    return ordered


def collect_candidate_ports(ports: Optional[Sequence[int]] = None, *, include_windows_process_scan: bool = True) -> List[int]:
    """Build the ordered list of ports to probe (without connecting)."""
    candidates: List[int] = []
    if ports is not None:
        candidates.extend(int(p) for p in ports)
    else:
        candidates.extend(_env_candidate_ports())
        candidates.append(DEFAULT_CONTROL_PORT)
        if include_windows_process_scan:
            candidates.extend(port for port, _pid in _windows_vectorcraft_listen_ports())
    return _dedupe_ports(candidates)


def discover_instances(ports: Optional[Sequence[int]] = None, *, host: str = DEFAULT_HOST, connect_timeout: float = 0.35, include_windows_process_scan: bool = True) -> List[LiveInstance]:
    """Probe candidate ports; return those that answer ``ui.inspect``."""
    pid_by_port: Dict[int, Optional[int]] = {}
    if include_windows_process_scan and ports is None:
        for port, pid in _windows_vectorcraft_listen_ports():
            pid_by_port[port] = pid
        ## END for port, pid in _windows_vectorcraft_listen_ports()...

    instances: List[LiveInstance] = []
    for port in collect_candidate_ports(ports, include_windows_process_scan=include_windows_process_scan):
        snapshot = probe_control(host=host, port=port, connect_timeout=connect_timeout)
        if snapshot is None:
            continue
        ui = snapshot if isinstance(snapshot, dict) else {"value": snapshot}
        instances.append(LiveInstance(host=host, port=port, pid=pid_by_port.get(port), ui_inspect=ui))
    ## END for port in collect_candidate_ports(ports, include_windows_process_scan=include_windows_process_scan)...

    return instances


def connect_first(ports: Optional[Sequence[int]] = None, *, host: str = DEFAULT_HOST, connect_timeout: float = 0.35, include_windows_process_scan: bool = True, **live_kwargs) -> "LiveVectorCraft":
    """Connect to the first healthy control instance, or raise ``NoInstanceError``."""
    searched = collect_candidate_ports(ports, include_windows_process_scan=include_windows_process_scan)
    instances = discover_instances(ports=searched, host=host, connect_timeout=connect_timeout, include_windows_process_scan=False)
    if not instances:
        raise NoInstanceError(searched)
    return instances[0].connect(**live_kwargs)
