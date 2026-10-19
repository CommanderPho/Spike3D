"""Errors raised by the VectorCraft Python bridge."""

from __future__ import annotations

from typing import List, Optional, Sequence


class VectorCraftError(RuntimeError):
    """Base error for VectorCraft CLI failures."""

    def __init__(self, message: str, *, returncode: Optional[int] = None, stderr: str = "", stdout: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class CliNotFoundError(VectorCraftError):
    """Raised when ``vectorcraft-cli`` cannot be located on this machine."""

    def __init__(self, searched: Sequence[str]):
        searched_list: List[str] = [str(p) for p in searched]
        paths = "\n  - ".join(searched_list) if searched_list else "(none)"
        message = (
            "vectorcraft-cli not found. Set VECTORCRAFT_CLI, add it to PATH, "
            f"or install VectorCraft. Searched:\n  - {paths}"
        )
        super().__init__(message)
        self.searched = searched_list


class ControlError(VectorCraftError):
    """Raised when a live control-protocol request fails or the connection breaks."""

    def __init__(self, message: str, *, method: Optional[str] = None, reply: Optional[dict] = None):
        super().__init__(message)
        self.method = method
        self.reply = reply


class NoInstanceError(VectorCraftError):
    """Raised when no VectorCraft control server answers on the searched ports."""

    def __init__(self, searched_ports: Sequence[int]):
        ports = ", ".join(str(p) for p in searched_ports) if searched_ports else "(none)"
        message = (
            "No VectorCraft control instance found. Start the app with "
            f"`vectorcraft --control PORT` (default 7979). Searched ports: {ports}"
        )
        super().__init__(message)
        self.searched_ports = list(searched_ports)
