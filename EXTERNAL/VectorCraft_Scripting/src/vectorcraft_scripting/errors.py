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
