"""Shared file and stream utilities; no UI, model, or configuration loading."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


def configure_stdio():
    """Use UTF-8 for Windows redirected output and development tools."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not sys.stdin.isatty() and hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")


def atomic_write(path: Path, text: str):
    """Replace a file only after its complete UTF-8 contents have been written."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, contents: bytes):
    """Preserve exact bytes, including during configuration rollback."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".optimizer-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
