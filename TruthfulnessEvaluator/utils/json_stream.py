"""Helpers for streaming JSON array writes while keeping files valid at all times."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, BinaryIO


class StreamingJsonArrayWriter:
    """
    Incrementally append JSON records to an on-disk array while keeping the file valid.

    The writer always ensures the file contains a syntactically valid JSON array by
    temporarily sealing it with a closing `]` after every append. Before writing the next
    record it seeks back, inserts the comma/record, and seals it again. This allows tailing
    the file in real time without waiting for the run to finish.
    """

    _ARRAY_END = b"\n]\n"

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: BinaryIO | None = path.open("wb")
        self._is_empty = True
        self._initialize_file()

    def _initialize_file(self) -> None:
        if self._file is None:
            raise RuntimeError("StreamingJsonArrayWriter file handle is not available.")
        self._file.write(b"[\n]\n")
        self._file.flush()

    def append(self, obj: Any) -> None:
        if self._file is None:
            raise RuntimeError("StreamingJsonArrayWriter has been closed.")
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._file.seek(-len(self._ARRAY_END), os.SEEK_END)
        if not self._is_empty:
            self._file.write(b",\n")
        self._file.write(payload)
        self._file.write(self._ARRAY_END)
        self._file.flush()
        self._is_empty = False

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "StreamingJsonArrayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

