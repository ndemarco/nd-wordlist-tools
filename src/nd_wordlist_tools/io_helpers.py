from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import gzip
import bz2
import lzma
from typing import Iterable, Iterator, TextIO, Optional

# Notes:
# - Supports "-", .gz, .bz2, .xz automatically.
# - Uses UTF-8 with 'replace' to avoid crashing on bad bytes in the wild.
# - Buffered streaming, no materialization unless requested.
# - Optional de-dupe via a set or a Bloom filter (future).

def _open_read(path: str | Path) -> TextIO:
    if str(path) == "-":
        return sys.stdin
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode="rt", encoding="utf-8", errors="replace")
    if p.suffix in {".bz2", ".bzip2"}:
        return bz2.open(p, mode="rt", encoding="utf-8", errors="replace")
    if p.suffix in {".xz", ".lzma"}:
        return lzma.open(p, mode="rt", encoding="utf-8", errors="replace")
    return p.open("rt", encoding="utf-8", errors="replace", buffering=1024 * 1024)

def _open_write(path: str | Path, atomic: bool = True) -> TextIO:
    if str(path) == "-":
        return sys.stdout
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode="wt", encoding="utf-8", errors="strict", compresslevel=6)
    if p.suffix in {".bz2", ".bzip2"}:
        return bz2.open(p, mode="wt", encoding="utf-8", errors="strict", compresslevel=9)
    if p.suffix in {".xz", ".lzma"}:
        return lzma.open(p, mode="wt", encoding="utf-8", errors="strict", preset=6)
    if atomic:
        # simple atomic strategy: write to tmp and rename
        tmp = p.with_suffix(p.suffix + ".tmp")
        return _AtomicWriter(tmp, final=p)
    return p.open("wt", encoding="utf-8", errors="strict", buffering=1024 * 1024)

class _AtomicWriter:
    def __init__(self, tmp: Path, final: Path):
        self.tmp = tmp
        self.final = final
        self._fh = tmp.open("wt", encoding="utf-8", errors="strict", buffering=1024 * 1024)

    def write(self, data: str) -> int:
        return self._fh.write(data)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
            self.tmp.replace(self.final)
        finally:
            if self.tmp.exists():
                self.tmp.unlink(missing_ok=True)

    def __getattr__(self, name):
        return getattr(self._fh, name)

@contextmanager
def open_text(path: str | Path, mode: str) -> Iterator[TextIO]:
    if "r" in mode:
        fh = _open_read(path)
        try:
            yield fh
        finally:
            if fh is not sys.stdin:
                fh.close()
    elif "w" in mode:
        fh = _open_write(path, atomic=True)
        try:
            yield fh
        finally:
            if fh is not sys.stdout:
                fh.close()
    else:
        raise ValueError("mode must include 'r' or 'w'")

def iter_lines(paths: Iterable[str | Path]) -> Iterator[str]:
    """Yield normalized lines (stripped, skip empty)."""
    for path in paths:
        with open_text(path, "r") as fh:
            for line in fh:
                s = line.rstrip("\r\n")
                if s:
                    yield s

def write_lines(path: str | Path, lines: Iterable[str]) -> int:
    count = 0
    with open_text(path, "w") as fh:
        for s in lines:
            fh.write(s)
            fh.write("\n")
            count += 1
    return count

def dedupe(lines: Iterable[str]) -> Iterator[str]:
    """In-memory de-duplication. Replace later with a disk-backed or Bloom filter if needed."""
    seen: set[str] = set()
    for s in lines:
        if s not in seen:
            seen.add(s)
            yield s
