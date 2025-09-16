#!/usr/bin/env python3
"""
ndcombine — assembles phrase from words and fills.

INPUTS
------
pairs: lines of "w1 w2" (two whitespace-separated tokens per line)
fills: lines of fill strings (one per line, blank lines ignored)

BEHAVIOR
--------
For each line in pairs:
  parse -> (w1, w2)
  for each fill in fills:
    emit: w1 + fill + w2

EXAMPLES
--------
# Insert all fills between each pair
ndcombine --pairs pairs.txt --fills fills.txt > out.txt

# Fills from stdin
cat fills.txt | ndcombine -p pairs.txt -f - > out.txt

DESIGN
------
- Pair lines must contain exactly two tokens (whitespace-delimited).
- Blank lines in fills input are always skipped.
- Pairs are streamed; fills are loaded into memory (usually small).
"""

from __future__ import annotations

import sys
from typing import Iterator, List, Optional, TextIO
import typer

app = typer.Typer(
    add_completion=False,
    help="Insert fills between word pairs (w1 w2 → w1 + fill + w2).",
)


def _read_lines(f: TextIO) -> Iterator[str]:
    for raw in f:
        yield raw.rstrip("\n\r")


def _read_nonblank(f: TextIO) -> Iterator[str]:
    for s in _read_lines(f):
        if s != "":
            yield s


def _load_fills(f: TextIO) -> List[str]:
    return list(_read_nonblank(f))


@app.command()
def run(
    pairs: typer.FileText = typer.Option(
        ...,
        "--pairs",
        "-p",
        help="File of word pairs: 'w1 w2' per line (use '-' for stdin)",
    ),
    fills: typer.FileText = typer.Option(
        ..., "--fills", "-f", help="File of fills (one per line; use '-' for stdin)"
    ),
    out: Optional[typer.FileTextWrite] = typer.Option(
        None, "--out", "-o", help="Output file (default: stdout)"
    ),
) -> None:
    """
    Insert all fills between each pair line (w1 w2).
    """
    writer: TextIO = out if out is not None else sys.stdout

    fills_list = _load_fills(fills)
    if not fills_list:
        return

    total = 0

    for line in _read_lines(pairs):
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            typer.echo(
                f"error: invalid pair line (expected 2 tokens): {line}", err=True
            )
            raise typer.Exit(code=1)

        w1, w2 = parts

        for fill in fills_list:
            writer.write(f"{w1}{fill}{w2}\n")
            total += 1


def main() -> None:
    app()


if __name__ == "__main__":
    app()
