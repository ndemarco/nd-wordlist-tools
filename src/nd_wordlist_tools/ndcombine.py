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
    mode: str = typer.Option(
        ..., "--mode", "-m", help="Comma-separated modes: s (start), m (middle), e (end). E.g. -m s,m,e"
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

    # Parse mode string, preserve order, suppress duplicates
    mode_order = []
    seen = set()
    for m in (x.strip() for x in mode.split(",")):
        if m and m not in seen:
            if m not in {"s", "m", "e"}:
                typer.echo(f"error: Invalid mode '{m}'. Use s, m, or e.", err=True)
                raise typer.Exit(code=1)
            mode_order.append(m)
            seen.add(m)
    if not mode_order:
        typer.echo("error: At least one mode must be specified (s, m, e).", err=True)
        raise typer.Exit(code=1)

    pairs_path = None
    if hasattr(pairs, 'name') and pairs.name != '<stdin>':
        pairs_path = pairs.name

    for idx, m in enumerate(mode_order):
        # For each mode, re-open pairs file if not stdin
        if pairs_path:
            fh = open(pairs_path, 'r', encoding='utf-8')
        else:
            if idx == 0:
                fh = pairs
            else:
                typer.echo("error: cannot re-read pairs from stdin for multiple modes; use a file instead.", err=True)
                raise typer.Exit(code=1)
        try:
            for line in _read_lines(fh):
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
                    if m == "s":
                        writer.write(f"{fill}{w1}{w2}\n")
                        total += 1
                    elif m == "m":
                        writer.write(f"{w1}{fill}{w2}\n")
                        total += 1
                    elif m == "e":
                        writer.write(f"{w1}{w2}{fill}\n")
                        total += 1
        finally:
            if pairs_path:
                fh.close()


def main() -> None:
    app()


if __name__ == "__main__":
    app()
