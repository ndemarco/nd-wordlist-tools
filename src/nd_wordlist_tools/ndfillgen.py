#!/usr/bin/env python3
"""
ndfillgen — mask-based digit/symbol generator from circular sequences.

Mask:
  Pairs of directives. 'dN' emits the digit at 1-based position N in the sequence.
  'sN' emits the shifted symbol for that digit using the chosen locale.

Sequence length:
  L = max(N) over all directives in the mask.

Input:
  Either a single --charset string, or lines from --in / stdin, each line a charset.
  Example charset: 1234567890

Output:
  For each input charset, emit len(charset) lines, one per rotation, after applying the mask.

Examples:
  ndfillgen run --mask "d1s2d3s1"
  echo 1234567890 | ndfillgen run --mask s1d2s3 --in - --out -
"""

import sys
import re
from typing import List, Sequence, Tuple, TextIO, Optional
import typer
from typer import FileText

app = typer.Typer(add_completion=False)

_LOCALES = {
    "en-US": [")", "!", "@", "#", "$", "%", "^", "&", "*", "("],
    "en-GB": [")", "!", '"', "£", "$", "%", "^", "&", "*", "("],
    "de-DE": ["!", '"', "§", "$", "%", "&", "/", "(", ")", "="],
    "es-ES": ["!", '"', "·", "$", "%", "&", "/", "(", ")", "="],
    "es-LA": ["!", '"', "#", "$", "%", "&", "/", "(", ")", "="],
    "fr-FR": [")", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "it-IT": ["!", '"', "£", "$", "%", "&", "/", "(", ")", "="],
    "sv-SE": ["!", '"', "#", "¤", "%", "&", "/", "(", ")", "="],
    "ru-RU": ["!", '"', "№", ";", "%", ":", "?", "*", "(", ")"],
    "ja-JP": [")", "!", '"', "#", "$", "%", "&", "'", "(", ")"],
}

_PAIR_RE = re.compile(r"([sd])(\d+)$|([sd])(\d+)", re.ASCII)


def parse_mask(mask: str) -> Tuple[Tuple[str, int], ...]:
    if not isinstance(mask, str):
        raise TypeError("mask must be a string")
    mask = mask.strip()
    if not mask:
        raise ValueError("mask is empty")
    if mask[0] not in ("s", "d"):
        raise ValueError("mask must start with 's' or 'd'")

    out: List[Tuple[str, int]] = []
    pos = 0
    for m in _PAIR_RE.finditer(mask):
        if m.start() != pos:
            bad_index = pos
            ch = mask[bad_index]
            if ch in ("s", "d"):
                raise ValueError(
                    f"expected integer after directive at index {bad_index}"
                )
            raise ValueError(f"unexpected character {ch!r} at index {bad_index}")
        directive = m.group(1) or m.group(3)
        digits = m.group(2) or m.group(4)
        out.append((directive, int(digits)))
        pos = m.end()
    if pos != len(mask):
        raise ValueError("trailing garbage in mask")
    return tuple(out)


def shifted_symbol(locale: str, digit: int) -> str:
    if locale not in _LOCALES:
        raise ValueError(f"unsupported locale: {locale}")
    if not (0 <= digit <= 9):
        raise ValueError(f"digit must be in 0..9, got {digit}")
    return _LOCALES[locale][digit]


def circular_sequences(charset: List[str], length: int) -> List[str]:
    n = len(charset)
    if n == 0:
        return []
    results: List[str] = []
    for i in range(n):
        seq = "".join(charset[(i + j) % n] for j in range(length))
        results.append(seq)
    return results


def apply_directives(
    template: Sequence[str], directives: Tuple[Tuple[str, int], ...], locale: str
) -> str:
    out: List[str] = []
    for directive, pos in directives:
        if pos <= 0:
            raise ValueError(f"positions are 1-based; got {pos}")
        if pos > len(template):
            raise ValueError(f"position {pos} exceeds template length {len(template)}")
        ch = template[pos - 1]
        if directive == "d":
            out.append(ch)
        elif directive == "s":
            if not ch.isdigit():
                raise ValueError(f"template[{pos - 1}]='{ch}' not a digit for 's'")
            out.append(shifted_symbol(locale, int(ch)))
        else:
            raise ValueError(f"unknown directive: {directive}")
    return "".join(out)


def _open_in(path: Optional[TextIO]) -> TextIO:
    return path if path is not None else sys.stdin


def _open_out(path: Optional[TextIO]) -> TextIO:
    return path if path is not None else sys.stdout


@app.command()
def run(
    mask: str = typer.Option(
        ..., "--mask", "-m", help="Mask like 'd1s2d3'. 1-based positions."
    ),
    charset: str = typer.Option(
        "1234567890",
        "--charset",
        "-c",
        help="Charset string to rotate (ignored if --in is provided).",
    ),
    infile: Optional[FileText] = typer.Option(
        None, "--in", help="Path to file with one charset per line. Use '-' for stdin."
    ),
    outfile: Optional[typer.FileTextWrite] = typer.Option(
        None, "--out", help="Path to write results. Use '-' for stdout."
    ),
    locale: str = typer.Option(
        "en-US", "--locale", help="Keyboard locale for 's' directives."
    ),
):
    """
    Read charsets and emit masked outputs from circular sequences of length L=max(mask positions).
    """
    directives = parse_mask(mask)
    L = max(pos for _, pos in directives)

    in_fh = _open_in(infile)
    out_fh = _open_out(outfile)

    # If --in provided, read multiple charsets. Else, use single provided --charset.
    if infile is None:
        line = charset.strip()
        if line:
            chars = list(line)
            for seq in circular_sequences(chars, L):
                out_fh.write(apply_directives(seq, directives, locale) + "\n")
    else:
        for raw in in_fh:
            line = raw.strip()
            if not line:
                continue
            chars = list(line)
            for seq in circular_sequences(chars, L):
                out_fh.write(apply_directives(seq, directives, locale) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
