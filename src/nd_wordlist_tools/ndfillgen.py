#!/usr/bin/env python3
"""
ndfillgen — deterministic generator for digit/symbol “fill” sequences.

PURPOSE
    Produce short “fill” strings (digits and/or their shifted-symbol counterparts)
    from a compact, human-readable DSL. The output is a pure stream of fills
    suitable for piping into ndcombine (which actually inserts them between/around
    words) or directly into a cracking engine.

KEY IDEAS
    • Determinism: identical inputs and DSL yield the same sequence order forever.
    • Compact notation: a single DSL line can describe a very large fill keyspace.
    • Splittable keyspace: supports --start-index/--count for chunked/distributed runs.

DIGIT → SHIFTED-SYMBOL MAP
    1→!   2→@   3→#   4→$   5→%   6→^   7→&   8→*   9→(   0→)

DSL OVERVIEW
    Form (per line):
        <SEED>  "->"  <ACTION-LIST>

    SEED (choose one):
        @<digits>            Literal digit seed (e.g., @56)
        #range(a..b)         Inclusive numeric range, zero-padded to max width
                             (e.g., #range(01..12) → 01,02,…,12)
        #mask(<mask>)        Mask per position using ?d or a digit set {…}
                             (e.g., #mask(?d?d) or #mask({12}{39}{07}))
    ACTION-LIST:
        Space-separated layouts; each layout expands SEED into one or more fills.
        Layout keywords:
          D        digits in original order
          D~       digits reversed
          S        shifted symbols for the digits (map above), in original order
          S~       shifted symbols reversed
          zip(ds)  interleave digit, symbol → d1,s1,d2,s2,…
          zip(sd)  interleave symbol, digit → s1,d1,s2,d2,…
          zip(d s~)   interleave digits with REVERSED symbols
          zip(s~ d)   interleave REVERSED symbols with digits

    EXAMPLES
        @56 -> D
            emits: 56

        @56 -> S
            emits: ^@

        @56 -> D S zip(ds) zip(sd)
            emits: 56
                   ^@
                   5^6@
                   ^5@6

        #range(10..12) -> D S
            emits: 10 11 12
                   )! !! @!

        #mask({12}{39}{07}) -> zip(ds)
            positions:
              p1 ∈ {1,2}, p2 ∈ {3,9}, p3 ∈ {0,7}
            emits all zipped (digit,symbol) interleavings for each 3‑digit choice.

COMMAND-LINE (typical)
    ndfillgen [OPTIONS]
      Reads DSL lines from stdin (or --in) and writes fills to stdout (or --out).
      Each non-empty, non-comment line must be a single "<SEED> -> <ACTION-LIST>".

OPTIONS (proposed/typical)
    --in PATH            Read DSL from file instead of stdin
    --out PATH           Write fills to file instead of stdout
    --start-index N      Skip first N fills in the global enumeration (for sharding)
    --count K            Emit at most K fills (for sharding)
    --dedupe             Suppress duplicate fills within the run
    --quiet              Suppress progress to stderr

DETERMINISTIC ORDERING
    For each DSL line:
      1) Expand SEED to a list of digit sequences in a fixed, documented order:
         - @<digits>: single sequence (verbatim)
         - #range(a..b): numeric ascending, left‑padded to max width
         - #mask(...): cartesian product in position order; within each position:
                       digits appear in ascending ASCII order (0..9) unless an
                       explicit set {…} is given; sets iterate in the user’s order
      2) For each sequence, expand ACTION-LIST left→right. For “zip(…)” forms,
         the first argument is emitted first in each pair.
      3) Concatenate per-line outputs in file order.

SPLITTING (RANK/UNRANK)
    The generator maintains a global, zero-based index over the emitted fills.
    With --start-index S and --count C, ndfillgen emits the half-open range
    [S, S+C). This enables distributed runs and precise logging of tried ranges.

EXIT CODES
    0 = success
    non-zero = parse or runtime error

NOTES
    • Lines beginning with "#" are comments and ignored.
    • Blank lines are ignored.
    • Inputs should be UTF-8; outputs are LF-terminated ASCII.

EXTRA EXAMPLES
    # two digits, both orders, plus their symbols
    @12 -> D D~ S S~
      → 12, 21, !@, @!

    # “shifted-only” fills for all 2-digit numbers 00..99
    #range(00..99) -> S

    # mix of seeds and layouts in one file
    @56 -> zip(ds) S
    #mask(?d?d) -> D
    #range(7..9) -> zip(sd)

IMPLEMENTATION HINTS (internal)
    • Keep the digit→symbol map table local and immutable.
    • Use generators to stream; avoid materializing large expansions.
    • Unit-test: seed expansion, each layout, and ordering; goldens for rank/unrank.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, List, Sequence, Tuple
import re
import sys
import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)

# -----------------------------
# Errors
# -----------------------------


class DSLParseError(ValueError):
    """Raised when the DSL string is syntactically invalid."""


class DSLSemanticError(ValueError):
    """Raised when a DSL string is syntactically valid but semantically invalid."""


# -----------------------------
# Symbol mapping
# -----------------------------

_DIGIT_TO_SHIFT = {
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
}


def shift_of_digit(d: str) -> str:
    """Return the shifted number-row symbol for a single digit.

    Args:
        d: A single character in '0'..'9'.

    Returns:
        The corresponding shifted symbol.

    Raises:
        ValueError: if *d* is not a digit.
    """
    try:
        return _DIGIT_TO_SHIFT[d]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Not a digit: {d!r}") from exc
    

# --- DSL help wiring -------------------------------------------------------
import typer

def _dsl_help_text() -> str:
    """Return a concise DSL syntax guide for --dsl-help.

    Kept in-sync with the module docstring.
    """
    return (
        "Form:\n"
        "  <seed> -> <action>\n\n"
        "SEED:\n"
        "  @<digits>               # literal (e.g., @12)\n"
        "  #range(a..b)            # inclusive, equal-width (e.g., 0000..9999)\n"
        "  #mask(<mask>)           # ?d or ?{digits} per position\n\n"
        "ACTION (layouts, space-separated):\n"
        "  D | D~                  # digits (order / reversed)\n"
        "  S | S~                  # symbols (order / reversed)\n"
        "  zip(ds|sd|d s~|s~ d)    # interleave by position\n\n"
        "SHIFT MAP:\n"
        "  1→! 2→@ 3→# 4→$ 5→% 6→^ 7→& 8→* 9→( 0→)\n\n"
        "EXAMPLES:\n"
        "  nd_num2symbols run \"@12 -> D S\"      # 12!@\n"
        "  nd_num2symbols run \"@12 -> S D\"      # !@12\n"
        "  nd_num2symbols run \"@12 -> D S~\"     # 12@!\n"
        "  nd_num2symbols run \"@12 -> zip(ds)\"  # 1!2@\n"
        "  nd_num2symbols run \"@12 -> zip(sd)\"  # !1@2\n"
    )

# -----------------------------
# Seed specifications
# -----------------------------


@dataclass(frozen=True)
class SeedSpec:
    """Specification for generating seed digit strings.

    Exactly one of the fields is populated.
    """

    literal: str | None = None
    range_start: str | None = None
    range_end: str | None = None
    mask_tokens: Tuple[str, ...] | None = None  # tokens like '?d' or '?{12}'

    def expand(self) -> Iterator[str]:
        """Yield each seed as a digit string.

        Yields:
            Strings composed solely of digits '0'..'9'.
        """
        if self.literal is not None:
            yield self.literal
            return

        if self.range_start is not None and self.range_end is not None:
            a, b = self.range_start, self.range_end
            if len(a) != len(b):
                raise DSLSemanticError("range endpoints must have equal width")
            ia = int(a)
            ib = int(b)
            if ia > ib:
                raise DSLSemanticError("range start must be <= end")
            width = len(a)
            for i in range(ia, ib + 1):
                yield f"{i:0{width}d}"
            return

        if self.mask_tokens is not None:
            yield from _expand_mask_tokens(self.mask_tokens)
            return

        raise DSLSemanticError("empty seed spec")  # pragma: no cover


# -----------------------------
# Mask expansion (subset)
# -----------------------------

_MASK_TOKEN_RE = re.compile(r"\?d|\?\{[0-9]+\}")


def parse_mask(mask: str) -> Tuple[str, ...]:
    """Parse a subset of maskprocessor mask into tokens.

    Supported tokens:
        - ``?d`` for digits 0..9
        - ``?{<digits>}`` for an explicit per-position digit set (e.g., ``?{12}``).

    Args:
        mask: The mask string inside ``#mask( … )``.

    Returns:
        A tuple of token strings.

    Raises:
        DSLParseError: if the mask contains unsupported tokens or stray text.
    """
    tokens: List[str] = []
    pos = 0
    while pos < len(mask):
        m = _MASK_TOKEN_RE.match(mask, pos)
        if not m:
            # Show a helpful snippet around the error
            context = mask[pos : min(len(mask), pos + 16)]
            raise DSLParseError(f"unsupported mask token at: {context!r}")
        tok = m.group(0)
        tokens.append(tok)
        pos = m.end()
    if not tokens:
        raise DSLParseError("empty mask")
    return tuple(tokens)


def _expand_mask_tokens(tokens: Sequence[str]) -> Iterator[str]:
    """Expand mask tokens into all digit strings (depth-first, lexicographic).

    Args:
        tokens: Sequence of tokens from :func:`parse_mask`.

    Yields:
        Digit strings matching the mask.
    """
    # Precompute choices for each position
    choices: List[Sequence[str]] = []
    for t in tokens:
        if t == "?d":
            choices.append(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])  # fixed order
        elif t.startswith("?{") and t.endswith("}"):
            set_digits = t[2:-1]
            if not set_digits or any(ch not in "0123456789" for ch in set_digits):
                raise DSLParseError(f"invalid digit set in token: {t}")
            choices.append(list(set_digits))
        else:  # pragma: no cover - parser guarantees tokens
            raise DSLParseError(f"unsupported token: {t}")

    # DFS iterative to avoid recursion limits (though masks are small)
    stack: List[Tuple[int, List[str]]] = [(0, [])]
    while stack:
        idx, acc = stack.pop()
        if idx == len(choices):
            yield "".join(acc)
            continue
        # iterate in forward order but push reversed to keep DFS lexicographic
        for ch in reversed(choices[idx]):
            stack.append((idx + 1, acc + [ch]))


# -----------------------------
# Action parsing
# -----------------------------


@dataclass(frozen=True)
class Layout:
    kind: str  # 'D', 'S', or 'ZIP'
    reverse: bool = False
    # For ZIP only
    zip_left: Tuple[str, bool] | None = None  # ('d'|'s', reversed)
    zip_right: Tuple[str, bool] | None = None


@dataclass(frozen=True)
class Action:
    layouts: Tuple[Layout, ...]


_SEED_RE = re.compile(
    r"^\s*(?:@(?P<lit>[0-9]+)|#range\((?P<a>[0-9]+)\.\.(?P<b>[0-9]+)\)|#mask\((?P<mask>[^)]*)\))\s*$"
)


def parse_program(program: str) -> Tuple[SeedSpec, Action]:
    """Parse a full DSL program of the form ``<seed> -> <action>``.

    Args:
        program: The input DSL string.

    Returns:
        (SeedSpec, Action)

    Raises:
        DSLParseError: on syntax errors.
    """
    if "->" not in program:
        raise DSLParseError("program must contain '->'")
    seed_str, action_str = program.split("->", 1)
    seed_str = seed_str.strip()
    action_str = action_str.strip()

    # Seed
    m = _SEED_RE.match(seed_str)
    if not m:
        raise DSLParseError("invalid seed; expected @<digits>, #range(a..b), or #mask(…) ")
    if m.group("lit") is not None:
        seed = SeedSpec(literal=m.group("lit"))
    elif m.group("a") is not None and m.group("b") is not None:
        seed = SeedSpec(range_start=m.group("a"), range_end=m.group("b"))
    else:
        mask_inner = m.group("mask") or ""
        tokens = parse_mask(mask_inner)
        seed = SeedSpec(mask_tokens=tokens)

    action = parse_action(action_str)
    return seed, action


def parse_action(action: str) -> Action:
    """Parse the action (layouts) portion.

    Args:
        action: Layout sequence, e.g. "D S~", "zip(ds)", "S D".

    Returns:
        Action object.

    Raises:
        DSLParseError: if any layout token is invalid.
    """
    parts = [p for p in action.strip().split() if p]
    if not parts:
        raise DSLParseError("action is empty")

    layouts: List[Layout] = []
    for p in parts:
        if p in ("D", "D~", "S", "S~"):
            kind = p[0]
            reverse = p.endswith("~")
            layouts.append(Layout(kind=kind, reverse=reverse))
            continue
        if p.startswith("zip(") and p.endswith(")"):
            inner = p[4:-1].strip()
            left, right = _parse_zip_spec(inner)
            layouts.append(
                Layout(kind="ZIP", zip_left=left, zip_right=right)
            )
            continue
        raise DSLParseError(f"unknown layout token: {p!r}")

    return Action(tuple(layouts))


def _parse_zip_spec(spec: str) -> Tuple[Tuple[str, bool], Tuple[str, bool]]:
    """Parse the specification inside ``zip( … )``.

    Accepts either compact forms (e.g., "ds", "d~s", "sd~") or spaced
    forms (e.g., "d s~").

    Returns:
        ((left_kind, left_rev), (right_kind, right_rev)) where kind in {'d','s'}.
    """
    s = spec.replace(" ", "")
    if not s:
        raise DSLParseError("empty zip spec")

    # Normalize e.g. 'd~s~' or 'ds' or 's~d'
    # Find tokens in order: [ 'd'|'s' ][ '~'? ][ 'd'|'s' ][ '~'? ]
    i = 0
    tokens: List[Tuple[str, bool]] = []
    while i < len(s):
        k = s[i]
        if k not in {"d", "s"}:
            raise DSLParseError("zip() expects 'd' and 's' operands")
        i += 1
        rev = False
        if i < len(s) and s[i] == "~":
            rev = True
            i += 1
        tokens.append((k, rev))
        if len(tokens) > 2:
            raise DSLParseError("zip() takes exactly two operands")
    if len(tokens) != 2:
        raise DSLParseError("zip() requires two operands, e.g., zip(ds) or zip(d s~)")
    if tokens[0][0] == tokens[1][0]:
        raise DSLParseError("zip() requires one 'd' and one 's' operand")
    return tokens[0], tokens[1]


# -----------------------------
# Action evaluation
# -----------------------------


def evaluate_action(digits: str, action: Action) -> str:
    """Apply an Action to a seed digit string.

    The seed digits are preserved except where a layout explicitly reverses the
    digits stream via ``D~`` or uses interleaving via ``zip``.

    Args:
        digits: The seed digit string.
        action: Parsed action.

    Returns:
        The resulting output string for this seed.
    """
    d_stream = list(digits)
    s_stream = [shift_of_digit(ch) for ch in d_stream]

    out_parts: List[str] = []
    for layout in action.layouts:
        if layout.kind == "D":
            seq = d_stream[::-1] if layout.reverse else d_stream
            out_parts.append("".join(seq))
        elif layout.kind == "S":
            seq = s_stream[::-1] if layout.reverse else s_stream
            out_parts.append("".join(seq))
        elif layout.kind == "ZIP":
            if not layout.zip_left or not layout.zip_right:
                raise DSLSemanticError("malformed ZIP layout")  # pragma: no cover
            left = _select_stream(d_stream, s_stream, layout.zip_left)
            right = _select_stream(d_stream, s_stream, layout.zip_right)
            if len(left) != len(right):
                # With valid seeds, lengths always match
                raise DSLSemanticError("zip() operand lengths differ")
            interleaved: List[str] = []
            for a, b in zip(left, right):
                interleaved.append(a)
                interleaved.append(b)
            out_parts.append("".join(interleaved))
        else:  # pragma: no cover - exhaustive
            raise DSLSemanticError(f"unknown layout kind: {layout.kind}")

    return "".join(out_parts)


def _select_stream(d_stream: Sequence[str], s_stream: Sequence[str], spec: Tuple[str, bool]) -> List[str]:
    kind, rev = spec
    base = d_stream if kind == "d" else s_stream
    return list(reversed(base)) if rev else list(base)


# -----------------------------
# CLI
# -----------------------------


@app.callback(invoke_without_command=True)
def _root(  # type: ignore[unused-argument]
    dsl_help: bool = typer.Option(
        False,
        "--dsl-help",
        help="Show DSL syntax help and exit.",
    ),
):
    """Root CLI callback. Use --dsl-help to print the DSL quick reference."""
    if dsl_help:
        typer.echo(_dsl_help_text())
        raise typer.Exit(0)


@app.command("run")
def cli_run(
    program: str = typer.Argument(
        ..., help="DSL program: '<seed> -> <action>' (quote to preserve spaces)"
    ),
) -> None:
    """Expand the DSL program and print each output line to stdout.

    Example:
        ndfillgen run "@12 -> D S"    # prints: 12!@
        ndfillgen run "#mask(?d?d) -> zip(ds)"  # prints 100 lines
    """
    try:
        seed_spec, action = parse_program(program)
        for seed in seed_spec.expand():
            out = evaluate_action(seed, action)
            sys.stdout.write(out + "\n")
    except (DSLParseError, DSLSemanticError) as e:
        typer.secho(f"error: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)


@app.command("keyspace")
def cli_keyspace(
    seed: str = typer.Argument(..., help="Seed only: @<digits> | #mask(...) | #range(a..b)"),
) -> None:
    """Print the number of seed expansions for a seed specification.

    This does not inspect the action because layouts do not change cardinality.
    """
    # Reuse the seed parser by composing a dummy action (ignored here)
    try:
        seed_spec, _ = parse_program(f"{seed} -> D")
    except DSLParseError as e:
        typer.secho(f"error: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Count lazily but efficiently; avoid consuming massive memory.
    count = 0
    for _ in seed_spec.expand():
        count += 1
    typer.echo(str(count))


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
