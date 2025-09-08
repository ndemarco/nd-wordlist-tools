#!/usr/bin/env python3
"""
ndcase — shell-safe capitalization variant generator for two-word pairs.

DSL v2 (shell-safe):
  [WORDSEL:]OP:START,LEN
    WORDSEL ∈ { w1, w2, w* }   (default: w*)
    OP      ∈ { U }            (U = generate all case permutations over span)
    START   ∈ int
              • Positive values count from the left (1 = first character).
              • Negative values count from the right (-1 = last character).
              • 0 is invalid.
    LEN     ∈ int > 0

Examples:
  U:1,3         # both words: positions 1–3 (inclusive range in human terms)
  w1:U:-3,2     # word 1 only: last 3rd and 2nd chars
  w2:U:5,1      # word 2 only: 5th char

Multiple directives can be provided separated by spaces on the CLI or by newlines
in a rules file (lines starting with '#' are comments). Legacy syntax is not accepted.

Input:
  Lines of two whitespace-separated words, e.g., "hello world".

Output:
  All capitalization variants for each input line as directed by rules.

CLI:
  -r/--rule RULE        Repeatable rule option.
  --rules-file PATH     Load rules from a file (one per line; '#' for comments).
  -i/--in FILE          Input file (default: stdin).
  -o/--out PATH         Output file (default: stdout).
  --dedup-memory        In-run memory dedup (per-process set).
  --seen-db PATH        SQLite database to persist seen candidates (skip re-emits).
"""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import typer

app = typer.Typer(
    add_completion=False,
    help="Generate capitalization variants for two-word pairs using the DSL.",
)

# ---------- DSL parsing ----------

# Strict, no legacy accepted.
DIRECTIVE_RE = re.compile(r"^(?:(w[12]|w\*)\:)?([A-Za-z])\:(-?\d+),(\d+)$")

# ---------- Data structures / naming ----------

WordSel = str  # 'w1', 'w2', 'w*'


@dataclass(frozen=True)
class Directive:
    wordsel: WordSel  # 'w1' | 'w2' | 'w*'
    op: str  # currently 'U'
    start: int  # signed
    length: int  # > 0


@dataclass(frozen=True)
class DirectiveFamily:
    """Group of directives with same op, keyed by selector."""

    op: str
    spans_w1: Tuple[Tuple[int, int], ...]  # (start, length) after binding/normalization
    spans_w2: Tuple[Tuple[int, int], ...]


def parse_directive(text: str) -> Directive:
    """Parse a single directive in v2 DSL form."""
    m = DIRECTIVE_RE.match(text.strip())
    if not m:
        raise typer.BadParameter(
            f"Invalid directive '{text}'. Expected [w1|w2|w*:]OP:START,LEN (e.g., 'U:1,3' or 'w1:U:-3,2')."
        )
    wordsel, op, start_s, len_s = m.groups()
    wordsel = wordsel or "w*"
    if wordsel not in {"w1", "w2", "w*"}:
        raise typer.BadParameter(f"WORDSEL must be w1, w2, or w*; got '{wordsel}'.")
    if op not in {"U"}:
        raise typer.BadParameter(f"Unknown op '{op}'. Supported: U")
    try:
        start = int(start_s)
        length = int(len_s)
    except ValueError:
        raise typer.BadParameter(
            f"START and LEN must be integers; got '{start_s},{len_s}'."
        )
    if length <= 0:
        raise typer.BadParameter(f"LEN must be > 0; got {length}.")
    return Directive(wordsel=wordsel, op=op, start=start, length=length)


def parse_rules_inline(rules: Sequence[str]) -> List[Directive]:
    out: List[Directive] = []
    for r in rules:
        if not r.strip():
            continue
        out.append(parse_directive(r))
    return out


def parse_rules_file(path: Path) -> List[Directive]:
    out: List[Directive] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(parse_directive(s))
            except typer.BadParameter as e:
                raise typer.BadParameter(f"{path}:{ln}: {e}") from None
    return out


# ---------- Planning / binding ----------


def _bind_span(word_len: int, start: int, length: int) -> Tuple[int, int]:
    """
    Convert (start,length) into absolute [i, i+length) over word_len.
    Negative start counts from right; -1 is last char.
    Raises ValueError if the span falls outside the word.
    """
    if start < 0:
        i = word_len + start  # e.g., len=5, start=-1 -> i=4
    elif start == 0:
        raise ValueError(
            "Invalid START=0 is invalid. 1 represents the first character from the left.\n"
            "Negative positions count from the right (-1 represents the last character)."
        )
    else:
        i = start - 1  # convert to 1-based
    j = i + length
    if i < 0 or j > word_len:
        raise ValueError(
            f"Span out of range for word length {word_len}: start={start}, len={length} -> [{i},{j})"
        )
    return (i, length)


def build_span_plan(
    directives: Sequence[Directive],
    w1_len: int,
    w2_len: int,
) -> DirectiveFamily:
    """
    Build a normalized, absolute-span plan for op='U' across w1 and w2.
    Returns a DirectiveFamily consolidating spans per word.
    """
    spans_w1: List[Tuple[int, int]] = []
    spans_w2: List[Tuple[int, int]] = []
    for d in directives:
        if d.op != "U":
            # Currently only U is supported, but keep structure for future ops.
            continue
        targets = []
        if d.wordsel in ("w1", "w*"):
            targets.append(("w1", w1_len))
        if d.wordsel in ("w2", "w*"):
            targets.append(("w2", w2_len))
        for label, wlen in targets:
            try:
                abs_span = _bind_span(wlen, d.start, d.length)
            except ValueError as e:
                raise typer.BadParameter(str(e))
            if label == "w1":
                spans_w1.append(abs_span)
            else:
                spans_w2.append(abs_span)

    spans_w1 = coalesce_directive_families(spans_w1)
    spans_w2 = coalesce_directive_families(spans_w2)
    return DirectiveFamily(op="U", spans_w1=tuple(spans_w1), spans_w2=tuple(spans_w2))


def coalesce_directive_families(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge identical/overlapping/adjacent spans to reduce redundancy.
    Input/Output spans are (start, length) with absolute indices.
    """
    if not spans:
        return []
    # Convert to intervals [a,b)
    intervals = sorted(
        ((s, s + length) for (s, length) in spans), key=lambda x: (x[0], x[1])
    )
    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a <= cur_b:  # overlap or touch
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    # Back to (start, length)
    return [(a, b - a) for (a, b) in merged]


# ---------- Execution engine ----------


class SpanPlanCache:
    """
    Cache built plans keyed by (w1_len, w2_len, signature).
    Signature is a hashable tuple of directives.
    """

    def __init__(self) -> None:
        self._cache: dict[Tuple[int, int, Tuple[Directive, ...]], DirectiveFamily] = {}

    def get(
        self, w1_len: int, w2_len: int, directives: Sequence[Directive]
    ) -> DirectiveFamily:
        key = (w1_len, w2_len, tuple(directives))
        if key not in self._cache:
            self._cache[key] = build_span_plan(directives, w1_len, w2_len)
        return self._cache[key]


def _permute_case(word: str, spans: Sequence[Tuple[int, int]]) -> Iterator[str]:
    """
    For a single word, generate all case permutations over the union of spans.
    Non-span characters pass through unchanged.
    """
    if not spans:
        yield word
        return

    # Build a mask of indices that are part of any span.
    mask = [False] * len(word)
    for start, length in spans:
        for i in range(start, start + length):
            mask[i] = True

    # Collect indices to toggle
    toggle_idxs = [i for i, m in enumerate(mask) if m]
    if not toggle_idxs:
        yield word
        return

    # For each combination, build the variant
    base = list(word)
    n = len(toggle_idxs)
    for bits in range(1 << n):
        w = base[:]
        for k, idx in enumerate(toggle_idxs):
            c = w[idx]
            # For 'U', we generate *both* cases across the span via bitmask.
            if ((bits >> k) & 1) == 1:
                w[idx] = c.upper()
            else:
                w[idx] = c.lower()
        yield "".join(w)


def apply_family_to_pair(
    w1: str, w2: str, fam: DirectiveFamily
) -> Iterator[Tuple[str, str]]:
    """Apply a DirectiveFamily (currently op='U') to (w1, w2)."""
    if fam.op != "U":
        return
    for v1 in _permute_case(w1, fam.spans_w1):
        for v2 in _permute_case(w2, fam.spans_w2):
            yield (v1, v2)


# ---------- Dedup / persistence ----------


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS seen (candidate TEXT PRIMARY KEY)")
    conn.commit()


def seen_add_many(conn: sqlite3.Connection, cands: Iterable[str]) -> None:
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen(candidate) VALUES (?)", ((c,) for c in cands)
        )


def seen_filter_new(conn: sqlite3.Connection, cands: Iterable[str]) -> Iterator[str]:
    # Efficient membership test via LEFT JOIN approach is overkill here; simple SELECT works fine batch-wise.
    cur = conn.cursor()
    for c in cands:
        cur.execute("SELECT 1 FROM seen WHERE candidate=?", (c,))
        if cur.fetchone():
            continue
        yield c


# ---------- CLI ----------


def _open_in(path: Optional[Path]):
    if path is None:
        return sys.stdin
    return open(path, "r", encoding="utf-8")


def _open_out(path: Optional[Path]):
    if path is None:
        return sys.stdout
    return open(path, "w", encoding="utf-8")


@app.command()
def run(
    rules: List[str] = typer.Argument(
        None,
        metavar="RULE ...",
        help="One or more directives like 'U:1,3' or 'w1:U:-3,2'.",
    ),
    rule_opt: List[str] = typer.Option(
        None,
        "--rule",
        "-r",
        help="Add a directive (repeatable), e.g., -r U:1,3 -r w1:U:-2,2",
    ),
    rules_file: Optional[Path] = typer.Option(
        None,
        "--rules-file",
        help="Load directives from a file (one per line). Lines starting with '#' are comments.",
    ),
    infile: Optional[Path] = typer.Option(
        None, "--in", "-i", help="Input file (default: stdin)."
    ),
    outfile: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output file (default: stdout)."
    ),
    dedup_memory: bool = typer.Option(
        False, "--dedup-memory", help="Enable in-run memory dedup (per-process set)."
    ),
    seen_db: Optional[Path] = typer.Option(
        None,
        "--seen-db",
        help="SQLite file for persistent dedup/progress. Candidates already present will NOT be re-emitted.",
    ),
):
    """
    Generate capitalization variants for two-word pairs using the DSL.
    """
    # Gather and parse rules
    collected: List[str] = []
    if rules:
        collected.extend(rules)
    if rule_opt:
        collected.extend(rule_opt)
    if rules_file:
        parsed = parse_rules_file(rules_file)
    else:
        if not collected:
            raise typer.BadParameter(
                "At least one RULE or --rule/--rules-file is required."
            )
        parsed = parse_rules_inline(collected)

    # Read input & write output
    cache = SpanPlanCache()
    memory_seen: set[str] = set() if dedup_memory else set()

    # Setup DB if requested
    conn: Optional[sqlite3.Connection] = None
    writer = _open_out(outfile)
    try:
        if seen_db:
            conn = sqlite3.connect(seen_db)
            ensure_db(conn)

        with _open_in(infile) as reader:
            for line_num, line in enumerate(reader, 1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) != 2:
                    raise typer.BadParameter(
                        f"Input line {line_num}: expected exactly two whitespace-separated words, got: '{s}'"
                    )
                w1, w2 = parts[0], parts[1]
                fam = cache.get(len(w1), len(w2), parsed)

                # Generate candidates and serialize as "w1 w2"
                gen_iter = (
                    "{} {}".format(a, b) for (a, b) in apply_family_to_pair(w1, w2, fam)
                )

                # Dedup in-memory
                if dedup_memory:
                    gen_iter = (c for c in gen_iter if c not in memory_seen)

                # Dedup via DB
                if conn is not None:
                    gen_iter = seen_filter_new(conn, gen_iter)

                batch: List[str] = []
                for cand in gen_iter:
                    if dedup_memory:
                        memory_seen.add(cand)
                    batch.append(cand)

                    # Periodic flush to DB to avoid memory growth
                    if conn is not None and len(batch) >= 1000:
                        seen_add_many(conn, batch)
                        writer.write("\n".join(batch) + "\n")
                        writer.flush()
                        batch.clear()

                if conn is not None and batch:
                    seen_add_many(conn, batch)

                if batch:
                    writer.write("\n".join(batch) + "\n")

    finally:
        if writer is not sys.stdout:
            writer.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    app()
