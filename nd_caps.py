#!/usr/bin/env python3
"""
nd_caps — patterned capitalization generator for two-word pairs.

MVP features implemented:

- DSL of directives: [w1:|w2:]U(start,len)
  * If no word selector is given, the directive applies to BOTH words (expanded internally).
  * start >= 0 counts from the left; start < 0 counts from the right (-1 is last char).
  * len ≥ 1; each directive generates all case permutations within that span (2^len variants,
    including the all-lowercase mask).

- Per-line, per-word normalization for the actual word length s:
  * Convert right-anchored starts to left indices (s - k), clip to bounds,
    merge duplicate starts by keeping the largest max_len.
  * Build a plan of (start, max_len) entries sorted by start.
  * Cache plans by (target, s) for reuse across words of the same length.

- Streaming I/O: reads stdin or --in; writes stdout or --out.
- Deterministic enumeration order:
  * For each start (ascending), iterate masks 0..(2^span-1) in ascending order before moving to the next start.
  * If both words have variants, emit the cross product in a stable nested order.

- Stats and logging:
  * --stats-only prints JSON stats (no candidates).
  * -v / --verbose prints plan builds/cache-hits to stderr (one log per new (target,s)).
  * Skipped spans (out-of-bounds/clipped-to-zero) counted per plan build.

- Dedup/progress:
  * Optional in-run memory dedup: --dedup-memory (per-process set).
  * Optional persistent dedup across runs: --seen-db PATH (SQLite, stdlib).
    Candidates already present are not re-emitted (and counted as duplicates_suppressed).

Requirements: Python 3.10+, Typer (CLI). Everything else is stdlib.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import typer

# ----------------------------- Data structures ----------------------------- #


@dataclass(frozen=True)
class Family:
    """
    A directive family before normalization to a particular word length.

    anchor:
        'L' if anchored from the left (start >= 0 in the DSL),
        'R' if anchored from the right (start < 0 in the DSL).

    start:
        Non-negative position parameter. For 'L', it's the left index a (0-based).
        For 'R', it's the right offset b (i.e., original DSL start was -b).

    len_max:
        Maximum run length implied by this family (all 1..len_max are allowed).
    """

    anchor: str  # 'L' or 'R'
    start: int
    len_max: int


@dataclass(frozen=True)
class PlanEntry:
    """
    A normalized, per-length plan entry for a word of length s.

    start:
        Left-anchored start index (0-based) within the word.

    len_max:
        Maximum run length that remains in-bounds for this start.
    """

    start: int
    len_max: int


@dataclass
class Stats:
    """Aggregated runtime statistics."""

    lines_total: int = 0
    lines_valid: int = 0
    lines_invalid: int = 0
    skipped_spans: int = 0  # during plan building
    duplicates_suppressed: int = 0  # due to seen store or in-run dedup
    variants_total_emitted: int = 0
    max_variants_per_line: int = 0
    max_variants_line_index: Optional[int] = None
    duration_ms: float = 0.0
    plans_built: int = 0
    plans_cache_hits: int = 0


# ----------------------------- Parsing & DSL ------------------------------- #

RULE_RE = re.compile(
    r"""^(?:(?P<target>w1|w2)\s*:\s*)?U\(\s*(?P<start>-?\d+)\s*,\s*(?P<len>\d+)\s*\)$""",
    re.IGNORECASE,
)


class CLIError(typer.BadParameter):
    """CLI-friendly exception for argument/DSL issues."""


def parse_rule(rule: str) -> Tuple[Optional[str], Family]:
    """
    Parse a single DSL rule like "U(1,5)" or "w1:U(-3,2)".

    Returns:
        (target, Family)
        target is 'w1' | 'w2' | None (None means applies to both and will be expanded by the caller)
    """
    m = RULE_RE.match(rule.strip())
    if not m:
        raise CLIError(f"Invalid directive syntax: {rule!r}")
    target = m.group("target")
    start_i = int(m.group("start"))
    len_max = int(m.group("len"))
    if len_max < 1:
        raise CLIError(f"len must be >= 1 in {rule!r}")

    if start_i >= 0:
        fam = Family(anchor="L", start=start_i, len_max=len_max)
    else:
        k = -start_i
        if k < 1:
            # This case can't happen since start_i < 0 implies k >= 1
            raise CLIError(f"Invalid negative start in {rule!r}")
        fam = Family(anchor="R", start=k, len_max=len_max)

    return (target.lower() if target else None, fam)


def expand_targets(
    parsed: Sequence[Tuple[Optional[str], Family]]
) -> Tuple[List[Family], List[Family]]:
    """
    Expand target=None families to both w1 and w2.

    Returns:
        (families_w1, families_w2)
    """
    w1: List[Family] = []
    w2: List[Family] = []
    for tgt, fam in parsed:
        if tgt is None:
            w1.append(fam)
            w2.append(fam)
        elif tgt.lower() == "w1":
            w1.append(fam)
        elif tgt.lower() == "w2":
            w2.append(fam)
        else:
            raise CLIError(f"Unknown target: {tgt!r}")
    return w1, w2


def collapse_families(families: Sequence[Family]) -> List[Family]:
    """
    Compile-time dedup: for identical (anchor, start), keep only the largest len_max.

    This is safe because U(start, Lmax) implies all U(start, l) for l in 1..Lmax
    for the same anchor. Cross-anchor cannot be collapsed without knowing s.
    """
    best: Dict[Tuple[str, int], int] = {}
    for fam in families:
        key = (fam.anchor, fam.start)
        prev = best.get(key)
        if prev is None or fam.len_max > prev:
            best[key] = fam.len_max
    return [Family(anchor=a, start=i, len_max=L) for (a, i), L in sorted(best.items())]


# ----------------------------- Plans & caching ----------------------------- #


def normalize_and_plan(
    families: Sequence[Family], s: int
) -> Tuple[List[PlanEntry], int]:
    """
    Normalize families to a left-anchored, merged plan for word length s.

    Steps:
      - Convert right-anchored starts to left indices (s - b).
      - Drop out-of-bounds; clip len_max to not exceed word end.
      - Merge by identical start (keep the largest len_max).
      - Sort by start.

    Returns:
      (plan_entries, skipped_count)
    """
    skipped = 0
    candidates: Dict[int, int] = {}  # start -> len_max (max)

    for fam in families:
        if fam.anchor == "L":
            start_idx = fam.start
        else:  # 'R'
            start_idx = s - fam.start

        # Drop if out of bounds
        if start_idx < 0 or start_idx >= s:
            skipped += 1
            continue

        cap = s - start_idx
        if cap <= 0:
            skipped += 1
            continue

        eff_len = min(fam.len_max, cap)  # cap the eff_len to start_idx + max_len
        if eff_len <= 0:
            skipped += 1
            continue

        prev = candidates.get(start_idx)
        if prev is None or eff_len > prev:
            candidates[start_idx] = eff_len

    plan = [PlanEntry(start=k, len_max=v) for k, v in sorted(candidates.items())]
    return plan, skipped


class PlanCache:
    """
    Cache of normalized plans keyed by word length.

    Uses separate instances for w1 and w2 because compile-time families can
    differ by target.
    """

    def __init__(self, families: Sequence[Family]) -> None:
        self._families: List[Family] = list(families)
        self._cache: Dict[int, List[PlanEntry]] = {}
        self.skipped_spans_for_length: Dict[int, int] = {}

    def get(self, s: int) -> Tuple[List[PlanEntry], bool, int]:
        """
        Get or build the plan for length s.

        Returns:
            (plan, cache_hit, skipped_count_for_this_build)
        """
        if s in self._cache:
            return self._cache[s], True, 0
        plan, skipped = normalize_and_plan(self._families, s)
        self._cache[s] = plan
        self.skipped_spans_for_length[s] = skipped
        return plan, False, skipped


# ------------------------------ Uppercasing -------------------------------- #


def _apply_uppercase_mask(word: str, start: int, span_len: int, mask_bits: int) -> str:
    """
    Apply a bitmask to uppercase within a contiguous span.

    For i in [0..span_len-1], if the i-th bit of mask_bits is 1, uppercase
    character at position (start + i). Positions outside the span are unchanged.
    """
    if span_len <= 0:
        return word
    # Convert to list for efficient per-char mutation
    chars = list(word)
    for i in range(span_len):
        if (mask_bits >> i) & 1:
            idx = start + i
            # Bounds are guaranteed by plan construction, but guard anyway
            if 0 <= idx < len(chars):
                chars[idx] = chars[idx].upper()
    return "".join(chars)


def enumerate_variants(word: str, plan: Sequence[PlanEntry]) -> tuple[List[str], int]:
    """
    Enumerate capitalization variants per the normalized plan using bitmasks.

    Semantics:
      For each entry (start, len_max), emit ALL case permutations across that span:
      masks 0..(2^len_max - 1). Mask 0 (no uppercase) is included.

    Deterministic order:
      - Iterate entries in ascending start (as sorted in the plan),
      - For each entry, iterate mask_bits ascending (0..2^len_max - 1).
      - First time a concrete variant appears, it is emitted; later duplicates are
        suppressed but counted.

    Returns:
      (variants, duplicates_suppressed_within_word)
    """
    out: List[str] = []
    seen: set[str] = set()
    suppressed = 0

    for entry in plan:
        span_len = entry.len_max
        # Enumerate all masks over this span
        limit = 1 << span_len  # 2^span_len
        for mask_bits in range(limit):
            v = _apply_uppercase_mask(word, entry.start, span_len, mask_bits)
            if v in seen:
                suppressed += 1
                continue
            seen.add(v)
            out.append(v)

    return out, suppressed


# ------------------------------- Seen stores ------------------------------- #


class SeenStore:
    """Interface for stream-safe candidate deduplication."""

    def check_and_add(self, candidate: str) -> bool:
        """
        Return True if the candidate is newly added (not seen before).
        Return False if it has already been seen.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """Close resources (if any)."""
        pass


class MemorySeenStore(SeenStore):
    """In-memory set-based seen store."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check_and_add(self, candidate: str) -> bool:
        if candidate in self._seen:
            return False
        self._seen.add(candidate)
        return True


class SqliteSeenStore(SeenStore):
    """
    SQLite-backed seen store (std lib only). Suitable for large runs and reuse across sessions.

    The database contains a single table:
      CREATE TABLE IF NOT EXISTS seen (candidate TEXT PRIMARY KEY);
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (candidate TEXT PRIMARY KEY);")
        self._conn.commit()

    def check_and_add(self, candidate: str) -> bool:
        try:
            with self._conn:  # implicit transaction per insert
                self._conn.execute("INSERT INTO seen(candidate) VALUES (?)", (candidate,))
            return True
        except sqlite3.IntegrityError:
            # Already present
            return False

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# --------------------------------- Logging --------------------------------- #


def log_plan(
    target: str, s: int, plan: Sequence[PlanEntry], status: str, skipped: int
) -> None:
    """
    Human-friendly plan log to stderr, printed when a plan is BUILT or on CACHE_HIT if verbose.
    """
    print(
        f"[caps-plan] target={target} len={s} status={status} entries={len(plan)} skipped_spans={skipped}",
        file=sys.stderr,
    )
    for entry in plan:
        # Visual mask with '@' for max_len at this start
        mask = ["." for _ in range(s)]
        for i in range(entry.start, entry.start + entry.len_max):
            if 0 <= i < s:
                mask[i] = "@"
        mask_str = "".join(mask)
        print(
            f'  - start={entry.start} span_len={entry.len_max} mask="{mask_str}" '
            f'masks=2^{entry.len_max} ({1<<entry.len_max} variants)',
            file=sys.stderr,            
        )


# ---------------------------------- CLI ------------------------------------ #

app = typer.Typer(add_completion=False, help="nd_caps — patterned capitalization generator.")


def _open_in(path: Optional[Path]) -> Iterable[str]:
    if path is None:
        return sys.stdin
    return path.open("r", encoding="utf-8", errors="strict")


def _open_out(path: Optional[Path]):
    if path is None:
        return sys.stdout
    return path.open("w", encoding="utf-8", errors="strict")


def _validate_word(token: str) -> bool:
    return bool(token) and token.isascii() and token.islower() and token.isalpha()


def _process_line(
    line: str,
    line_index: int,
    plan_w1: List[PlanEntry],
    plan_w2: List[PlanEntry],
    outfh,
    stats: Stats,
    stats_only: bool,
    seen: Optional[SeenStore],
) -> None:
    stats.lines_total += 1
    line = line.strip()
    if not line:
        stats.lines_invalid += 1
        return

    parts = line.split()
    if len(parts) != 2 or not _validate_word(parts[0]) or not _validate_word(parts[1]):
        stats.lines_invalid += 1
        return

    stats.lines_valid += 1
    w1, w2 = parts[0], parts[1]

    # Build variants deterministically
    if plan_w1:
        variants_w1, sup1 = enumerate_variants(w1, plan_w1)
        stats.duplicates_suppressed += sup1
    else:
        variants_w1 = [w1]

    if plan_w2:
        variants_w2, sup2 = enumerate_variants(w2, plan_w2)
        stats.duplicates_suppressed += sup2
    else:
        variants_w2 = [w2]

    # Cross product emission
    emitted_this_line = 0
    for v1 in variants_w1:
        for v2 in variants_w2:
            candidate = f"{v1} {v2}"
            # Seen-store gating
            if seen is not None:
                if not seen.check_and_add(candidate):
                    stats.duplicates_suppressed += 1
                    continue
            if not stats_only:
                print(candidate, file=outfh)
            emitted_this_line += 1

    if emitted_this_line > stats.max_variants_per_line:
        stats.max_variants_per_line = emitted_this_line
        stats.max_variants_line_index = line_index

    stats.variants_total_emitted += emitted_this_line


@app.command()
def run(
    rules: List[str] = typer.Argument(
        ..., metavar="RULE ...", help="One or more directives like 'U(1,5)' or 'w1:U(-3,2)'."
    ),
    infile: Optional[Path] = typer.Option(
        None, "--in", "-i", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input file (default: stdin)."
    ),
    outfile: Optional[Path] = typer.Option(
        None, "--out", "-o", writable=True, help="Output file (default: stdout)."
    ),
    stats_only: bool = typer.Option(
        False, "--stats-only", help="Do not emit candidates; compute and print JSON stats to stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging to stderr."),
    dedup_memory: bool = typer.Option(
        False, "--dedup-memory", help="Enable in-run memory dedup (per-process set)."
    ),
    seen_db: Optional[Path] = typer.Option(
        None,
        "--seen-db",
        help="SQLite file for persistent dedup/progress. Candidates already present will NOT be re-emitted.",
    ),
) -> None:
    """
    Generate capitalization variants for two-word pairs using the nd_caps DSL.

    DSL:
      RULE := [w1:|w2:]U(start,len)
        - If no target prefix is given, the directive applies to BOTH words.
        - start >= 0 counts from the left; start < 0 counts from the right (−1 is last char).
        - len >= 1; the rule generates ALL case permutations within that span (2^len variants).

    Examples:
      nd-caps 'U(0,1)'                 # Uppercase first letter of both words
      nd-caps 'w1:U(1,3)' 'w2:U(-2,2)' # Separate targets
      nd-caps 'U(1,5)'                 # Both words; lengths 1..5 starting at index 1 (or from right when negative)
    """
    t0 = time.perf_counter()
    # Parse and expand rules
    try:
        parsed = [parse_rule(r) for r in rules]
    except CLIError as e:
        raise typer.BadParameter(str(e)) from e

    fam_w1_raw, fam_w2_raw = expand_targets(parsed)
    fam_w1 = collapse_families(fam_w1_raw)
    fam_w2 = collapse_families(fam_w2_raw)

    # Plan caches per target
    cache_w1 = PlanCache(fam_w1)
    cache_w2 = PlanCache(fam_w2)

    # Seen store selection
    seen: Optional[SeenStore] = None
    try:
        if seen_db is not None:
            seen = SqliteSeenStore(seen_db)
        elif dedup_memory:
            seen = MemorySeenStore()
    except Exception as e:
        raise typer.Exit(code=2) from e

    stats = Stats()

    # IO
    in_stream = _open_in(infile)
    out_stream = _open_out(outfile)

    try:
        for idx, line in enumerate(in_stream):
            # Compute plans for each word length
            # We only build/log when a new length is encountered.
            # Note: we parse words first to avoid computing on invalid lines.
            raw = line.strip()
            parts = raw.split()
            # Fast path: if invalid, let _process_line account for it.
            s1 = len(parts[0]) if len(parts) == 2 and _validate_word(parts[0]) else 0
            s2 = len(parts[1]) if len(parts) == 2 and _validate_word(parts[1]) else 0

            plan1, hit1, skipped1 = cache_w1.get(s1) if s1 > 0 else ([], True, 0)
            plan2, hit2, skipped2 = cache_w2.get(s2) if s2 > 0 else ([], True, 0)

            if not hit1 and verbose:
                log_plan("w1", s1, plan1, "BUILT", skipped1)
                stats.plans_built += 1
            elif hit1 and verbose and s1 > 0:
                log_plan("w1", s1, plan1, "CACHE_HIT", 0)
                stats.plans_cache_hits += 1

            if not hit2 and verbose:
                log_plan("w2", s2, plan2, "BUILT", skipped2)
                stats.plans_built += 1
            elif hit2 and verbose and s2 > 0:
                log_plan("w2", s2, plan2, "CACHE_HIT", 0)
                stats.plans_cache_hits += 1

            stats.skipped_spans += skipped1 + skipped2

            _process_line(
                line=line,
                line_index=idx,
                plan_w1=plan1,
                plan_w2=plan2,
                outfh=out_stream,
                stats=stats,
                stats_only=stats_only,
                seen=seen,
            )

    finally:
        if out_stream is not sys.stdout:
            out_stream.close()
        if seen is not None:
            seen.close()

    stats.duration_ms = (time.perf_counter() - t0) * 1000.0

    if stats_only:
        # Print JSON stats to stdout (not stderr) and exit with 0
        result = {
            "lines_total": stats.lines_total,
            "lines_valid": stats.lines_valid,
            "lines_invalid": stats.lines_invalid,
            "variants_total_emitted": stats.variants_total_emitted,
            "max_variants_per_line": stats.max_variants_per_line,
            "max_variants_line_index": stats.max_variants_line_index,
            "skipped_spans": stats.skipped_spans,
            "duplicates_suppressed": stats.duplicates_suppressed,
            "plans_built": stats.plans_built,
            "plans_cache_hits": stats.plans_cache_hits,
            "duration_ms": round(stats.duration_ms, 3),
        }
        # If stdout was redirected to a file via --out, we must still write stats to stdout.
        # So ensure we print here (stdout).
        print(json.dumps(result, ensure_ascii=False))
    # Otherwise, normal emission already written to out_stream


if __name__ == "__main__":
    app()
