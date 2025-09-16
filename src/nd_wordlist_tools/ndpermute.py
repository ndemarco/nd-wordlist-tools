#!/usr/bin/env python3

from typing import Iterator, List, Tuple, Optional
import typer
from typer import FileText

cli = typer.Typer(add_completion=False)

Pattern = str  # e.g. Literal["AB", "BA", "AA"]


def generate_pairs(
    base: List[str],
    extra: Optional[List[str]],
    patterns: List[Pattern],
) -> Iterator[Tuple[str, str]]:
    """
    Generate word pairs according to patterns:
    - AB: higher priority word first, lower priority second
    - BA: lower priority word first, higher priority second
    - AA: same word twice

    Priority is defined by list position in the combined corpus,
    with all base words first, then extra words.

    Avoid emitting duplicates. If extra corpus is given: only yield pairs where at least
    one word comes from the extra corpus.
    """
    if extra:
        combined = base + extra
        base_set = set(base)
        extra_set = set(extra)
    else:
        combined = base
        base_set = set(base)
        extra_set = set()  # unused in single-corpus mode

    # Drop duplicates while preserving order.
    # `raw` preserves original case for error reporting
    for raw in dict.fromkeys(patterns):
        p = raw.upper()
        if p == "AB":
            for i, w1 in enumerate(combined):
                for w2 in combined[i + 1 :]:
                    if extra and (w1 in base_set and w2 in base_set):
                        continue  # skip base-only pairs
                    yield (w1, w2)

        elif p == "BA":
            for i, w1 in enumerate(combined):
                for w2 in combined[i + 1 :]:
                    if extra and (w1 in base_set and w2 in base_set):
                        continue
                    yield (w2, w1)

        elif p == "AA":
            for w in combined:
                if extra and (w in base_set and w not in extra_set):
                    continue  # skip base-only self-pairs
                yield (w, w)

        else:
            raise ValueError(f"Unknown pattern: {raw}")


@cli.command()
def run(
    in_file: FileText = typer.Option(
        None,
        "--in-file",
        "-i",
        help="Input corpus file ('-' for stdin)",
    ),
    extra_corpus: Optional[FileText] = typer.Option(
        None, "--extra-corpus", "-e", help="Supplemental entries to base corpus"
    ),
    patterns: str = typer.Option(
        "AB",
        "--patterns",
        "-p",
        help="Comma-separated: AB,BA,AA",
        show_default=True,
    ),
    unique: bool = typer.Option(
        False, "--unique", "-u", help="Skip pairs where word1 == word2"
    ),
):
    """
    Generate word pairs per patterns. If extra is provided, only produce pairs
    where at least one word is from the extra corpus.
    """
    # Read corpora
    if in_file is None:
        # Print Typer's help text and exit with code 0, matching -h/--help
        from typer.main import get_command
        import click

        cmd = get_command(cli)
        ctx = click.Context(cmd)
        typer.echo(cmd.get_help(ctx))
        raise typer.Exit(code=0)
    base_words = [w for line in in_file for w in line.strip().split() if w]
    if not base_words:
        typer.echo("Error: Input corpus is empty.", err=True)
        raise typer.Exit(1)

    extra_words: list[str] = []
    if extra_corpus:
        extra_words = [w for line in extra_corpus for w in line.strip().split() if w]
        if not extra_words:
            typer.echo("Error: Extra corpus is empty.", err=True)
            raise typer.Exit(1)

    # Parse patterns
    pattern_list = [p.strip().upper() for p in patterns.split(",") if p.strip()]
    valid_patterns = {"AB", "BA", "AA"}
    for p in pattern_list:
        if p not in valid_patterns:
            typer.echo(f"Error: Invalid pattern '{p}'. Use AB, BA, or AA.", err=True)
            raise typer.Exit(1)

    # Generate pairs
    pairs = generate_pairs(
        base_words, extra_words if extra_words else None, pattern_list
    )

    if unique:
        pairs = ((w1, w2) for w1, w2 in pairs if w1 != w2)

    for w1, w2 in pairs:
        print(f"{w1}\t{w2}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
