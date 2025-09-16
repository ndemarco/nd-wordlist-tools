# Copilot Instructions for nd-wordlist-tools

## Project Overview
This repository provides command-line tools for generating and manipulating wordlists, with a focus on mask-based digit/symbol generation from circular sequences. The main logic is in `src/nd_wordlist_tools/`, with each tool as a separate module (e.g., `ndfillgen.py`, `ndpermute.py`).

## Architecture & Key Components
- **src/nd_wordlist_tools/**: Core Python modules. Each file is a CLI tool using [Typer](https://typer.tiangolo.com/) for argument parsing.
  - `ndfillgen.py`: Generates outputs from charsets using a mask of digit/symbol directives. Mask parsing and circular sequence logic are key.
  - `ndpermute.py`, `ndcombine.py`, `ndcase.py`: Other wordlist manipulation tools (see code for details).
- **demo/**: Example input/output files for testing and demonstration.
- **pyproject.toml**: Project metadata and dependencies.

## Developer Workflows
- **Run CLI tools**: Use `python -m src.nd_wordlist_tools.ndfillgen run ...` or directly as scripts.
- **Testing**: No formal test suite detected; validate changes by running tools on files in `demo/` and checking outputs.
- **Builds**: No build step required; pure Python, ready to run.
- **Debugging**: Add print statements or use Python debuggers. Typer CLI options make it easy to test different inputs.

## Project-Specific Patterns
- **Mask Directives**: Masks are strings like `d1s2d3`, parsed by `parse_mask()`. Directives are always 1-based and alternate between digit (`dN`) and symbol (`sN`).
- **Locale Handling**: Symbol output depends on the `--locale` option, mapped in `_LOCALES`.
- **Input/Output**: Tools accept input from files or stdin, and output to files or stdout. Use `--in -` and `--out -` for piping.
- **Error Handling**: Errors are raised with clear messages for invalid masks, positions, or locales.

## Integration & Dependencies
- **Typer**: All CLI interfaces use Typer. Arguments and options are defined in the function signatures.
- **No external data sources**: All logic is self-contained except for CLI input/output.

## Examples
- Generate masked output: `python -m src.nd_wordlist_tools.ndfillgen run --mask "d1s2d3" --charset "1234567890"`
- Use demo files: `python -m src.nd_wordlist_tools.ndfillgen run --mask "s1d2s3" --in demo/base.txt --out demo/permuted.out`

## Conventions
- All positions in masks are 1-based.
- Locale must match one of the keys in `_LOCALES`.
- Input files should have one charset per line.

## Key Files
- `src/nd_wordlist_tools/ndfillgen.py`: Main mask-based generator.
- `demo/`: Example data for manual testing.

---
If any section is unclear or missing important project-specific details, please provide feedback or point to relevant files to improve these instructions.
