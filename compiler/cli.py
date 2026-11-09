"""Small installed CLI over the same Gemmini Python compilation service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import compile_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="admitted linalg or merlin_iface source")
    parser.add_argument("--output", type=Path, required=True, help="fresh compilation directory")
    args = parser.parse_args(argv)
    try:
        result = compile_source(args.input.read_text(encoding="utf-8"), output=args.output)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"gemmini-opt: {type(exc).__name__}: {exc}\n")
    print(result.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
