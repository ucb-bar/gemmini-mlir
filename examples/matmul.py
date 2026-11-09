"""Emit a signed int8 matmul with int32 accumulation; no execution is implied."""

import argparse
from pathlib import Path

from gemmini import compile_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/build/matmul"))
    args = parser.parse_args()
    result = compile_source(Path(__file__).with_suffix(".mlir").read_text(encoding="utf-8"), output=args.output)
    print(result.manifest)


if __name__ == "__main__":
    main()
