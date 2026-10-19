"""Verify archive integrity without executing the generated compiler or tools."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    checked = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"unsafe archive path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
        if name in checked:
            raise ValueError(f"duplicate archive path: {name}")
        checked.add(name)
    inventory = json.loads((root / "experiment/compiler-inventory.json").read_text())
    published = {row["path"]: row for row in inventory["files"] if row["published"]}
    observed = {str(p.relative_to(root / "compiler")) for p in (root / "compiler").rglob("*") if p.is_file()}
    if observed != set(published):
        raise ValueError("compiler file set differs from original inventory")
    for name, row in published.items():
        path = root / "compiler" / name
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"compiler differs from original: {name}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
    usage = json.loads((root / "experiment/usage.json").read_text())
    tokens = usage["tokens"]
    if tokens["input_tokens"] + tokens["output_tokens"] != tokens["total_tokens"]:
        raise ValueError("usage total does not reconcile")
    if tokens["cached_input_tokens"] + tokens["fresh_input_tokens"] != tokens["input_tokens"]:
        raise ValueError("input subdivisions do not reconcile")
    for key, value in tokens.items():
        if sum(actor["totals"][key] for actor in usage["actors"]) != value:
            raise ValueError(f"actor totals do not reconcile: {key}")
    print(f"Verified {len(checked)} archive files and {len(published)} original compiler files.")
    print("Integrity verification only; no qualification or full-model pass is inferred.")


if __name__ == "__main__":
    main()
