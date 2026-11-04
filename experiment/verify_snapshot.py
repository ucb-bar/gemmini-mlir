#!/usr/bin/env python3
"""Check archived source bytes and modes without executing the compiler."""
import hashlib, json, os
from pathlib import Path
root = Path(__file__).resolve().parent.parent
inventory = json.loads((root / "experiment/source-files.json").read_text())
digest = hashlib.sha256()
for row in inventory["files"]:
    path = root / row["branch_path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sha256"], row["branch_path"]
    assert bool(path.stat().st_mode & 0o111) == row["executable"], row["branch_path"]
    digest.update(row["source_path"].encode() + b"\0" + data + b"\0")
assert digest.hexdigest() == inventory["tree_sha256"], "source-tree digest"
print("Verified", len(inventory["files"]), "compiler files:", digest.hexdigest())
