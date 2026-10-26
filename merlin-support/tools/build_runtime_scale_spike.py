#!/usr/bin/env python3
"""Build a separately pinned Spike oracle; never edit/install an existing library.

Use only after selecting/inspecting the model and recording the signed-scale
ReLU witness. The caller deliberately selects the new extension afterwards.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build(args):
    inputs = {
        name: pin(getattr(args, name)) for name in ("source", "header", "params_header")
    }
    patch = (
        Path(__file__).resolve().parents[1] / "patches/runtime_scale_spike_relu.patch"
    )
    original_params = args.params_header.read_bytes()
    if b"ACC_SCALE" not in original_params or b"float" not in original_params:
        raise ValueError("selected params header lacks the f32 accumulator-scale ABI")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, dest in (
        ("source", "gemmini.cc"),
        ("header", "gemmini.h"),
        ("params_header", "gemmini_params.h"),
    ):
        shutil.copyfile(getattr(args, name), args.output / dest)
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=args.output,
        check=True,
    )
    if (args.output / "gemmini_params.h").read_bytes() != original_params:
        raise ValueError("RNE/params-header pin changed during repair")
    command = [
        args.cxx,
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-O3",
        "-I",
        str(args.riscv / "include"),
        "-L",
        str(args.riscv / "lib"),
        "-Wl,-rpath," + str(args.riscv / "lib"),
        "-o",
        "libgemmini.so",
        "gemmini.cc",
    ]
    subprocess.run(command, cwd=args.output, check=True)
    manifest = {
        "schema": "gemmini.runtime_scale_spike_build.v1",
        "inputs": inputs,
        "patch": pin(patch),
        "command": command,
        "source": pin(args.output / "gemmini.cc"),
        "header": pin(args.output / "gemmini.h"),
        "params_header": pin(args.output / "gemmini_params.h"),
        "extension": pin(args.output / "libgemmini.so"),
        "qualification": "build_only; signed-ReLU probe and explicit extension selection required",
    }
    (args.output / "runtime-scale-build.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "header", "params-header", "riscv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args(argv)
    for name in ("source", "header", "params_header", "riscv", "output"):
        setattr(args, name, getattr(args, name).absolute())
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
