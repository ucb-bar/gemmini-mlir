"""Build an isolated observational extension; never install into production."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(
            "Unknown extension seam; expected exactly one owned insertion point"
        )
    return text.replace(old, new, 1)


def build(source, riscv, output):
    source, riscv, output = map(Path, (source, riscv, output))
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name in ("gemmini.cc", "gemmini.h", "gemmini_params.h"):
        inputs[name] = pin(source / name)
        shutil.copy2(source / name, output / name)
    hook = Path(__file__).with_name("telemetry.h")
    shutil.copy2(hook, output / "telemetry.h")
    shutil.copy2(Path(__file__), output / "build.py")
    header = (output / "gemmini.h").read_text()
    header = replace_once(
        header,
        "class gemmini_t : public extension_t",
        '#include "telemetry.h"\n\nclass gemmini_t : public extension_t',
    )
    header = replace_once(
        header,
        "private:\n  gemmini_state_t gemmini_state;",
        "public:\n  void record_telemetry(reg_t pc, rocc_insn_t insn, reg_t rs1, reg_t rs2) {\n"
        "    gemmini_primitive_telemetry::instance().record(pc, insn.funct, rs1, rs2, gemmini_state);\n  }\n"
        "private:\n  gemmini_state_t gemmini_state;",
    )
    (output / "gemmini.h").write_text(header)
    code = (output / "gemmini.cc").read_text()
    seam = "reg_t xd = gemmini->CUSTOMFN(XCUSTOM_ACC)(u.r, xs1, xs2);"
    code = replace_once(
        code, seam, seam + "\n  gemmini->record_telemetry(pc, u.r, xs1, xs2);"
    )
    (output / "gemmini.cc").write_text(code)
    # Execute identical Spike bytes with a separately pinned extension library.
    shutil.copy2(riscv / "bin/spike", output / "spike")
    command = [
        "g++",
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-O3",
        "-I" + str(riscv / "include"),
        str(output / "gemmini.cc"),
        "-o",
        str(output / "libgemmini_telemetry.so"),
    ]
    subprocess.run(command, check=True)
    result = {
        "schema": "gemmini_spike_telemetry_isolated_build_v1",
        "inputs": inputs,
        "spike_source": pin(riscv / "bin/spike"),
        "hook": pin(output / "telemetry.h"),
        "builder": pin(output / "build.py"),
        "compiler_argv": command,
        "outputs": {
            name: pin(output / name)
            for name in (
                "spike",
                "gemmini.cc",
                "gemmini.h",
                "gemmini_params.h",
                "libgemmini_telemetry.so",
            )
        },
        "production_modified": False,
        "activation": "MERLIN_GEMMINI_TELEMETRY output path; absent means disabled",
        "scope": "Functional operand observations only; no hardware cycles or physical DRAM model",
    }
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "build_receipt": str(output / "build.json"),
                "extension_sha256": result["outputs"]["libgemmini_telemetry.so"][
                    "sha256"
                ],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--riscv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.riscv, args.output)
