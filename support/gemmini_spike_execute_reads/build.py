"""Isolated functional extension for stock narrow execute reads from ACC.

The pinned upstream extension aborts on an ACC A address and has an explicit
unsupported-preload TODO. This build copies its inputs and adds the stock RTL
read scale/activation semantics. It never replaces or installs a production
Spike binary/library. Timing remains a functional instruction proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("unsupported upstream source seam")
    return text.replace(old, new, 1)


def patch(text):
    text = replace_once(
        text,
        "gemmini_state.sys_acc_shift = new_sys_acc_shift;",
        "gemmini_state.sys_acc_shift = acc_scale_t_bits_to_acc_scale_t(new_sys_acc_shift);",
    )
    seam = "void gemmini_t::compute(reg_t a_addr, reg_t bd_addr, bool preload) {\n"
    helper = r"""  // Stock ExecuteController requests full=false and supplies CONFIG_EX scale/activation.
  auto read_execute_elem = [&](uint32_t address, size_t row, size_t column) -> elem_t {
    if (address == UINT32_MAX) return 0;
    if (address & (uint32_t(1) << 31)) {
      if (address & (uint32_t(1) << 29))
        throw std::invalid_argument("full-width ACC execute read is unsupported by stock RTL");
      const uint32_t base = address & UINT32_C(0x1fffffff);
      const acc_t value = gemmini_state.accumulator.at(base + row).at(column);
      return apply_activation(acc_scale(value, gemmini_state.sys_acc_shift), gemmini_state.sys_act);
    }
    return gemmini_state.spad.at(address + row).at(column);
  };
"""
    text = replace_once(text, seam, seam + helper)
    unsupported = """        // TODO: Handle preloads from accumulator, values are shifted and activated before preload
        if (~gemmini_state.preload_sp_addr != 0) {
          assert(((gemmini_state.preload_sp_addr >> 30) & 0b11) == 0); // Preloads from accumulator not supported
        }
"""
    text = replace_once(text, unsupported, "")
    text = replace_once(
        text,
        "gemmini_state.spad.at(gemmini_state.preload_sp_addr + r).at(c)",
        "read_execute_elem(gemmini_state.preload_sp_addr, r, c)",
    )
    text = replace_once(
        text,
        "gemmini_state.spad.at(bd_addr_real + i).at(j)",
        "read_execute_elem(bd_addr_real, i, j)",
    )
    text = replace_once(
        text,
        "gemmini_state.spad.at(a_addr_real + r).at(c)",
        "read_execute_elem(a_addr_real, r, c)",
    )
    text = replace_once(
        text,
        "gemmini_state.spad.at(bd_addr_real + r).at(c)",
        "read_execute_elem(bd_addr_real, r, c)",
    )
    return text


def build(args):
    source, riscv, output = (
        value.resolve() for value in (args.source, args.riscv, args.output)
    )
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name in ("gemmini.cc", "gemmini.h", "gemmini_params.h"):
        inputs[name] = pin(source / name)
        shutil.copy2(source / name, output / name)
    builder = Path(__file__).resolve()
    shutil.copy2(builder, output / "build.py")
    original = (output / "gemmini.cc").read_text()
    (output / "gemmini.cc").write_text(patch(original))
    shutil.copy2(riscv / "bin/spike", output / "spike")
    compiler = Path(shutil.which("g++")).resolve()
    command = [
        str(compiler),
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-O3",
        "-I" + str(riscv / "include"),
        str(output / "gemmini.cc"),
        "-o",
        str(output / "libgemmini_execute_reads.so"),
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    (output / "compile.stdout").write_text(run.stdout)
    (output / "compile.stderr").write_text(run.stderr)
    if run.returncode:
        raise RuntimeError(run.stderr)
    result = {
        "schema": "gemmini_spike_execute_reads_isolated_build_v1",
        "inputs": inputs,
        "compiler": pin(compiler),
        "builder": pin(output / "build.py"),
        "compiler_argv": command,
        "production_spike": pin(riscv / "bin/spike"),
        "outputs": {
            name: pin(output / name)
            for name in (
                "spike",
                "gemmini.cc",
                "gemmini.h",
                "gemmini_params.h",
                "libgemmini_execute_reads.so",
            )
        },
        "production_modified": False,
        "scope": "New functional capability, not timing or bitstream equivalence; unchanged-SPAD path requires independent output/hist regression closure",
        "unsupported": "Full-width ACC execute reads explicitly refuse; normalization activations are not newly qualified",
    }
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(output / "build.json"),
                "library_sha256": result["outputs"]["libgemmini_execute_reads.so"][
                    "sha256"
                ],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--riscv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    build(p.parse_args())
