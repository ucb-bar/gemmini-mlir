"""Compare a pinned convolution object with complete reduction residency.

An explicit fixture receipt owns geometry, numeric policy and operand bytes.
The control uses its exact original object. Both arms align data to common
addresses, and check every output plus both guard regions after measurement.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.tables import rtl_facts as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture-dir", type=Path, required=True)
    ap.add_argument(
        "--schedule", choices=("control", "compact", "full_reduction"), required=True
    )
    ap.add_argument("--bn", type=int)
    ap.add_argument("--prefetch-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--max-cycles", type=int, default=5000000)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()
    fixture = args.fixture_dir.resolve()
    source_receipt = json.loads((fixture / "receipt.json").read_text())
    shape = ConvShape(**source_receipt["shape"])
    if args.prefetch_b and args.schedule != "compact":
        raise ValueError("weight prefetch requires explicit compact schedule")
    if args.bn is not None:
        if args.schedule == "control":
            raise ValueError("exact control does not accept a tile override")
        shape = replace(shape, bn=args.bn)
    shape.validate()
    if shape.explicit_halo:
        raise ValueError("this paired resident probe requires unpadded inputs")
    counts = {
        "a": shape.h * shape.w * shape.cin,
        "b": 9 * shape.cin * shape.cout,
        "expected": shape.oh * shape.ow * shape.cout,
    }
    dtype = np.int8 if shape.output_dtype == "i8" else np.int32
    data = {}
    for name, count in counts.items():
        path = fixture / (name + ".bin")
        if sha(path) != source_receipt[name + "_sha256"]:
            raise ValueError("captured operand/output changed: " + name)
        array = np.fromfile(path, dtype if name == "expected" else np.int8)
        if len(array) != count:
            raise ValueError("captured extent disagrees with shape: " + name)
        data[name] = array
    w = args.workdir.resolve()
    w.mkdir(parents=True, exist_ok=False)
    (w / "capture_receipt.json").write_text(json.dumps(source_receipt, indent=2) + "\n")
    symbol = source_receipt["kernel_symbol"]
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol) is None:
        raise ValueError("explicit source symbol is not a C identifier")
    if args.schedule == "control":
        original = Path(source_receipt["source_kernel_object"])
        if sha(original) != source_receipt["source_kernel_object_sha256"]:
            raise ValueError("original control object changed")
        shutil.copyfile(original, w / "kernel.o")
        compilation = {
            "status": "exact_original_object",
            "object_sha256": sha(w / "kernel.o"),
            "original_path": str(original),
        }
        resource = None
    else:
        generator = (
            GoldenResidentConv(shape, prefetch_b=args.prefetch_b)
            if args.schedule == "compact"
            else GoldenResidentStripeConv(shape)
        )
        module = generator.build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(symbol)
        compilation = compile_module(module, args.llvm_bin, w)
        if args.schedule == "compact":
            resource = {
                "input_rows": shape.cin // F.DIM * generator.plane,
                "weight_rows": shape.bn * F.DIM,
                "weight_base_row": generator.bbase,
                "accumulator_rows": len(generator.row_tiles) * shape.bn * F.DIM,
            }
        else:
            resource = {
                "input_rows": generator.input_rows,
                "weight_rows": generator.weight_rows,
                "weight_base_row": generator.bbase,
                "stripe_rows": generator.stripe_rows,
                "accumulator_rows": generator.stripe_rows
                * generator.xt
                * shape.bn
                * F.DIM,
            }
    assembly = []
    for i, (name, array) in enumerate(data.items()):
        array.tofile(w / (name + ".bin"))
        assembly.append(
            ".section .data\n.balign "
            + ("1048576" if i == 0 else "64")
            + "\n.global captured_"
            + name
            + "\ncaptured_"
            + name
            + ':\n.incbin "'
            + str(w / (name + ".bin"))
            + '"\n'
        )
    (w / "inputs.S").write_text("".join(assembly))
    subprocess.run(
        [
            str(args.llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(w / "inputs.S"),
            "-o",
            str(w / "inputs.o"),
        ],
        check=True,
        capture_output=True,
    )
    n = counts["expected"]
    ctype = "int8_t" if shape.output_dtype == "i8" else "int32_t"
    source = w / "probe.c"
    source.write_text(
        f"""#include <stdint.h>
#include <stdio.h>
#define N {n}
extern int8_t captured_a[{counts["a"]}], captured_b[{counts["b"]}];
extern const {ctype} captured_expected[N];
struct output_box {{uint8_t before[2048];{ctype} output[N];uint8_t after[2048];}};
struct output_box captured_box __attribute__((aligned(1048576)));
extern void {symbol}(int8_t*,int8_t*,{ctype}*);
int main(void) {{
 for(int i=0;i<N;i++)captured_box.output[i]=-37;
 for(int i=0;i<2048;i++)captured_box.before[i]=captured_box.after[i]=0x5a;
 uint64_t begin,end;
 __asm__ volatile("csrr %0,mcycle":"=r"(begin)::"memory");
 {symbol}(captured_a,captured_b,captured_box.output);
 __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
 printf("CAPTURED_CONV_CYCLES %d\\n",(int)(end-begin));
 for(int i=0;i<N;i++)if(captured_box.output[i]!=captured_expected[i]){{
  printf("CAPTURED_CONV_FAIL i%d got%d want%d\\n",i,captured_box.output[i],captured_expected[i]);return 1;}}
 for(int i=0;i<2048;i++)if(captured_box.before[i]!=0x5a||captured_box.after[i]!=0x5a){{
  printf("CAPTURED_CONV_GUARD_FAIL i%d\\n",i);return 2;}}
 printf("CAPTURED_CONV_PASS N={n} SCHEDULE={args.schedule}\\n");return 0;
}}
"""
    )
    built = build_program(
        [source, w / "kernel.o", w / "inputs.o"],
        w,
        target="gemmini",
        max_loaded_bytes=None,
    )
    audit = audit_elf(built.elf.read_bytes())
    (w / "nofsm_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "pass":
        raise ValueError("paired convolution capsule gained FSM instruction")
    nm = subprocess.run(
        [str(args.llvm_bin / "llvm-nm"), "--defined-only", str(built.elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    addresses = {
        match[2]: int(match[1], 16)
        for match in re.finditer(
            r"^([0-9a-f]+) \w (captured_a|captured_b|captured_expected|captured_box)$",
            nm,
            re.MULTILINE,
        )
    }
    if len(addresses) != 4:
        raise ValueError("operand/output addresses missing from linked ELF")
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=args.max_cycles,
        timeout_s=args.timeout_s,
        backdoor=True,
        stdout_path=w / "gsim.stdout",
    )
    marker = f"CAPTURED_CONV_PASS N={n} SCHEDULE={args.schedule}"
    passed = bool(run.completed and run.returncode == 0 and marker in run.stdout_tail)
    cycles = re.findall(r"^CAPTURED_CONV_CYCLES (\d+)$", run.stdout_tail, re.MULTILINE)
    result = {
        "schema": "gemmini_captured_convolution_capsule_v1",
        "status": "pass" if passed else "fail",
        "schedule": args.schedule,
        "prefetch_b": args.prefetch_b,
        "shape": asdict(shape),
        "source_shape": source_receipt["shape"],
        "fixture_receipt_sha256": sha(fixture / "receipt.json"),
        "fixture_pins": {name: sha(w / (name + ".bin")) for name in data},
        "compilation": compilation,
        "resource": resource,
        "common_operand_addresses": addresses,
        "elf_sha256": built.elf_sha256,
        "source_c_sha256": sha(source),
        "kernel_cycles": int(cycles[0]) if len(cycles) == 1 else None,
        "metric_scope": "Complete device kernel on immutable ABI fixture; no host work or whole-model timing transfer",
        "run_completed": run.completed,
        "run_returncode": run.returncode,
        "outputs_checked": n if passed else None,
        "guards_checked": 4096 if passed else None,
        "nofsm_status": audit["status"],
        "engine_sha256": run.engine.get("binary_sha256"),
        "stdout": run.stdout_tail,
        "stderr": run.stderr_tail,
    }
    (w / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
