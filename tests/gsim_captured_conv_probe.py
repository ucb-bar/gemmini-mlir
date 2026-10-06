"""Compare a pinned contraction object with an explicit target schedule.

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
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_resident_conv import GoldenResidentConv
from mlir_oot.golden_resident_stripe_conv import GoldenResidentStripeConv
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.tables import rtl_facts as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    driver_source = Path(__file__).resolve()
    driver_source_sha256 = sha(driver_source)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture-dir", type=Path, required=True)
    ap.add_argument(
        "--schedule",
        choices=(
            "control",
            "compact",
            "full_reduction",
            "strided_resident",
            "stride_residue",
            "coalesced_resident_a",
            "mesh_flip",
            "compact_commands",
            "resident_acc_stripes",
            "capacity_cached_b",
            "resident_weight_packets",
            "flat_resident_planes",
            "flat_control_loops",
        ),
        required=True,
    )
    ap.add_argument("--bn", type=int)
    ap.add_argument("--bm", type=int, help="explicit accumulator stripe tiles")
    ap.add_argument("--row-tiles", type=int, help="explicit complete-B M tile group")
    ap.add_argument("--benchmark-header", type=Path)
    ap.add_argument("--weight-issue-tiles", type=int)
    ap.add_argument("--retain-weight-commands", action="store_true")
    ap.add_argument("--prefetch-b", action="store_true")
    ap.add_argument("--llvm-bin", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--max-cycles", type=int, default=5000000)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()
    fixture = args.fixture_dir.resolve()
    source_receipt = json.loads((fixture / "receipt.json").read_text())
    dense = "m" in source_receipt["shape"]
    shape = (Shape if dense else ConvShape)(**source_receipt["shape"])
    if dense and args.schedule not in (
        "control",
        "coalesced_resident_a",
        "mesh_flip",
        "resident_acc_stripes",
        "capacity_cached_b",
    ):
        raise ValueError("dense fixture requires an explicit dense schedule")
    if not dense and args.schedule in (
        "coalesced_resident_a",
        "mesh_flip",
        "resident_acc_stripes",
        "capacity_cached_b",
    ):
        raise ValueError("selected dense schedule requires a dense fixture")
    if args.prefetch_b and args.schedule != "compact":
        raise ValueError("weight prefetch requires explicit compact schedule")
    if args.row_tiles is not None and args.schedule != "capacity_cached_b":
        raise ValueError("row tile choice requires complete cached B")
    if (
        args.weight_issue_tiles is not None
        and args.schedule != "resident_weight_packets"
    ):
        raise ValueError(
            "weight packet size requires explicit resident weight packet schedule"
        )
    if args.retain_weight_commands and args.schedule != "resident_weight_packets":
        raise ValueError(
            "weight loop retention requires explicit resident weight packets"
        )
    if args.bn is not None:
        if args.schedule in ("control", "mesh_flip", "compact_commands"):
            raise ValueError(
                "exact control and isolated mesh mode do not accept tile overrides"
            )
        shape = replace(shape, bn=args.bn)
    if args.bm is not None:
        if args.schedule != "resident_acc_stripes":
            raise ValueError(
                "M stripe override requires explicit resident accumulator stripes"
            )
        shape = replace(shape, bm=args.bm, cache_a=False, prefetch_b=False)
    shape.validate()
    if not dense and shape.explicit_halo:
        raise ValueError("this paired resident probe requires unpadded inputs")
    counts = (
        {"a": shape.m * shape.k, "b": shape.k * shape.n, "expected": shape.m * shape.n}
        if dense
        else {
            "a": shape.h * shape.w * shape.cin,
            "b": 9 * shape.cin * shape.cout,
            "expected": shape.oh * shape.ow * shape.cout,
        }
    )
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
        if args.schedule == "compact_commands":
            generator = GoldenResidentConv(
                shape,
                **source_receipt.get("resident_options", {}),
                compact_commands=True,
            )
        elif args.schedule == "flat_control_loops":
            from mlir_oot.golden_flat_conv import GoldenFlatConv

            generator = GoldenFlatConv(
                shape,
                wide_a=True,
                separate_b_bank=True,
                virtual_padding=True,
                loop_spatial=True,
            )
        elif args.schedule == "flat_resident_planes":
            generator = GoldenResidentConv(
                shape, flat_spatial_planes=True, compact_commands=True
            )
        elif args.schedule == "resident_weight_packets":
            options = dict(source_receipt.get("resident_options", {}))
            options.update(
                prefetch_b=True,
                compact_commands=args.retain_weight_commands,
                weight_issue_tiles=(
                    args.weight_issue_tiles
                    if args.weight_issue_tiles is not None
                    else 2
                ),
            )
            generator = GoldenResidentConv(shape, **options)
        elif args.schedule == "capacity_cached_b":
            from mlir_oot.dense_schedule import select_capacity_cached_b

            generator, capacity_proof = select_capacity_cached_b(
                GoldenGemm(shape),
                row_tiles=args.row_tiles if args.row_tiles is not None else 1,
            )
            shape = generator.shape
        elif args.schedule == "resident_acc_stripes":
            from mlir_oot.golden_resident_stripe_gemm import GoldenResidentStripeGemm

            generator = GoldenResidentStripeGemm(shape, stripe_tiles=shape.bm)
            shape = generator.shape
        elif args.schedule == "mesh_flip":
            if not shape.cache_a or not shape.reuse_b:
                raise ValueError(
                    "mesh flip screen needs a complete cached A stationary-reuse control"
                )
            shape = replace(shape, reuse_b=False)
            placement = source_receipt.get("prefetch_b_rows")
            generator = GoldenGemm(
                shape,
                resident_a_load_tiles=source_receipt.get("resident_a_load_tiles", 1),
                prefetch_b_rows=tuple(placement) if placement else None,
            )
        elif args.schedule == "coalesced_resident_a":
            placement = source_receipt.get("prefetch_b_rows")
            generator = GoldenGemm(
                shape,
                resident_a_load_tiles=4,
                prefetch_b_rows=tuple(placement) if placement else None,
            )
        elif args.schedule == "stride_residue":
            from mlir_oot.conv_schedule import source_stride_resource_layout

            generator, _resource_choice = source_stride_resource_layout(
                shape, row_residue=True
            )
            shape = generator.conv
        elif args.schedule == "strided_resident":
            input_rows = (shape.cin // F.DIM) * (shape.h + 2) * (shape.w + 2)
            generator = GoldenResidentConv(
                shape,
                weight_base=((input_rows + F.DIM - 1) // F.DIM) * F.DIM,
                source_stride=True,
            )
        else:
            generator = (
                GoldenResidentConv(shape, prefetch_b=args.prefetch_b)
                if args.schedule == "compact"
                else GoldenResidentStripeConv(shape)
            )
        module = generator.build()
        module.body.block.first_op.properties["sym_name"] = StringAttr(symbol)
        compilation = compile_module(module, args.llvm_bin, w)
        if args.schedule == "capacity_cached_b":
            resource = capacity_proof
        elif args.schedule == "resident_acc_stripes":
            resource = {
                "input_rows": generator.input_rows,
                "weight_rows": generator.weight_rows,
                "weight_base_row": generator.bbase,
                "stripe_tiles": generator.stripe_tiles,
                "accumulator_rows": generator.stripe_tiles * shape.bn * F.DIM,
                "full_reduction_weight_lifetime": "all increasing K panels retained across M stripes for current N group",
                "input_lifetime": "complete A reserved through every N group; immutable caller storage",
            }
        elif args.schedule == "flat_control_loops":
            resource = {
                "band_rows": generator.band_rows,
                "input_panel_rows": generator.shape.bm * 4 * F.DIM,
                "accumulator_rows": generator.shape.bm * shape.bn * F.DIM,
            }
        elif dense:
            resource = {
                "input_rows": shape.bm * ((shape.k + F.DIM - 1) // F.DIM) * F.DIM,
                "resident_a_load_tiles": generator.resident_a_load_tiles,
                "prefetch_b_rows": generator.prefetch_b_rows,
                "weight_panel_rows": shape.bn * F.DIM,
                "accumulator_rows": shape.bm * shape.bn * F.DIM,
            }
        elif args.schedule in (
            "compact",
            "strided_resident",
            "stride_residue",
            "compact_commands",
            "resident_weight_packets",
            "flat_resident_planes",
        ):
            resource = {
                "input_rows": shape.cin // F.DIM * generator.plane,
                "weight_rows": shape.bn * F.DIM,
                "weight_base_row": generator.bbase,
                "accumulator_rows": len(generator.row_tiles) * shape.bn * F.DIM,
                "weight_issue_tiles": generator.weight_issue_tiles,
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
    arrays = dict(data, check_a=data["a"], check_b=data["b"])
    for i, (name, array) in enumerate(arrays.items()):
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
extern const int8_t captured_check_a[{counts["a"]}], captured_check_b[{counts["b"]}];
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
 for(int i=0;i<{counts["a"]};i++)if(captured_a[i]!=captured_check_a[i])return 3;
 for(int i=0;i<{counts["b"]};i++)if(captured_b[i]!=captured_check_b[i])return 4;
 printf("CAPTURED_CONV_PASS N={n} SCHEDULE={args.schedule}\\n");return 0;
}}
"""
    )
    benchmark_header = None
    if args.benchmark_header is not None:
        benchmark_header = w / "benchmark_buffer.h"
        shutil.copyfile(args.benchmark_header, benchmark_header)
        text = source.read_text()
        text = text.replace(
            "#include <stdio.h>", '#include <stdio.h>\n#include "benchmark_buffer.h"'
        )
        text = text.replace(
            " for(int i=0;i<N;i++)captured_box.output[i]=-37;",
            " merlin_benchmark_fill(captured_box.output,0xdb,sizeof(captured_box.output));",
        ).replace(
            " for(int i=0;i<2048;i++)captured_box.before[i]=captured_box.after[i]=0x5a;",
            " uint8_t expected_guard[2048];\n"
            " merlin_benchmark_fill(expected_guard,0x5a,sizeof(expected_guard));\n"
            " merlin_benchmark_fill(captured_box.before,0x5a,sizeof(captured_box.before));\n"
            " merlin_benchmark_fill(captured_box.after,0x5a,sizeof(captured_box.after));",
        )
        start = text.index(" for(int i=0;i<N;i++)if(captured_box.output")
        end = text.index(' printf("CAPTURED_CONV_PASS', start)
        text = (
            text[:start]
            + (
                " if(merlin_benchmark_first_difference(captured_expected,captured_box.output,sizeof(captured_box.output))!=sizeof(captured_box.output))return 1;\n"
                " if(merlin_benchmark_first_difference(expected_guard,captured_box.before,2048)!=2048||merlin_benchmark_first_difference(expected_guard,captured_box.after,2048)!=2048)return 2;\n"
                f" if(merlin_benchmark_first_difference(captured_check_a,captured_a,{counts['a']})!={counts['a']})return 3;\n"
                f" if(merlin_benchmark_first_difference(captured_check_b,captured_b,{counts['b']})!={counts['b']})return 4;\n"
            )
            + text[end:]
        )
        source.write_text(text)
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
    spike = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
    spike_argv = [
        str(spike),
        "-g",
        "--extension=gemmini",
        "--isa=rv64gc",
        "-m0x80000000:0x80000000",
        str(built.elf),
    ]
    spike_run = subprocess.run(
        spike_argv, capture_output=True, timeout=900, check=False
    )
    (w / "spike.stdout").write_bytes(spike_run.stdout)
    (w / "spike.stderr").write_bytes(spike_run.stderr)
    marker = f"CAPTURED_CONV_PASS N={n} SCHEDULE={args.schedule}"
    if spike_run.returncode or marker not in spike_run.stdout.decode():
        raise ValueError("complete strict target capsule failed")
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
        "schema": "gemmini_captured_dense_capsule_v1"
        if dense
        else "gemmini_captured_convolution_capsule_v1",
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
        "benchmark_header": {
            "path": str(benchmark_header),
            "sha256": sha(benchmark_header),
            "comparison": "Every byte outside ROI; shared exact word comparator with byte tails",
            "output_poison": "Every output byte0xdb before measured call",
        }
        if benchmark_header
        else None,
        "probe_driver": {
            "path": str(driver_source),
            "sha256": driver_source_sha256,
            "digest_stage": "startup before compilation or execution",
        },
        "kernel_cycles": int(cycles[0]) if len(cycles) == 1 else None,
        "metric_scope": "Complete device kernel on immutable ABI fixture; no host work or whole-model timing transfer",
        "run_completed": run.completed,
        "run_returncode": run.returncode,
        "outputs_checked": n if passed else None,
        "guards_checked": 4096 if passed else None,
        "immutable_input_bytes_checked": counts["a"] + counts["b"] if passed else None,
        "strict_target": {
            "status": "pass",
            "spike_argv": spike_argv,
            "engine_sha256": sha(spike),
            "stdout_sha256": sha(w / "spike.stdout"),
            "stderr_sha256": sha(w / "spike.stderr"),
        },
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
