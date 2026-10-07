"""One complete norm-certified PV target pair; no instruction-cycle inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import source_aware_pv_native_screen as screen
from merlin.common.paths import runtime_dir
from merlin.llvmlower.balanced_radix_groups import (
    encode_balanced_integer_planes,
    plan_balanced_radix_groups,
)
from merlin.llvmlower.projected_integer_pv_codegen import (
    ProjectedIntegerPVObserverPlan,
    emit_projected_integer_pv_observer,
)
from merlin.perf.layer_bench import build_program

from mlir_oot.no_fsm_audit import audit_elf

OUT = screen.OUT.parent / "source-aware-pv-prepared-norm-complete-target-20261007"
PRIOR = Path(
    "/scratch/agustin/tmp/gemmini-source-integer-pv-20261007/out/artifacts/probes/source-integer-pv-target-v4-20261007"
)
DIRECTED = PRIOR.parent / "source-integer-pv-prepared-directed-target-20261007"
NORMS = screen.OUT.parent / "source-aware-pv-prepared-norm-screen-20261007"
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)
ALLOCATOR = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/malloc.o"
)


def main():
    OUT.mkdir(exist_ok=False)
    assert json.loads((NORMS / "qualification.json").read_text())["status"] == "pass"
    factor = json.loads(
        (
            screen.OUT.parent
            / "source-aware-projected-pv-observers-20261007/diagnostics.json"
        ).read_text()
    )["contexts"][0]["source_quant_factor_word"]
    alpha = json.loads((screen.OUT / "declaration.json").read_text())[
        "typed_source_joins"
    ][0]["activation_scale_word"]
    plan = ProjectedIntegerPVObserverPlan(
        4,
        8,
        64,
        alpha,
        factor,
        screen.POLICY,
        plan_balanced_radix_groups(
            radix_bits=8,
            lhs_digits=2,
            rhs_digits=3,
            reduction_length=8,
            accumulator_bits=32,
            reconstruction_bits=64,
        ),
    )
    screen.save(OUT / "plan.json", __import__("dataclasses").asdict(plan))
    (OUT / "executor.c").write_text(
        emit_projected_integer_pv_observer(plan, symbol="integer_observer")
    )
    (OUT / "target_capability.h").write_bytes(
        (DIRECTED / "target_capability.h").read_bytes()
    )
    local = (
        screen.OLD
        / "out/artifacts/probes/dynamic-i8-attention-original-attribution-20261007/context_00_family1"
    )
    cap = (
        screen.OLD
        / "out/artifacts/probes/tiny-attention-original-v-projection-capture-v2-20261007/context_00"
    )
    p = np.load(local / "a.npy").reshape(4, 64, 8)
    code = np.ascontiguousarray(
        np.load(cap / "code.npy").reshape(8, 4, 64).transpose(1, 0, 2)
    )
    arrays = {
        "p": p,
        "code": code,
        "beta": np.load(cap / "channel_scale.npy").reshape(4, 64),
        "v": np.load(cap / "v.npy"),
        "original": np.load(
            screen.OLD
            / "out/artifacts/probes/dynamic-i8-attention-quant-observations-20261007/context_00/original.npy"
        ),
    }
    arrays["candidate"] = arrays["original"].copy()
    encoded_p = encode_balanced_integer_planes(
        np.rint(p.astype("f8") * 16384).astype("i8"),
        radix_bits=8,
        digits=2,
        epoch="original-P14:0",
    )
    encoded_v = encode_balanced_integer_planes(
        code, radix_bits=8, digits=3, epoch="original-integer-V:0"
    )
    assert encoded_p.admitted.all() and encoded_v.admitted.all()
    for g, group in enumerate(plan.radix.groups):
        a = np.concatenate([encoded_p.planes[pa] for pa, pb in group.pairs], axis=2)
        b = np.concatenate([encoded_v.planes[pb] for pa, pb in group.pairs], axis=1)
        arrays[f"partial{g}"] = (a.astype("i8") @ b.astype("i8")).astype("i4")
        np.save(OUT / f"group_{g}_reference.npy", arrays[f"partial{g}"])
    data = ["#include <stdint.h>\n"]
    for name, array in arrays.items():
        if array.dtype == np.dtype("f4"):
            kind, words = "float", [float(x).hex() + "f" for x in array.reshape(-1)]
        elif array.dtype == np.dtype("i4"):
            kind, words = "int32_t", [str(int(x)) for x in array.reshape(-1)]
        else:
            assert array.dtype == np.dtype("i1")
            kind, words = "int8_t", [str(int(x)) for x in array.reshape(-1)]
        data.append(
            f"_Alignas(64) const {kind} data_{name}[{array.size}]={{"
            + ",".join(words)
            + "};\n"
        )
    (OUT / "data.c").write_text("".join(data))
    driver = (
        (PRIOR / "main.c")
        .read_text()
        .replace(
            "data_partial3[16384],data_partial4[16384],data_partial5[16384]",
            "data_partial3[16384]",
        )
    )
    driver = driver.replace(
        "source_pairs,unadmitted_rows,product_calls",
        "source_pairs,onehot_rows,product_calls",
    )
    driver = driver.replace(
        "const float*,const int32_t*,const float*,int8_t*",
        "const float*,const int32_t*,const float*,const float*,int8_t*",
    )
    driver = driver.replace(
        "ref[6]={data_partial0,data_partial1,data_partial2,data_partial3,data_partial4,data_partial5}",
        "ref[4]={data_partial0,data_partial1,data_partial2,data_partial3}",
    )
    driver = driver.replace(
        "integer_observer(data_p,data_code,data_beta,output+64",
        "integer_observer(data_p,data_code,data_beta,data_v,output+64",
    )
    driver = driver.replace("98304", "65536").replace(
        "count.unadmitted_rows", "count.onehot_rows"
    )
    driver = driver.replace(
        "else control();\n  if(compare_bytes(ref",
        "else { memset(output+64,73,16384); if(candidate())return 9; for(size_t j=0;j<16384;j++)if(output[64+j]!=73)return 10; control(); }\n  if(compare_bytes(ref",
    )
    (OUT / "main.c").write_text(driver)
    sysroot = subprocess.check_output([str(GCC), "-print-sysroot"], text=True).strip()
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-ffp-contract=off",
        "--sysroot=" + sysroot,
        "-isystem",
        sysroot + "/include",
        "-I",
        str(runtime_dir() / "c"),
    ]
    commands = []
    for name in ("executor.c", "data.c", "main.c"):
        source = OUT / name
        obj = source.with_suffix(".o")
        extra = (
            ["-include", str(OUT / "target_capability.h")]
            if name == "executor.c"
            else []
        )
        argv = [
            str(screen.CLANG),
            *flags,
            *extra,
            "-MD",
            "-MF",
            str(source.with_suffix(".d")),
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        commands.append(argv)
        run = subprocess.run(argv, capture_output=True, text=True)
        (OUT / (source.stem + ".compile.stdout")).write_text(run.stdout)
        (OUT / (source.stem + ".compile.stderr")).write_text(run.stderr)
        assert run.returncode == 0, run.stderr
    reused = [
        PRIOR / "source.o",
        *[PRIOR / f"product_{k}/kernel.o" for k in (8, 16, 24)],
        ALLOCATOR,
    ]
    program = build_program(
        [OUT / "executor.o", OUT / "data.o", *reused, OUT / "main.o"],
        OUT / "build",
        target="gemmini",
        max_loaded_bytes=None,
    )
    audit = audit_elf(program.elf.read_bytes())
    screen.save(OUT / "nofsm.json", audit)
    assert audit["status"] == "pass"
    cmd = [
        str(GCC.with_name("spike")),
        "--isa=rv64gc",
        "--extension=gemmini",
        "-m0x80000000:0x400000000",
        str(program.elf),
    ]
    commands.append(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    (OUT / "spike.stdout").write_text(result.stdout)
    (OUT / "spike.stderr").write_text(result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        if line.startswith("INTEGER_PV_ROW "):
            fields = dict(x.split("=", 1) for x in line.split()[1:])
            rows.append(
                {k: int(v, 16 if k == "digest" else 10) for k, v in fields.items()}
            )
    pins = {
        str(p): screen.sha(p)
        for p in [
            Path(__file__),
            NORMS / "qualification.json",
            screen.CORE / "src/merlin/llvmlower/projected_integer_pv_codegen.py",
            screen.CORE / "src/merlin/llvmlower/balanced_radix_groups.py",
            DIRECTED / "target_capability.h",
            PRIOR / "main.c",
            PRIOR / "source.ll",
            *reused,
            screen.CLANG,
            GCC,
            GCC.with_name("spike"),
        ]
    }
    for root in [OUT]:
        for path in root.rglob("*"):
            if path.is_file():
                pins[str(path)] = screen.sha(path)
    for dependency in OUT.glob("*.d"):
        for name in (
            dependency.read_text().replace("\\\n", " ").split(":", 1)[1].split()
        ):
            path = Path(name)
            if path.is_file():
                pins[str(path.resolve())] = screen.sha(path)
    record = {
        "schema": "source_aware_pv_prepared_norm_complete_target_v1",
        "status": "pass"
        if result.returncode == 0 and "INTEGER_PV_COMPLETE" in result.stdout
        else "fail",
        "elf_path": str(program.elf),
        "elf_sha256": screen.sha(program.elf),
        "zeroFSM": True,
        "original_i8_exact": 16384,
        "exact_device_readout_words": 65536,
        "logical_product_plane_pairs": 6,
        "batched_group_callbacks": 4,
        "actual_KV_GEMM_calls": 16,
        "readout_bytes": 262144,
        "source_geometry": {"tokens": 8, "heads": 32, "kv_heads": 4, "channels": 64},
        "cost_scope": "Complete originalPV+14opquant consumer versus P14/i32V packing, all4batched/16KVdevice calls/readbacks, i64reconstruction, prepared row/column bound summaries, point certificates/source replay, onehot source replay, original observer, frame and stores. QK/P/projection/Vviews are original preexisting live inputs; workspace allocation and poison outside both ROI.",
        "rows": rows,
        "commands": commands,
        "returncode": result.returncode,
        "cycles": "UNKNOWN: Spike retired counters are not hardware cycles",
        "whole_model": "NOT RUN for this exact-observer successor; no normal route",
        "token_usage_available": False,
        "pins": pins,
    }
    if rows:
        record["mean_retired_instructions"] = {
            str(i): sum(r["instructions"] for r in rows if r["id"] == i) / 2
            for i in (0, 1)
        }
    screen.save(OUT / "qualification.json", record)
    print(result.stdout, flush=True)
    print(json.dumps({k: v for k, v in record.items() if k != "pins"}), flush=True)
    assert record["status"] == "pass", result.stderr


if __name__ == "__main__":
    main()
