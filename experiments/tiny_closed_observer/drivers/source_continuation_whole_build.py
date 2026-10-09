"""Normal source-bound lazy table prototype on the immutable full 2004 control.

Model IDs identify experimental evidence only. Generic table and continuation
selection are proved from the typed source DAG in Merlin, not function labels.
This file owns the stock ABI/mode glue and controlled existing-object link.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.late_quant_rne import _functions, _identity, _tokens
from merlin.llvmlower.source_continuation_outline import (
    SourceContinuationBinding,
    outline_source_continuations,
)
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    emit_immutable_bytes_llvm,
    emit_source_interval_lookup,
    find_closed_scalar_i8_observers,
)
from merlin.llvmlower.source_expression_interval_llvm import (
    rewrite_source_interval_lookup,
)
from merlin.runtime.backends.spike_model import _transform_host_ir
from merlin.runtime.host_provider import _function_abis

from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf

Y = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
C = Path("/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006")
O = Y / "out/artifacts/probes/source-continuation-normal-whole-20261007"
OLD = Y / "out/artifacts/probes/tiny-rectangular-whole-20261006/whole"
TABLES = Y / "out/artifacts/probes/source-interval-calibration-v2-20261006"
N = (
    C
    / "out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic"
)
B = Path(
    "/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build"
)
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
GCC = Path(
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
)
PRIOR = (
    Y
    / "out/artifacts/probes/source-expression-interval-table-20261006/normal_whole_v3/target_v2/controlled_link.json"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


commands = []


def run(argv):
    argv = list(map(str, argv))
    commands.append(argv)
    return subprocess.run(argv, capture_output=True, text=True, check=True)


def pin(paths):
    return {str(p): sha(p) for p in paths if Path(p).is_file()}


def main():
    assert not O.exists(), "Fresh outputs only"
    O.mkdir(parents=True)
    sealed = json.loads(
        (Y / "docs/perf_records/source_continuation_lazy_qualified.json").read_text()
    )
    for path, digest in sealed["pins"].items():
        assert sha(path) == digest, path
    typed = N / "typed_prepacket.generic.mlir"
    source = OLD / "lower/model.ll"
    module = parse_mlir_text(typed.read_text())
    effects = IntervalEffectContract(True, True, True, True, True)
    proofs, refused = find_closed_scalar_i8_observers(module, effects=effects)
    assert len(proofs) == 22 and not refused
    assert len({p.expression.canonical_sha256 for p in proofs}) == 1
    table = build_source_interval_table(
        proofs[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    assert (
        table.data
        == (
            Y
            / "out/artifacts/probes/source-interval-partition-analysis-20261006/table_16.bin"
        ).read_bytes()
    )
    (O / "table.ll").write_text(
        emit_immutable_bytes_llvm(
            table.data, symbol="source_interval_table", alignment=64
        )
    )
    (O / "lookup.c").write_text(
        emit_source_interval_lookup(
            table_name="source_interval_table",
            activation_name="source_activation",
            quantizer_name="source_quantize",
            lookup_name="source_lookup_activation",
            leading_bits=16,
        )
    )
    activation = (TABLES / "source_activation.ll").read_text()
    activation, continuation = outline_source_continuations(
        activation,
        bindings=(
            SourceContinuationBinding("source_activation", proofs[0].expression),
        ),
        expected_source_sha256=hashlib.sha256(activation.encode()).hexdigest(),
        effects=effects,
    )
    assert activation == (TABLES / "source_activation.ll").read_text().replace(
        "alwaysinline", "noinline cold", 1
    )
    (O / "source_activation.ll").write_text(activation)
    (O / "source_quantize.ll").write_bytes((TABLES / "source_quantize.ll").read_bytes())

    control, hook = _transform_host_ir(
        source,
        O / "control/host_llvm",
        merlin_host_llvm_transform(LLVM, combine_clamp=True),
    )
    assert sha(control) == sha(OLD / "host_llvm/model.ll")
    assert sha(control.parent / "model.native.ll") == sha(
        OLD / "host_llvm/model.native.ll"
    )
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O3",
        "-ffreestanding",
        "-fno-builtin",
    ]
    for native in (False, True):
        suffix = ".native" if native else ""
        out = O / f"control/expanded{suffix}.ll"
        run(
            [
                LLVM / "llvm-link",
                "-S",
                control.parent / f"model{suffix}.ll",
                B / f"host_llvm/expanded_bridge{suffix}.ll",
                "-o",
                out,
            ]
        )
        assert sha(out) == sha(OLD / f"host_llvm/expanded{suffix}.ll")
    run(
        [
            LLVM / "clang",
            *flags,
            "-c",
            O / "control/expanded.ll",
            "-o",
            O / "control/model.o",
        ]
    )
    assert sha(O / "control/model.o") == sha(OLD / "model.o")
    save(O / "control/source_binding.json", hook)

    for native in (False, True):
        case = O / ("native" if native else "target")

        def transform(path, work, native=native):
            original = path.read_text()
            changed, report = rewrite_source_interval_lookup(
                original,
                proofs=proofs,
                table=table,
                lookup_symbol="source_lookup_activation",
                effects=effects,
            )
            assert len(report["routes"]) == 44
            tokens, selected_tokens = _tokens(original), _tokens(changed)
            bodies, selected_bodies = _functions(tokens), _functions(selected_tokens)
            selected = {r["function_body_index"] for r in report["routes"]}
            assert len(selected) == 22
            abis = _function_abis(original)
            edits, guards = [], []
            for index in sorted(selected):
                body = bodies[index]
                start = max(
                    t.start
                    for t in tokens
                    if t.text == "define" and t.start < body[0].start
                )
                stop = next(
                    t.end for t in tokens if t.text == "}" and t.start > body[-1].end
                )
                name = _identity(
                    next(
                        t.text
                        for t in tokens
                        if start <= t.start < body[0].start and t.text.startswith("@")
                    )
                )
                assert (
                    abis[name].result == "void"
                    and abis[name].arguments == ("ptr",) * 5
                    and not abis[name].variadic
                )
                assert all(ch.isascii() and (ch.isalnum() or ch in "._") for ch in name)
                old_body = original[start:stop]
                new_body = selected_bodies[index]
                left = max(
                    t.start
                    for t in selected_tokens
                    if t.text == "define" and t.start < new_body[0].start
                )
                right = next(
                    t.end
                    for t in selected_tokens
                    if t.text == "}" and t.start > new_body[-1].end
                )
                text = changed[left:right]
                source_name, rne_name = (
                    name + ".__source_interval_original",
                    name + ".__source_interval_rne",
                )
                assert old_body.startswith("define internal void @" + name)
                clone = old_body.replace(
                    "define internal void @" + name, "define void @" + source_name, 1
                )
                candidate = text.replace(
                    "define internal void @" + name, "define void @" + rne_name, 1
                )
                edits.append((left, right, clone + "\n" + candidate))
                guards.append(
                    {
                        "original_symbol": name,
                        "source_symbol": source_name,
                        "candidate_symbol": rne_name,
                        "original_body_sha256": hashlib.sha256(
                            old_body.encode()
                        ).hexdigest(),
                        "candidate_body_sha256": hashlib.sha256(
                            text.encode()
                        ).hexdigest(),
                    }
                )
            for left, right, replacement in sorted(edits, reverse=True):
                changed = changed[:left] + replacement + changed[right:]
            for g in guards:
                changed += (
                    "\ndeclare void @"
                    + g["original_symbol"]
                    + "(ptr,ptr,ptr,ptr,ptr)\n"
                )
            (work / "source_table.ll").write_text(changed)
            glue = ["#include <fenv.h>"] if native else []
            for index, g in enumerate(guards):
                args = ",".join("void* a" + str(i) for i in range(5))
                values, types = (
                    ",".join("a" + str(i) for i in range(5)),
                    ",".join(["void*"] * 5),
                )
                mode = (
                    "if(fegetround()!=FE_TONEAREST)"
                    if native
                    else 'unsigned mode;__asm__ volatile("csrr %0,frm":"=r"(mode)::"memory");if(mode)'
                )
                glue.extend(
                    [
                        f'extern void source_{index}({types}) __asm__("{g["source_symbol"]}");',
                        f'extern void rne_{index}({types}) __asm__("{g["candidate_symbol"]}");',
                        f'void guard_{index}({args}) __asm__("{g["original_symbol"]}");',
                        f"void guard_{index}({args}){{{mode}source_{index}({values});else rne_{index}({values});}}",
                    ]
                )
            (work / "mode_guard.c").write_text("\n".join(glue) + "\n")
            cflags = (
                ["-O3", "-ffp-contract=off", "-fPIC"]
                if native
                else [*flags, "-ffp-contract=off"]
            )
            for csource in (O / "lookup.c", work / "mode_guard.c"):
                run(
                    [
                        LLVM / "clang",
                        *cflags,
                        "-S",
                        "-emit-llvm",
                        csource,
                        "-o",
                        work / (csource.stem + ".ll"),
                    ]
                )
            run(
                [
                    LLVM / "llvm-link",
                    "-S",
                    work / "source_table.ll",
                    O / "source_activation.ll",
                    O / "source_quantize.ll",
                    work / "lookup.ll",
                    work / "mode_guard.ll",
                    O / "table.ll",
                    "-o",
                    work / "linked.ll",
                ]
            )
            # Pure lookup/quantizer inlining exposes the same source-bounded RNE
            # observers. The cold source activation retains its unchanged body.
            late = merlin_host_llvm_transform(LLVM, combine_clamp=True)(
                work / "linked.ll", work / "late_rne"
            )
            if native:
                late = work / "late_rne/model.native.ll"
            suffix = ".native" if native else ""
            expanded = work / f"expanded{suffix}.ll"
            run(
                [
                    LLVM / "llvm-link",
                    "-S",
                    late,
                    B / f"host_llvm/expanded_bridge{suffix}.ll",
                    "-o",
                    expanded,
                ]
            )
            refs = Counter(
                t.text for t in _tokens(expanded.read_text()) if t.text.startswith("@")
            )
            old_refs = Counter(
                t.text
                for t in _tokens((OLD / f"host_llvm/expanded{suffix}.ll").read_text())
                if t.text.startswith("@")
            )
            writers = json.loads(
                (B / "device_host_abi/writer_contracts.json").read_text()
            )
            symbols = {
                "@" + n
                for r in writers["routes"]
                for n in (
                    r["symbol"],
                    r["borrowed_symbol"],
                    r["symbol"] + "__fresh_tensor_result",
                )
            }
            assert {n: refs[n] for n in symbols} == {n: old_refs[n] for n in symbols}
            assert sum(r["calls"] for r in writers["routes"]) == 155
            save(
                work / "source_binding.json",
                {
                    "source_bindings": report,
                    "whole_helper_guards": guards,
                    "continuation": continuation,
                    "all_original_device155_references_conserved": True,
                    "effects": vars(effects),
                    "leading_bits": table.leading_bits,
                    "readonly_bytes": len(table.data),
                    "source_table_sha256": table.sha256,
                    "scope": "Normal pre-object host hook on frozen2004 upstream LLVM/typed source, not a new whole capture or lowering.",
                    "pins": pin(
                        [
                            typed,
                            source,
                            B / "device_host_abi/writer_contracts.json",
                            *work.glob("*"),
                            *work.glob("late_rne/*"),
                        ]
                    ),
                },
            )
            return expanded

        selected, hook = _transform_host_ir(source, case / "host_llvm", transform)
        save(case / "normal_hook_receipt.json", hook)

    prior = json.loads(PRIOR.read_text())
    argv = [
        str(O / "control/model.o")
        if x in prior["baseline_objects"] and Path(x).name == "model.o"
        else x
        for x in prior["baseline_reproduction_argv"]
    ]
    argv[-1] = str(O / "baseline_reproduced.elf")
    run(argv)
    assert (
        sha(O / "baseline_reproduced.elf")
        == sha(OLD / "model.elf")
        == "39c55f91736b5dffbb1d0460b5372891c492a669202988c722b468e6fe297d9a"
    )
    (O / "baseline_reproduced.elf").unlink()
    os.link(OLD / "model.elf", O / "baseline_reproduced.elf")
    selected = O / "target/host_llvm/expanded.ll"
    run([LLVM / "clang", *flags, "-c", selected, "-o", O / "target/model.o"])
    candidate_argv = [
        str(O / "target/model.o") if x == str(O / "control/model.o") else x
        for x in argv
    ]
    candidate_argv[-1] = str(O / "target/model.elf")
    objects = [Path(x) for x in candidate_argv if x.endswith(".o")]
    before = pin(objects)
    run(candidate_argv)
    assert before == pin(objects)
    audit = audit_elf((O / "target/model.elf").read_bytes())
    assert audit["status"] == "pass"
    save(O / "target/model.nofsm_audit.json", audit)
    dump = run(
        [GCC.with_name("riscv64-unknown-elf-objdump"), "-dr", O / "target/model.o"]
    ).stdout
    (O / "target/model.dump").write_text(dump)
    nm = run([LLVM / "llvm-nm", "--print-size", O / "target/model.o"]).stdout
    (O / "target/model.nm").write_text(nm)
    calls = [
        line
        for line in dump.splitlines()
        if "R_RISCV_CALL" in line and line.split()[-1] == "source_activation"
    ]
    assert len(calls) == 45  # 44 selected lanes plus retained generic lookup
    assert (
        len(
            [
                line
                for line in nm.splitlines()
                if line.endswith(" source_interval_table")
                and int(line.split()[1], 16) == 512 * 1024
            ]
        )
        == 1
    )
    save(
        O / "build.json",
        {
            "schema": "normal_lazy_source_continuation_whole_build_v1",
            "commands": commands,
            "control_job": 2004,
            "baseline_byte_exact": True,
            "baseline_elf_sha256": sha(OLD / "model.elf"),
            "baseline_compile_object_byte_exact": True,
            "only_changed_linked_object": "model.o",
            "candidate_objects": before,
            "candidate_link_argv": candidate_argv,
            "candidate_elf_sha256": sha(O / "target/model.elf"),
            "candidate_model_object_sha256": sha(O / "target/model.o"),
            "cold_source_call_sites": len(calls),
            "leading_bits": 16,
            "table_bytes": len(table.data),
            "all155devicebindings": True,
            "whole_hardware_cycles": "UNKNOWN",
            "inherited_marker": "37bdf9be0856",
            "marker_contract": "Inherited nonunique1880 marker; exact ELF and object pins identify this candidate.",
            "scope": "All8tokens/22layers/256000 outputs; normal host hook plus controlled2004 object link, not a fresh upstream recapture. Explicit16bit table+source-exact cold continuation; no whole-cycle forecast from2053.",
            "token_usage_available": False,
            "pins": pin(
                [
                    *objects,
                    typed,
                    source,
                    PRIOR,
                    Path(__file__),
                    *O.glob("*"),
                    *O.glob("control/*"),
                    *O.glob("control/host_llvm/*"),
                    *O.glob("target/*"),
                    *O.glob("target/host_llvm/*"),
                    *O.glob("native/*"),
                    *O.glob("native/host_llvm/*"),
                    *O.glob("*/host_llvm/late_rne/*"),
                    C / "src/merlin/llvmlower/source_expression_interval.py",
                    C / "src/merlin/llvmlower/source_expression_interval_llvm.py",
                    C / "src/merlin/llvmlower/source_continuation_outline.py",
                    LLVM / "clang",
                    LLVM / "llvm-link",
                    LLVM / "llvm-as",
                    GCC,
                ]
            ),
        },
    )
    print("NORMAL_WHOLE_LAZY_BUILD_PASS", sha(O / "target/model.elf"), flush=True)


if __name__ == "__main__":
    main()
