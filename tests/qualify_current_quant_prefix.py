"""Seal the immutable complete source-prefix capsule; never submit jobs."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from current_quant_prefix_probe import extract, pin
from merlin.llvmlower.bounded_rne_interval import (
    build_integer_observation_table,
    emit_integer_observation_lookup,
    localize_bounded_rne_observer,
    validate_localized_rne_observer,
)
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from xdsl.dialects.builtin import TensorType, i8
from xdsl.dialects.linalg.ops import GenericOp

from mlir_oot.frontend.parse import parse_module
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.quant_prefix_capsule import digest, parse

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("/scratch/agustin/tmp/merlin-constant-rne-prefix-20261007")
FIXTURE = ROOT / "out/current_quant_prefix_v3"
CAPSULE = ROOT / "out/current_quant_prefix_capsule_v4"
SEAL = ROOT / "out/current_quant_prefix_seal_v2"
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")


class Elf:
    """Read bounded ELF64 sections/symbols, including zero-sized assembly labels."""

    def __init__(self, path):
        self.data = Path(path).read_bytes()
        if self.data[:6] != b"\x7fELF\x02\x01":
            raise ValueError("ELF64 little-endian binding required")
        offset = struct.unpack_from("<Q", self.data, 40)[0]
        stride, count, _names_index = struct.unpack_from("<HHH", self.data, 58)
        if stride != 64 or offset + count * stride > len(self.data):
            raise ValueError("bounded ELF section table required")
        self.sections = [
            struct.unpack_from("<IIQQQQIIQQ", self.data, offset + i * stride)
            for i in range(count)
        ]
        self.symbols = {}
        for section in self.sections:
            if section[1] != 2:
                continue
            strings = self.sections[section[6]]
            string_data = self.data[strings[4] : strings[4] + strings[5]]
            if section[9] != 24 or section[5] % 24:
                raise ValueError("bounded ELF64 symbol table required")
            for at in range(section[4], section[4] + section[5], 24):
                name, info, other, owner, value, size = struct.unpack_from(
                    "<IBBHQQ", self.data, at
                )
                label = string_data[name:].split(b"\0", 1)[0].decode()
                if label and owner and owner < len(self.sections):
                    self.symbols.setdefault(label, []).append(
                        (info, other, owner, value, size)
                    )

    def symbol(self, name):
        matches = self.symbols[name]
        if len(matches) != 1:
            raise ValueError("ambiguous requested ELF symbol: " + name)
        return matches[0]

    def bytes(self, name, count):
        _, _, owner, value, size = self.symbol(name)
        section = self.sections[owner]
        relative = value - section[3]
        if (
            section[1] == 8
            or relative < 0
            or relative + count > section[5]
            or (size and count > size)
        ):
            raise ValueError("symbol bytes exceed declared section extent")
        at = section[4] + relative
        if at + count > len(self.data):
            raise ValueError("symbol bytes exceed ELF file")
        return self.data[at : at + count]


def command(argv, name):
    argv = list(map(str, argv))
    (SEAL / (name + ".argv.json")).write_text(json.dumps(argv, indent=2) + "\n")
    with (SEAL / (name + ".log")).open("w") as output:
        result = subprocess.run(
            argv, stdout=output, stderr=subprocess.STDOUT, timeout=180, check=False
        )
    if result.returncode:
        raise ValueError("qualification command failed: " + name)


def main():
    SEAL.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((FIXTURE / "prelabel.json").read_text())
    capsule = json.loads((CAPSULE / "prelabel.json").read_text())
    # Original declarations are immutable. Reclose every embedded path/hash.
    embedded_pins = {}

    def reclose(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                actual = pin(value["path"])
                if actual["sha256"] != value["sha256"] or (
                    "bytes" in value and actual["bytes"] != value["bytes"]
                ):
                    raise ValueError(
                        "stale inherited source declaration: " + value["path"]
                    )
                embedded_pins[actual["path"]] = actual
            for item in value.values():
                reclose(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                reclose(item)

    reclose(capsule)
    original, argument, quant, padding = extract()
    from merlin.xdsl_dialects._common import text

    if text(original, generic=True) != (FIXTURE / "source.mlir").read_text():
        raise ValueError("actual source extraction changed")
    if (
        argument != fixture["source_argument_index"]
        or hashlib.sha256(quant.encode()).hexdigest()
        != fixture["source_quantization_sha256"]
    ):
        raise ValueError("actual source quantizer binding changed")
    if hashlib.sha256(padding.encode()).hexdigest() != fixture["source_padding_sha256"]:
        raise ValueError("actual source padding binding changed")
    folded = parse_module((FIXTURE / "constant_folded.mlir").read_text())
    targets = [
        op
        for op in folded.walk()
        if isinstance(op, GenericOp)
        and len(op.results) == 1
        and op.results[0].type == TensorType(i8, [1, 224, 224, 3])
    ]
    if len(targets) != 1:
        raise ValueError("one actual typed quantization map required")
    effects = IntervalEffectContract(True, True, True, True, True)
    observer = localize_bounded_rne_observer(
        targets[0].body.block,
        targets[0].body.block.args[0],
        symbol="original_scalar",
        effects=effects,
    )
    table = build_integer_observation_table(
        observer.proof, effects=effects, leading_bits=18, max_table_bytes=524288
    )
    validate_localized_rne_observer(observer)
    if (
        table.data != (FIXTURE / "table.bin").read_bytes()
        or table.expression_sha256 != fixture["expression_sha256"]
    ):
        raise ValueError(
            "complete source interval proof does not reproduce actual table bytes"
        )
    # Delivery adds a compile-only binary32 ABI assertion; frozen timed C/ELFs
    # remain untouched. All executor/table/callback bytes after declarations match.
    current_lookup = emit_integer_observation_lookup(
        table_name="observation_cells",
        source_name="original_scalar",
        lookup_name="exact_lookup",
        leading_bits=18,
    )
    frozen_lookup = (FIXTURE / "lookup_target.c").read_text()
    unchanged = current_lookup[current_lookup.index("extern const unsigned char") :]
    if not frozen_lookup[
        frozen_lookup.index("extern const unsigned char") :
    ].startswith(unchanged):
        raise ValueError("delivery changed the frozen lookup executor")
    (SEAL / "delivery_abi_guard.c").write_text(current_lookup)
    command(
        [
            LLVM / "clang",
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-fsyntax-only",
            SEAL / "delivery_abi_guard.c",
        ],
        "delivery_abi_guard",
    )
    for label, source in (
        ("source", FIXTURE / "candidate/target.ll"),
        ("linked", CAPSULE / "candidate_inline.ll"),
    ):
        command(
            [
                LLVM / "llvm-extract",
                "-S",
                "--func=original_scalar",
                source,
                "-o",
                SEAL / (label + "_fallback.ll"),
            ],
            label + "_fallback_extract",
        )
    command(
        [LLVM / "llvm-diff", SEAL / "source_fallback.ll", SEAL / "linked_fallback.ll"],
        "source_fallback_equivalence",
    )
    inline_text = (CAPSULE / "candidate_inline.ll").read_text()
    hot_calls = [
        line
        for line in inline_text.splitlines()
        if "call" in line and "@exact_lookup(" in line
    ]
    if hot_calls:
        raise ValueError("lookup call survived actual LLVM inlining")
    expected = (FIXTURE / "expected.bin").read_bytes()
    input_data = (FIXTURE / "input.bin").read_bytes()
    if len(expected) != 158700 or len(input_data) != 602112:
        raise ValueError("original complete byte contract changed")
    decoded = {}
    for arm, selector in (("control", 0), ("candidate", 1)):
        elf = Elf(CAPSULE / (arm + ".elf"))
        for symbol, payload in (
            ("captured_input", input_data),
            ("captured_input_saved", input_data),
            ("captured_expected", expected),
            ("observation_cells", table.data),
        ):
            if elf.bytes(symbol, len(payload)) != payload:
                raise ValueError("actual ELF immutable source bytes differ: " + symbol)
        if elf.bytes("selected", 1) != bytes([selector]):
            raise ValueError("actual ELF selected arm differs")
        actual = parse(
            (CAPSULE / (arm + "_spike.log")).read_text(),
            arm=selector,
            input_bytes=len(input_data),
            quant_elements=150528,
            padded_elements=len(expected),
            guard_bytes=16384,
            expected_digest=digest(expected),
        )
        if actual["rows"][0]["input"] != elf.symbol("captured_input")[3]:
            raise ValueError(
                "observed input pointer does not name the bound ELF allocation"
            )
        if actual["rows"][0]["output"] != elf.symbol("output")[3] + 4096:
            raise ValueError(
                "observed output pointer does not name the private guarded ELF allocation"
            )
        fresh = audit_elf(elf.data)
        if (
            fresh != json.loads((CAPSULE / (arm + "_audit.json")).read_text())
            or fresh["status"] != "pass"
        ):
            raise ValueError("fresh final executable no-FSM audit differs")
        decoded[arm] = actual
    if (
        decoded["control"]["rows"][0]["input"]
        != decoded["candidate"]["rows"][0]["input"]
        or decoded["control"]["rows"][0]["output"]
        != decoded["candidate"]["rows"][0]["output"]
    ):
        raise ValueError("matched-arm addresses differ")
    left = (CAPSULE / "control.elf").read_bytes()
    right = (CAPSULE / "candidate.elf").read_bytes()
    differences = [
        i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b
    ]
    if (
        differences != [capsule["selector_byte_offset"]]
        or left[differences[0]] != 0
        or right[differences[0]] != 1
    ):
        raise ValueError(
            "actual matched ELFs differ beyond the one readonly arm selector"
        )
    flat = dict(embedded_pins)
    for directory in (FIXTURE, CAPSULE, SEAL):
        for path in directory.rglob("*"):
            if path.is_file() and "parser_tests" not in path.parts:
                item = pin(path)
                flat[item["path"]] = item
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if (
            path
            and Path(path).is_file()
            and (
                Path(path).is_relative_to(CORE / "src")
                or Path(path).is_relative_to(ROOT / "mlir_oot")
            )
        ):
            item = pin(path)
            flat[item["path"]] = item
    explicit = [
        Path(__file__),
        ROOT / "mlir_oot/quant_prefix_capsule.py",
        ROOT / "tests/test_quant_prefix_capsule.py",
        ROOT / "support/gemmini_gsim/backend.py",
        ROOT / "support/gemmini_gsim/provider.yaml",
        mlir_runtime_c(),
        CORE / "merlin/runtime/c/benchmark_buffer.h",
        Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike"),
        Path(
            "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
        ),
        *[
            LLVM / name
            for name in (
                "clang",
                "mlir-opt",
                "llvm-link",
                "opt",
                "llvm-extract",
                "llvm-diff",
                "llvm-objcopy",
            )
        ],
        *[
            CORE / "out/artifacts/constant_rne_prefix" / name
            for name in (
                "tests_final.log",
                "structure_final.log",
                "no_target_final.log",
                "no_regex_final.log",
                "format_final.log",
                "default_identity.json",
            )
        ],
    ]
    for path in explicit:
        item = pin(path)
        flat[item["path"]] = item
    # Pin legacy harness inputs without changing their section placement.
    harness = Path(
        "/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests"
    )
    for name in (
        "riscv-tests/benchmarks/common/crt.S",
        "riscv-tests/benchmarks/common/syscalls.c",
        "riscv-tests/benchmarks/common/test.ld",
    ):
        path = harness / name
        if path.is_file():
            item = pin(path)
            flat[item["path"]] = item
    proxy_left = decoded["control"]["rows"][0]["instructions"]
    proxy_right = decoded["candidate"]["rows"][0]["instructions"]
    core_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CORE, text=True
    ).strip()
    record = {
        "schema": "closed_integer_quant_prefix_stock_admission_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "/root/reference_parity",
        "core": {
            "path": str(CORE),
            "head": core_head,
            "parent": "eb15a85ce550ec63c27530d56af66893fe629766",
        },
        "primary": pin(CAPSULE / "prelabel.json"),
        "source_proof_rederived": True,
        "original_source_extraction_byte_identical": True,
        "interval_table_rederived_byte_identical": True,
        "linked_original_scalar_llvm_diff": "PASS",
        "hot_lookup_calls": 0,
        "executor_body_unchanged": True,
        "delivery_difference": "compile-only signed-i8/binary32 ABI assertions; frozen timed source and ELFs untouched",
        "arms": capsule["arms"],
        "selector_only_byte_offset": differences[0],
        "strict_results": decoded,
        "native_source": {
            "complete_padded_outputs_each_arm": 158700,
            "original_quantized_values": 150528,
            **capsule["native_cells"],
        },
        "protocol": {
            "parser": "mlir_oot.quant_prefix_capsule.parse",
            "kwargs": {
                "input_bytes": 602112,
                "quant_elements": 150528,
                "padded_elements": 158700,
                "guard_bytes": 16384,
                "expected_digest": digest(expected),
            },
            **capsule["protocol"],
        },
        "numeric_contract": capsule["numeric_contract"],
        "partition": {
            "bits": 18,
            "requested_bytes": 524288,
            "certified_cells": table.certified_cells,
            "zero_cells": table.zero_cells,
            "saturated_cells": table.saturated_cells,
        },
        "scope": capsule["scope"],
        "control_scope": capsule["control_scope"],
        "spike_counter_semantics": "functional retired-instruction proxy; not Rocket/FireSim cycle timing",
        "retired_delta_percent": 100 * (proxy_right - proxy_left) / proxy_left,
        "gsim_status": "NOT_RUN",
        "firesim_status": "UNMEASURED",
        "whole_model_status": "NOT_BUILT",
        "model_status": capsule["model_status"],
        "performance_admission": "one complete matched stock cost decision; no predicted winner or whole-cycle credit",
        "retained_failures": [
            {
                "path": str(ROOT / "out/current_quant_prefix_capsule"),
                "reason": "unscoped runtime required unused libm/__errno; no execution qualification",
            },
            {
                "path": str(ROOT / "out/current_quant_prefix_capsule_v2"),
                "reason": "broad CRT/syscalls section-GC changed legacy TLS placement; refused",
            },
            {
                "path": str(ROOT / "out/current_quant_prefix_capsule_v3"),
                "reason": "numeric C completed but UART protocol corrupted after broad section-GC; explicitly not qualified",
            },
        ],
        "runtime_scope": "only fresh mlir_runtime.o uses function/data sections; legacy CRT/syscalls/allocator placement untouched; final all-executable-section audits pass",
        "token_usage_available": False,
        "file_pins": sorted(flat.values(), key=lambda row: row["path"]),
    }
    destination = SEAL / "qualification.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "qualification": pin(destination),
                "pins": len(flat),
                "core": core_head,
                "retired_delta_percent": record["retired_delta_percent"],
            }
        )
    )


if __name__ == "__main__":
    main()
