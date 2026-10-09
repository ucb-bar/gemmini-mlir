"""Normal source-cloned pointwise tile, exact current table and source fallback."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_word_cells import prepare_bounded_rne_word_cells
from merlin.llvmlower.late_quant_rne import _tokens
from merlin.llvmlower.scaled_integer_finite_llvm import (
    bind_finite_scale_helpers,
    emit_finite_scale_helper,
)
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    emit_immutable_bytes_llvm,
    emit_source_interval_i8_lookup,
    find_closed_scalar_i8_observers,
)
from merlin.llvmlower.source_expression_interval_llvm import (
    rewrite_source_interval_i8_lookup,
)

from mlir_oot.late_quant_rne import merlin_host_llvm_transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/artifacts/probes/paired-pointwise-consumer-20261007"
SOURCE = Path(
    "/scratch/agustin/tmp/merlin-sibling-producer-consumer-main-20261007/out/artifacts/probes/sibling-pointwise-tile-v6-20261007"
)
OLD = Path("/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006")
CONTEXT = (
    OLD / "out/artifacts/probes/source-continuation-contexts-v3-20261007/context_21"
)
ASSETS = OLD / "out/artifacts/probes/source-interval-calibration-v2-20261006"
LLVM = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rename(source, mapping):
    changes = [
        (t.start, t.end, mapping[t.text]) for t in _tokens(source) if t.text in mapping
    ]
    for start, stop, value in reversed(changes):
        source = source[:start] + value + source[stop:]
    return source


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "driver.py").write_bytes(Path(__file__).read_bytes())
    commands = []

    def run(argv):
        command = list(map(str, argv))
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    effects = IntervalEffectContract(True, True, True, True, True)
    observers, refused = find_closed_scalar_i8_observers(
        parse_mlir_text((SOURCE / "source.mlir").read_text()), effects=effects
    )
    assert len(observers) == 1 and not refused
    source = (SOURCE / "named.ll").read_text()
    assert "malloc" not in source and "memcpy" not in source
    table = build_source_interval_table(
        observers[0].expression,
        effects=effects,
        leading_bits=16,
        max_table_bytes=512 * 1024,
    )
    assert (
        table.data
        == (
            OLD
            / "out/artifacts/probes/source-interval-partition-analysis-20261006/table_16.bin"
        ).read_bytes()
    )
    changed, report = rewrite_source_interval_i8_lookup(
        source,
        proofs=observers,
        table=table,
        lookup_symbol="tile_lookup",
        effects=effects,
    )
    assert len(report["routes"]) == 2
    helpers = bind_finite_scale_helpers(
        source,
        routes=report["routes"],
        observers=observers,
        effects=effects,
        immutable_inputs=True,
        fresh_disjoint_output=True,
    )
    assert len(helpers) == 1 and len(helpers[0].scans) == 2
    (OUT / "source.ll").write_text(rename(source, {"@forward": "@tile_source"}))
    (OUT / "selected.ll").write_text(rename(changed, {"@forward": "@tile_rne"}))
    (OUT / "table.ll").write_text(
        emit_immutable_bytes_llvm(table.data, symbol="tile_table", alignment=64)
    )
    # Original source evaluator and original bounded quantizer are immutable;
    # symbol changes authenticate linking and never select numerical behavior.
    for old, new, names in (
        (
            "source_activation.ll",
            "activation.ll",
            {"@source_activation": "@tile_activation"},
        ),
        ("source_quantize.ll", "quantize.ll", {"@source_quantize": "@tile_quantize"}),
    ):
        (OUT / new).write_text(rename((ASSETS / old).read_text(), names))
    (OUT / "lookup.c").write_text(
        emit_source_interval_i8_lookup(
            table_name="tile_table",
            activation_name="tile_activation",
            quantizer_name="tile_quantize",
            lookup_name="tile_lookup",
            leading_bits=16,
            finite_inputs=helpers,
            finite_table=table,
            zero_observer_cells=(
                prepare_bounded_rne_word_cells(observers[0], effects=effects),
            ),
        )
    )
    (OUT / "scanner.c").write_text(
        emit_finite_scale_helper(
            helpers[0],
            wrapper_symbol="tile_scanned",
            finite_symbol="tile_rne",
            fallback_symbol="tile_source",
        )
    )
    target_flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O3",
        "-ffreestanding",
        "-fno-builtin",
        "-ffp-contract=off",
    ]
    for native in (True, False):
        case = OUT / ("native" if native else "target")
        case.mkdir()
        flags = ["-O3", "-fPIC", "-ffp-contract=off"] if native else target_flags
        if native:
            guard = "#include <fenv.h>\n#define BAD_MODE (fegetround()!=FE_TONEAREST)\n"
        else:
            guard = 'static unsigned current_mode(void){unsigned x;__asm__ volatile("csrr %0,frm":"=r"(x)::"memory");return x;}\n#define BAD_MODE current_mode()\n'
        guard += "extern void tile_source(void*,void*,void*,void*,void*);\nextern void tile_scanned(void*,void*,void*,void*,void*);\nvoid tile_current(void*a,void*sa,void*b,void*sb,void*out){if(BAD_MODE)tile_source(a,sa,b,sb,out);else tile_scanned(a,sa,b,sb,out);}\n"
        (case / "guard.c").write_text(guard)
        for stem, path in (
            ("lookup", OUT / "lookup.c"),
            ("scanner", OUT / "scanner.c"),
            ("guard", case / "guard.c"),
        ):
            run(
                [
                    LLVM / "clang",
                    *flags,
                    "-S",
                    "-emit-llvm",
                    path,
                    "-o",
                    case / (stem + ".ll"),
                ]
            )
        run(
            [
                LLVM / "llvm-link",
                "-S",
                *[
                    OUT / name
                    for name in (
                        "source.ll",
                        "selected.ll",
                        "table.ll",
                        "activation.ll",
                        "quantize.ll",
                    )
                ],
                *[case / (name + ".ll") for name in ("lookup", "scanner", "guard")],
                "-o",
                case / "linked.ll",
            ]
        )
        run(
            [
                LLVM / "opt",
                "-S",
                "-passes=always-inline",
                case / "linked.ll",
                "-o",
                case / "inlined.ll",
            ]
        )
        merlin_host_llvm_transform(LLVM, combine_clamp=True)(
            case / "inlined.ll", case / "late_rne"
        )
        selected = case / "late_rne" / ("model.native.ll" if native else "model.ll")
        run([LLVM / "clang", *flags, "-c", selected, "-o", case / "consumer.o"])
        if native:
            run(
                [
                    LLVM / "clang",
                    "-shared",
                    case / "consumer.o",
                    "-lm",
                    "-o",
                    case / "consumer.so",
                ]
            )
        (case / "consumer.dump").write_text(
            run([LLVM / "llvm-objdump", "-dr", case / "consumer.o"]).stdout
        )
    library = ctypes.CDLL(str(OUT / "native/consumer.so"))
    function = library.tile_current
    original = library.tile_source
    function.argtypes = original.argtypes = [ctypes.c_void_p] * 5
    modes = [0, 0x400, 0x800, 0xC00]
    fenv = ctypes.CDLL(None)
    g, u = [np.load(CONTEXT / name).reshape(8, 5632) for name in ("a.npy", "b.npy")]
    sg, su = [np.load(CONTEXT / name) for name in ("scale_a.npy", "scale_b.npy")]
    expected = np.load(CONTEXT / "expected.npy").reshape(8, 5632)
    events = []
    try:
        for mode in modes:
            assert fenv.fesetround(mode) == 0
            observed = np.empty_like(expected)
            for offset in range(0, 5632, 512):
                args = [
                    np.ascontiguousarray(g[:, offset : offset + 512]),
                    sg[offset : offset + 512].copy(),
                    np.ascontiguousarray(u[:, offset : offset + 512]),
                    su[offset : offset + 512].copy(),
                ]
                reference = np.full((8 * 512 + 128,), 73, np.int8)
                output = reference.copy()
                pins = [x.tobytes() for x in args]
                original(*[x.ctypes.data for x in args], reference.ctypes.data + 64)
                function(*[x.ctypes.data for x in args], output.ctypes.data + 64)
                assert np.array_equal(output, reference)
                assert np.all(output[:64] == 73) and np.all(output[-64:] == 73)
                assert [x.tobytes() for x in args] == pins
                observed[:, offset : offset + 512] = output[64:-64].reshape(8, 512)
            assert np.array_equal(observed, expected)
            events.append(
                {
                    "mode": mode,
                    "all45056_original_i8": True,
                    "input_and_guard_identity": True,
                }
            )
    finally:
        fenv.fesetround(0)
    pins = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            *SOURCE.rglob("*"),
            *OUT.rglob("*"),
            *CONTEXT.glob("*.npy"),
            ASSETS / "source_activation.ll",
            ASSETS / "source_quantize.ll",
        ]
        if p.is_file()
    }
    (OUT / "qualification.json").write_text(
        json.dumps(
            {
                "schema": "normal_source_pointwise_tile_native_v1",
                "status": "pass",
                "source_scalar_operation_order_precision_and_constants": "unchanged cloned typed source; ordinary upstream lowering",
                "maps": "compact owned i32 tile plus exact last-axis scale slices; output scattered by caller",
                "scope": "One original context21, all45056i8 words/four host modes; target/compound cost pending",
                "events": events,
                "source_routes": report,
                "commands": commands,
                "pins": pins,
                "token_usage_available": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("SOURCE_CLONED_TILE_NATIVE_PASS", flush=True)


if __name__ == "__main__":
    main()
