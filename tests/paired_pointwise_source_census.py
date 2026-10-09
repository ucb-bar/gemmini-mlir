"""Typed sibling closure plus explicit original-provider arithmetic witnesses.

Shape equality alone supplies no producer theorem. This experiment authenticates
the original from-zero signed-i8/i32 source contractions, complete writer ABI,
actual selected kernel and live fresh wrappers before supplying that theorem to
the generic analysis. Provider symbols bind implementations; they choose no
optimization policy. Every eligible consumer derives from maps, uses and source
semantics. Target issue width is an explicit resource choice.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    find_closed_scalar_i8_observers,
)
from merlin.llvmlower.streamed_pointwise_pair import (
    analyze_pointwise_siblings,
    outline_pointwise_tile,
    validate_pointwise_siblings,
)
from paired_gemm_source_probe import BUILD
from paired_pointwise_prepare import ROOT, SOURCE, sha
from xdsl.dialects import func

from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_gemm_phases import GoldenGemmPhases

TYPED = Path(
    "/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir"
)


def main():
    out = ROOT / "out/artifacts/probes/paired-pointwise-source-closure-v2-20261007"
    out.mkdir(parents=True, exist_ok=False)
    effects = IntervalEffectContract(True, True, True, True, True)
    catalog_path = BUILD / "device_catalog/device_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog_source = Path(catalog["source_snapshot"])
    assert sha(catalog_source) == catalog["source_sha256"]
    assert (
        sha(BUILD / "device_catalog/kernel.o")
        == catalog["compilation"]["object_sha256"]
    )
    abi = catalog["abi"]
    assert (
        abi["dtypes"] == ["i8", "i8", "i32"]
        and abi["pointee_layout"] == "dense_row_major"
    )
    assert abi["writer_effects"] == {
        "schema": "complete_output_writer_v1",
        "fully_written_arguments": [2],
        "retains_arguments": False,
        "frees_arguments": False,
    }
    original = parse_mlir_text(catalog_source.read_text())
    original_ops = tuple(original.walk())
    # Every source binding must be an actual exact signed from-zero body;
    # labels/shape/type metadata never replace semantic recognition.
    proofs = {}
    for binding in catalog["bindings"]:
        operation = original_ops[binding["source_operation_ordinal"]]
        dims = match_integer_gemm(operation)
        assert dims is not None
        assert [
            str(v.type) for v in (*operation.operands[:2], operation.results[0])
        ] == binding["tensor_types"]
        bound = dims.k * 128 * 128
        assert bound < 2**31
        kernel = next(k for k in catalog["kernels"] if k["symbol"] == binding["symbol"])
        schedule = kernel["schedule"]
        assert (
            schedule["output_dtype"] == "i32"
            and schedule["scale"] == 1
            and not schedule["bias"]
            and not schedule["relu"]
        )
        key = tuple(binding["tensor_types"])
        proofs.setdefault(key, []).append(
            {
                "ordinal": binding["source_operation_ordinal"],
                "kernel": binding["symbol"],
                "dimensions": asdict(dims),
                "prefix_absolute_bound": bound,
                "source_body": str(operation),
                "initialization": str(operation.operands[2].owner),
            }
        )
    module = parse_mlir_text(TYPED.read_text())
    before = str(module)
    functions = {
        f.sym_name.data: f for f in module.walk() if isinstance(f, func.FuncOp)
    }
    observers, refused = find_closed_scalar_i8_observers(module, effects=effects)
    assert len(observers) == 22 and not refused
    events = []
    first_clone = None
    for observer in observers:
        generic = observer.cut.owner
        while generic.name != "linalg.generic":
            generic = generic.parent_op()
        calls = []
        wrapper_witnesses = {}
        for value in generic.inputs:
            if getattr(value.owner, "name", None) != "tensor.expand_shape":
                continue
            call = value.owner.operands[0].owner
            assert isinstance(call, func.CallOp)
            wrapper = functions[call.callee.root_reference.data]
            operations = tuple(wrapper.body.block.ops)
            assert [op.name for op in operations] == [
                "bufferization.to_buffer",
                "bufferization.to_buffer",
                "memref.alloc",
                "func.call",
                "bufferization.to_tensor",
                "func.return",
            ]
            assert tuple(operations[0].operands) == (wrapper.body.block.args[0],)
            assert tuple(operations[1].operands) == (wrapper.body.block.args[1],)
            assert tuple(operations[3].operands) == (
                operations[0].results[0],
                operations[1].results[0],
                operations[2].results[0],
            )
            assert tuple(operations[4].operands) == (operations[2].results[0],)
            assert tuple(operations[5].operands) == (operations[4].results[0],)
            writer = functions[operations[3].callee.root_reference.data]
            assert not writer.body.blocks
            tensor_types = tuple(
                str(v.type) for v in (*call.operands[:2], call.results[0])
            )
            assert tensor_types in proofs
            assert list(writer.function_type.outputs) == []
            assert [str(t) for t in writer.function_type.inputs] == [
                s.replace("tensor<", "memref<", 1) for s in tensor_types
            ]
            wrapper_witnesses[call] = (
                wrapper,
                str(wrapper),
                writer,
                str(writer),
                tensor_types,
            )
            calls.append(call)

        def validate_producer(call):
            witness = wrapper_witnesses.get(call)
            if witness is None:
                return False
            wrapper, body, writer, declaration, types = witness
            return (
                str(wrapper) == body
                and str(writer) == declaration
                and tuple(str(v.type) for v in (*call.operands[:2], call.results[0]))
                == types
            )

        plan = analyze_pointwise_siblings(
            generic,
            calls,
            shared_argument=0,
            effects=effects,
            validate_producer=validate_producer,
        )
        validate_pointwise_siblings(plan, validate_producer=validate_producer)
        provider = GoldenGemmPhases(
            Shape(
                plan.rows,
                plan.columns,
                calls[0].operands[0].type.get_shape()[1],
                "i32",
                bm=1,
                bn=32,
                cache_a=True,
                wide_b=True,
                prefetch_b=True,
            ),
            columns=512,
        )
        clone = outline_pointwise_tile(
            plan, columns=512, symbol="forward", validate_producer=validate_producer
        )
        if observer.quant_factor_bits == 1133693052:
            assert first_clone is None
            first_clone = str(clone)
            # Normal emitted source includes one terminal newline. The first
            # audit attempt refused exact bytes solely for that serializer
            # terminator; retained source operations/SSA/types are identical.
            assert first_clone + "\n" == (SOURCE / "source.mlir").read_text()
            (out / "source.mlir").write_text(first_clone + "\n")
        events.append(
            {
                "quant_factor_bits": observer.quant_factor_bits,
                "source_expression_sha256": observer.expression.canonical_sha256,
                "rows": plan.rows,
                "columns": plan.columns,
                "same_input_SSA": True,
                "sole_integer_consumer": True,
                "source_calls": [str(c) for c in calls],
                "provider_resources": asdict(provider.resources),
                "source_shape_proofs": proofs[
                    tuple(
                        str(v.type)
                        for v in (*calls[0].operands[:2], calls[0].results[0])
                    )
                ],
                "opaque_arithmetic_permission": "Explicit authenticated original signed/fromzero GEMM producer plus fullwrite/read-only/noescape ABI; never inferred from call name",
                "scalar_tile_source_sha256": __import__("hashlib")
                .sha256(str(clone).encode())
                .hexdigest(),
            }
        )
    assert first_clone is not None and str(module) == before
    pins = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            TYPED,
            catalog_path,
            catalog_source,
            BUILD / "device_catalog/kernel.o",
            SOURCE / "source.mlir",
        )
    }
    (out / "qualification.json").write_text(
        json.dumps(
            {
                "schema": "typed_pointwise_sibling_all_source_closure_v1",
                "status": "pass",
                "consumer_count": 22,
                "producer_count": 44,
                "numerical_scope": "Source-exact signed integer product sum (all prefix bounds<2^31), unchanged closed scalar source arithmetic, explicit effect permission. Source closure alone supplies no physical alias/async legality or performance theorem. Actual selected new provider is separately strict-qualified.",
                "events": events,
                "pins": pins,
                "initial_attempt": "Unqualified terminal-newline byte refusal; exact source re-emission succeeds after the existing serializer terminator, with no operation/SSA/type/attribute change.",
                "hardware_cycles": "UNKNOWN",
                "whole_route": "Not installed; pending complete cost and ordinary ownership binding",
                "token_usage_available": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("ALL22_SOURCE_SIBLING_CLOSURE_PASS")


if __name__ == "__main__":
    main()
