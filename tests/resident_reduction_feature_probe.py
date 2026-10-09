"""Join a complete typed command trace to a qualified function PC census.

Requested payload, logical arithmetic and nominal padded geometry are separate
observations. No physical memory traffic, issue price or overlap is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from test_resident_conv_command_loops import executed_commands
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from mlir_oot.executed_features import census
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_features(module):
    """Resolve operands through Merlin; accept this explicit i8 resident ABI."""
    fn = module.body.block.first_op
    if not isinstance(fn, llvm.FuncOp) or len(fn.body.blocks.first.args) != 3:
        raise ValueError("three-pointer resident convolution ABI required")
    stream = executed_commands(module)
    classes, sizes = Counter(), Counter()
    a_rows, b_shapes, stores = Counter(), Counter(), Counter()
    configs = []
    steps = trace_static_function(
        fn,
        [StaticPointer(i, StaticInt(0, 64)) for i in range(3)],
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    )
    for step, (name, encoded, pointers) in zip(steps, stream, strict=True):
        op = step.operation
        if name != op.name:
            raise ValueError("trace operation identity changed")
        if encoded is not None:
            classes[F.FUNCT_NAMES[encoded[0]]] += 1
        if isinstance(op, G.ConfigExOp):
            configs.append(
                {
                    key: op.a(key, default)
                    for key, default in (
                        ("dataflow", isa.WEIGHT_STATIONARY),
                        ("a_stride", 1),
                        ("c_stride", 1),
                        ("a_transpose", 0),
                        ("b_transpose", 0),
                    )
                }
            )
        elif isinstance(op, G.ConfigLdOp):
            if op.a("scale", 1.0) != 1.0 or op.a("shrunk", 0):
                raise ValueError(
                    "byte payload requires the declared unscaled i8 load contract"
                )
        elif isinstance(op, G.MvinOp):
            if len(pointers) != 1 or op.a("local") & isa.ACC_ADDR_BIT:
                raise ValueError("scratchpad i8 load with one source pointer required")
            payload = op.a("rows") * op.a("cols")
            zero = pointers[0].base is None and pointers[0].offset.value == 0
            sizes[
                "zero_fill_payload_bytes" if zero else "requested_dma_load_bytes"
            ] += payload
            sizes["zero_fill_commands" if zero else "requested_dma_load_commands"] += 1
        elif isinstance(op, G.MvoutOp):
            if len(pointers) != 1 or not op.a("local") & isa.ACC_ADDR_BIT:
                raise ValueError(
                    "accumulator store with one destination pointer required"
                )
            element_bytes = 4 if op.a("local") & isa.ACC_FULL_ROW_BIT else 1
            sizes["requested_dma_store_bytes"] += (
                op.a("rows") * op.a("cols") * element_bytes
            )
            stores[(op.a("rows"), op.a("cols"), element_bytes)] += 1
        elif isinstance(op, G.PreloadOp):
            # Encoder is the field authority; both static and dynamic row operands
            # were resolved in executed_commands before this interpretation.
            bd = encoded[1] & ((1 << isa.ADDR_LEN) - 1)
            reused = bd == isa.GARBAGE_ADDR
            sizes[
                "preload_garbage_reuse" if reused else "preload_real_stationary_b"
            ] += 1
            if not reused:
                b_shapes[(op.a("bd_rows"), op.a("bd_cols"))] += 1
            c = encoded[2] & ((1 << isa.ADDR_LEN) - 1)
            sizes[
                "accumulator_add_preloads"
                if c & isa.ACC_ACCUMULATE_BIT
                else "accumulator_overwrite_preloads"
            ] += 1
        elif isinstance(op, G.ComputeOp):
            a_rows[op.a("a_rows")] += 1
            sizes["array_work_padded_mac_slots"] += F.DIM**3
            sizes["array_work_padded_rows"] += F.DIM
            sizes["moving_operand_logical_values"] += op.a("a_rows") * op.a("a_cols")
        elif isinstance(op, G.FenceOp):
            sizes["fence_commands"] += 1
    return {
        "command_classes": dict(sorted(classes.items())),
        "features": dict(sorted(sizes.items())),
        "compute_a_rows_histogram": {str(k): v for k, v in sorted(a_rows.items())},
        "real_b_preload_geometry_histogram": {
            str(k): v for k, v in sorted(b_shapes.items())
        },
        "store_geometry_histogram": {str(k): v for k, v in sorted(stores.items())},
        "execute_configs": configs,
        "command_count_including_fences": len(stream),
        "complete_encoded_sequence_repr_sha256": hashlib.sha256(
            repr(stream).encode()
        ).hexdigest(),
    }


def export(target_module, elf, histogram, *, symbol, scope_id):
    context = Context(allow_unregistered=True)
    for dialect in (Builtin, llvm.LLVM, G.GEMMINI):
        context.load_dialect(dialect)
    module = Parser(context, Path(target_module).read_text()).parse_module()
    module.verify()
    if module.body.block.first_op.sym_name.data != symbol:
        raise ValueError("target function and selected machine-code symbol differ")
    binding = {
        "elf_sha256": digest(elf),
        "histogram_sha256": digest(histogram),
        "scope_id": scope_id,
        "engine": {"kind": "spike_functional"},
    }
    observed = census(
        elf, histogram, symbols=[symbol], scope_id=scope_id, execution_binding=binding
    )
    source = command_features(module)
    actual = {k: v for k, v in observed["features"]["primitive_commands"].items() if v}
    if actual != source["command_classes"]:
        raise ValueError(
            "actual executed primitive classes differ from the complete typed trace"
        )
    observed["source_command_trace"] = source
    for name, value in source["features"].items():
        observed["features"][name] = value
        observed["feature_status"]["/features/" + name] = {
            "status": "complete_source_cfg_trace_with_actual_primitive_class_conservation",
            "target_module_sha256": digest(target_module),
            "numeric_or_operand_machine_equivalence": "Requires the separate source/object/capsule closure receipt",
            "physical_traffic_or_overlap": "UNKNOWN",
        }
    observed["features"]["array_work"] = source["features"][
        "array_work_padded_mac_slots"
    ]
    observed["features"]["requested_dma_bytes"] = (
        source["features"]["requested_dma_load_bytes"]
        + source["features"]["requested_dma_store_bytes"]
    )
    for name in ("array_work", "requested_dma_bytes"):
        observed["feature_status"]["/features/" + name] = {
            "status": "complete_source_cfg_trace_with_actual_primitive_class_conservation",
            "target_module_sha256": digest(target_module),
            "physical_traffic_or_overlap": "UNKNOWN",
        }
    observed["unknown_features"].update(
        {
            "cpu_issue_width": "UNKNOWN: no issue or resource prices supplied",
            "cpu_array_or_dma_overlap": "UNKNOWN: functional operands/histogram are not a hardware timeline",
            "cpu_integer_address_dependency_distances": "UNKNOWN: current shared producer covers FP register RAW only",
        }
    )
    observed["artifacts"]["target_module"] = {
        "path": str(Path(target_module).resolve()),
        "sha256": digest(target_module),
    }
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-module", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--histogram", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(
        args.target_module,
        args.elf,
        args.histogram,
        symbol=args.symbol,
        scope_id=args.scope_id,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESIDENT_REDUCTION_FEATURES_PASS", result["features"]["executed_instructions"]
    )


if __name__ == "__main__":
    main()
