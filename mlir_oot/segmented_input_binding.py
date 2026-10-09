"""Explicit source-proved borrowed matrix inputs for resident-A consumers.

Eligibility uses typed view geometry, the current compiled schedule and resource
proofs. Names identify catalog artifacts; they never select a scheduling rule.
The original objects remain available under their original symbols. Accepted
calls use distinct kernels, descriptor adapters and native scalar oracles.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

from merlin.llvmlower.segmented_input_acceptance import (
    SegmentedInputContract,
    rewrite_segmented_inputs,
)
from merlin.llvmlower.segmented_matrix_view import prove_segmented_matrix
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import CallOp

from .captured_requant_bundle import scalar_oracle
from .captured_residual_bundle import compile_adapter
from .frontend.parse import parse_module
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape
from .no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linker_path(llvm_bin):
    selected = Path(llvm_bin) / "ld.lld"
    if selected.is_file():
        return selected
    found = shutil.which("ld.lld")
    if found is None:
        raise ValueError("ld.lld required for accepted consumer composition")
    return Path(found)


def _identifier(value):
    if not value or not value.isascii() or not value.isidentifier():
        raise ValueError("C identifier required for the accepted consumer")


def adapter_source(shape, symbol, kernel, view, owner_shape, *, cached_a_output_blocks=False, prefetch_b_rows=None):
    """Ranked descriptor ABI for an unchanged dense owner and fresh output."""
    _identifier(symbol)
    _identifier(kernel)
    GoldenGemm(shape, input_view=view, cached_a_output_blocks=cached_a_output_blocks,
               prefetch_b_rows=prefetch_b_rows)
    owner_shape = tuple(owner_shape)
    if not owner_shape or any(type(v) is not int or v <= 0 for v in owner_shape):
        raise ValueError("positive static dense owner shape required")
    if prod(owner_shape) != view.source_elements:
        raise ValueError("owner descriptor extent differs from the accepted map")
    rank = len(owner_shape)
    a_type = symbol + "_input_descriptor"
    c_type = "int32_t" if shape.output_dtype == "i32" else "int8_t"

    def checks(name, dimensions):
        return " || ".join(
            [f"!{name}->aligned", f"{name}->offset<0"]
            + [
                f"{name}->sizes[{i}]!={v} || {name}->strides[{i}]!={prod(dimensions[i + 1 :])}"
                for i, v in enumerate(dimensions)
            ]
        )

    return f"""#include <stdint.h>
#if UINTPTR_MAX != UINT64_MAX
#error 64-bit ranked descriptor ABI required
#endif
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {{void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];}} memref2;
typedef struct {{void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];}} memref4;
#endif
typedef struct {{void *allocated,*aligned; intptr_t offset,sizes[{rank}],strides[{rank}];}} {a_type};
extern void {kernel}(int8_t*,int8_t*,{c_type}*);
void _mlir_ciface_{symbol}(memref2*r,{a_type}*a,memref2*b,memref2*c) {{
 if({checks("a", owner_shape)} || {checks("b", (shape.k, shape.n))} || {checks("c", (shape.m, shape.n))})__builtin_trap();
 {kernel}((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,({c_type}*)c->aligned+c->offset);
 *r=*c;
}}
"""


@dataclass(frozen=True)
class Binding:
    contract: SegmentedInputContract
    generator: GoldenGemm
    owner_shape: tuple
    original: dict


def derive(module, routes):
    """Derive views while closing the unchanged control's target IR identity."""
    by_symbol = {entry["symbol"]: entry for entry in routes}
    if len(by_symbol) != len(routes):
        raise ValueError("duplicate source catalog symbol")
    bindings, refused = {}, []
    for call in module.walk():
        if (
            not isinstance(call, CallOp)
            or call.callee.root_reference.data not in by_symbol
        ):
            continue
        route = by_symbol[call.callee.root_reference.data]
        if (
            route["direct_conv"]
            or len(call.arguments) != 3
            or route.get("integer_readout")
        ):
            continue
        try:
            proof = prove_segmented_matrix(call.arguments[0])
        except ValueError:
            continue
        view = proof.address
        if (
            view.row_stride == view.cols
            and view.segment_stride == view.segment_rows * view.cols
        ):
            continue
        try:
            shape = Shape(**route["schedule"])
            if shape.bias:
                raise ValueError(
                    "segmented consumer has no separately bound bias argument"
                )
            slots = route.get("dense_b_slot_policy_decision", {})
            rows = slots.get("prefetch_b_rows") if slots.get("applied") else None
            loads = route.get("resident_a_load_decision", {})
            width = loads.get("resident_a_load_tiles", 1) if loads.get("applied") else 1
            tail = route.get("dense_stationary_tail_decision", {}).get("applied", False)
            if type(tail) is not bool:
                raise ValueError("dense tail decision requires a boolean witness")
            control = GoldenGemm(
                shape,
                prefetch_b_rows=rows,
                resident_a_load_tiles=width,
                stationary_b_tail_before_last_full=tail,
                cached_a_output_blocks=route.get("dense_resident_output_block_decision", {}).get("applied", False),
            )
            emitted = control.build()
            emitted.body.block.first_op.properties["sym_name"] = StringAttr(
                route["kernel"]
            )
            actual = hashlib.sha256((str(emitted) + "\n").encode()).hexdigest()
            if actual != route["compilation"]["target_ir_sha256"]:
                raise ValueError(
                    "reconstructed control differs from its compiled target IR"
                )
            generator = control.with_emission_options(input_view=view)
            if route.get("numeric_contract", {}).get("max_output_lsb_error") != 0:
                raise ValueError(
                    "accepted input requires the existing exact numeric contract"
                )
        except ValueError as reason:
            refused.append({"source_symbol": route["symbol"], "reason": str(reason)})
            continue
        contract = SegmentedInputContract(
            route["symbol"], 0, route["symbol"] + "__segmented_a", view
        )
        binding = Binding(
            contract, generator, tuple(proof.owner.type.get_shape()), route
        )
        previous = bindings.get(contract.symbol)
        if previous and (
            previous.contract != contract or previous.owner_shape != binding.owner_shape
        ):
            raise ValueError("source calls disagree on the accepted owner/map")
        bindings[contract.symbol] = binding
    return tuple(bindings.values()), refused


def compile_bindings(bindings, llvm_bin, directory):
    """Compile each accepted primitive, descriptor adapter and exact oracle."""
    if not bindings:
        raise ValueError("at least one accepted source-bound consumer required")
    directory, llvm_bin = Path(directory), Path(llvm_bin)
    directory.mkdir(parents=True, exist_ok=False)
    objects, native, records = [], ["#include <stdint.h>"], []
    for binding in bindings:
        symbol = binding.contract.accepted_symbol
        kernel = symbol + "_kernel"
        work = directory / symbol
        device = binding.generator.build()
        device.body.block.first_op.properties["sym_name"] = StringAttr(kernel)
        compilation = compile_module(device, llvm_bin, work)
        adapter = adapter_source(
            binding.generator.shape,
            symbol,
            kernel,
            binding.contract.address,
            binding.owner_shape,
            cached_a_output_blocks=binding.generator.cached_a_output_blocks,
            prefetch_b_rows=binding.generator.prefetch_b_rows,
        )
        (work / "adapter.c").write_text(adapter)
        adapter_compilation = compile_adapter(
            work / "adapter.c", work / "adapter.o", llvm_bin
        )
        objects.extend((work / "kernel.o", work / "adapter.o"))
        native.append(
            adapter
            + scalar_oracle(
                binding.generator.shape,
                kernel,
                False,
                input_view=binding.contract.address,
                cached_a_output_blocks=binding.generator.cached_a_output_blocks,
                prefetch_b_rows=binding.generator.prefetch_b_rows,
            )
        )
        records.append(
            {
                "source_symbol": binding.contract.symbol,
                "accepted_symbol": symbol,
                "accepted_kernel": kernel,
                "owner_shape": binding.owner_shape,
                "address": binding.contract.address.__dict__,
                "original_target_ir_sha256": binding.original["compilation"][
                    "target_ir_sha256"
                ],
                "preserved_numeric_proof": binding.original["proof"],
                "compilation": compilation,
                "adapter_compilation": adapter_compilation,
            }
        )
    oracle = directory / "native_oracle.c"
    oracle.write_text("\n".join(native) + "\n")
    output = directory / "segmented_inputs.o"
    linker = linker_path(llvm_bin)
    command = [str(linker), "-r", *map(str, objects), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    audit = audit_elf(output.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("accepted input object contains forbidden FSM commands")
    receipt = {
        "schema": "gemmini_segmented_input_bindings_v1",
        "routes": records,
        "object_sha256": sha(output),
        "native_oracle_sha256": sha(oracle),
        "nofsm_audit": audit,
        "linker_argv": command,
        "linker_sha256": sha(linker),
    }
    (directory / "bindings.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return output, oracle, receipt


def merlin_callbacks(llvm_bin, routes, fresh_writers, base_callbacks, *, enabled=False):
    """Opt into actual selected source/objects before fresh-writer conversion.

    Returns (prepare, build, writer_contracts). The final callable returns the
    current explicit FreshTensorWriterContracts, which the host ABI stage must
    consume. Absent selection returns the original callbacks without I/O.
    """
    if type(enabled) is not bool:
        raise ValueError("segmented input binding requires explicit boolean selection")
    fresh_writers = tuple(fresh_writers)
    if not enabled:
        return (*base_callbacks, lambda: fresh_writers)
    base_prepare, base_build = base_callbacks
    state = {}

    def prepare(source, work):
        from merlin.xdsl_dialects._common import text

        original = Path(base_prepare(source, work))
        module = parse_module(original.read_text())
        bindings, refused = derive(module, routes)
        if not bindings:
            raise ValueError(
                "no source-bound resident consumer accepts a segmented input"
            )
        contracts = tuple(binding.contract for binding in bindings)
        report = rewrite_segmented_inputs(
            module, contracts, fresh_writers=fresh_writers
        )
        directory = Path(work) / "segmented_inputs"
        obj, oracle, receipt = compile_bindings(bindings, llvm_bin, directory)
        selected = directory / "prepared.mlir"
        selected.write_text(text(module, generic=True))
        parse_module(selected.read_text()).verify()
        changes = {contract.symbol: contract.accepted_symbol for contract in contracts}
        writers = tuple(
            replace(
                writer,
                symbol=changes[writer.symbol],
                borrowed_symbol=changes[writer.symbol] + "__borrowed_write",
            )
            if writer.symbol in changes
            else writer
            for writer in fresh_writers
        )
        state.update(
            original=original,
            original_sha256=sha(original),
            selected=selected,
            selected_sha256=sha(selected),
            obj=obj,
            oracle=oracle,
            receipt=receipt,
            report=report,
            refused=refused,
            writers=writers,
        )
        return selected

    def build(source, work):
        if (
            not state
            or sha(source) != state["selected_sha256"]
            or sha(state["original"]) != state["original_sha256"]
        ):
            raise ValueError("accepted source or original catalog identity changed")
        if (
            sha(state["obj"]) != state["receipt"]["object_sha256"]
            or sha(state["oracle"]) != state["receipt"]["native_oracle_sha256"]
        ):
            raise ValueError("accepted implementation object or native oracle changed")
        path, original_object = base_build(state["original"], work)
        manifest = json.loads(Path(path).read_text())
        snapshot = Path(work) / "catalog_source.mlir"
        original_snapshot = Path(work) / "segmented_base_catalog_source.mlir"
        if snapshot.is_file():
            shutil.copyfile(snapshot, original_snapshot)
        snapshot.write_bytes(Path(source).read_bytes())
        combined = Path(work) / "segmented_mixed.o"
        command = [
            str(linker_path(llvm_bin)),
            "-r",
            str(original_object),
            str(state["obj"]),
            "-o",
            str(combined),
        ]
        subprocess.run(command, check=True, capture_output=True)
        audit = audit_elf(combined.read_bytes())
        if audit["status"] != "pass":
            raise ValueError("accepted mixed catalog contains forbidden FSM commands")
        manifest["source_sha256"] = state["selected_sha256"]
        manifest["compilation"] = {
            "schema": "gemmini_segmented_mixed_compile_v1",
            "object_sha256": sha(combined),
            "object_nofsm_status": "pass",
            "original_compilation": manifest["compilation"],
            "segmented_bindings": state["receipt"],
            "linker_argv": command,
        }
        manifest["segmented_input_acceptance"] = {
            "source_sha256": state["original_sha256"],
            "selected_sha256": state["selected_sha256"],
            "source_rewrites": state["report"],
            "refused": state["refused"],
            "preserved_original_object_sha256": sha(original_object),
            "schedule_selection_controls_emission": True,
            "whole_model_accuracy": "UNKNOWN until checked",
            "selected_source_snapshot_sha256": sha(snapshot),
            "original_source_snapshot_sha256": sha(original_snapshot)
            if original_snapshot.is_file()
            else None,
        }
        manifest["native_oracle_sources"].append(str(state["oracle"]))
        selected = {
            record["source_symbol"]: record for record in state["receipt"]["routes"]
        }
        rewritten = []
        for route in manifest["fused_requantizations"]:
            record = selected.get(route["symbol"])
            if record is None:
                rewritten.append(route)
                continue
            updated = dict(route)
            updated.update(
                source_symbol=route["symbol"],
                symbol=record["accepted_symbol"],
                kernel=record["accepted_kernel"],
                compilation=record["compilation"],
                adapter_compilation=record["adapter_compilation"],
                segmented_input={
                    "owner_shape": record["owner_shape"],
                    "address": record["address"],
                },
                schedule_kind=route["schedule_kind"] + ",segmented_input",
            )
            rewritten.append(updated)
        if len(selected) != sum("segmented_input" in route for route in rewritten):
            raise ValueError(
                "accepted implementation is missing from the actual catalog"
            )
        manifest["fused_requantizations"] = rewritten
        Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
        return path, combined

    def writers():
        if not state:
            raise ValueError("segmented source preparation has not run")
        return state["writers"]

    return prepare, build, writers
