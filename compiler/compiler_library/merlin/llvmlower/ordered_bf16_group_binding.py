"""Normal preparation and source coverage for closed ordered BF16 groups.

Group policy follows live source arithmetic and types. Ordinals and hashes bind
instances only. This source-exact control does not claim that encoded device
products or a complete numerical certificate have been installed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from xdsl.dialects.builtin import StringAttr, bf16

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_fma_group_outline import outline_ordered_fma_group
from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups, validate_group_source
from merlin.llvmlower.ordered_fma_rewrite import _match
from merlin.xdsl_dialects._common import text


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _context_digest(operation):
    contexts = []
    ancestor = operation.parent_op()
    while ancestor is not None:
        contexts.append(
            dict(
                operation=ancestor.name,
                attributes=[(name, str(value)) for name, value in sorted(ancestor.attributes.items())],
                properties=[(name, str(value)) for name, value in sorted(ancestor.properties.items())],
            )
        )
        ancestor = ancestor.parent_op()
    return _digest(contexts)


def describe_source_group(module, group, source_sha256):
    """Inventory one complete source obligation before any provider rewrite."""
    validate_group_source(group)
    if not group.closed_bf16_endpoints or any(op.inputs[0].type.element_type != bf16 for op in group.contractions):
        raise ValueError("closed ordered BF16 source contractions required")
    positions = {operation: index for index, operation in enumerate(module.walk())}
    contractions = []
    for operation in group.contractions:
        transposed = _match(operation)
        if transposed is None:
            raise ValueError("ordered source contraction proof changed")
        lhs, rhs, result = operation.inputs[0].type, operation.inputs[1].type, operation.results[0].type
        contractions.append(
            dict(
                source_operation_ordinal=positions[operation],
                lhs_type=str(lhs),
                rhs_type=str(rhs),
                output_type=str(result),
                batch_dimensions=list(result.get_shape()[:-2]),
                m=result.get_shape()[-2],
                n=result.get_shape()[-1],
                k=lhs.get_shape()[-1],
                rhs_transposed=transposed,
                accumulation="positive_zero_increasing_k_f32_fma",
            )
        )
    return dict(
        source_sha256=source_sha256,
        source_operation_ordinals=[positions[operation] for operation in group.operations],
        source_operation_classes=dict(Counter(operation.name for operation in group.operations)),
        source_operations=len(group.operations),
        contractions=contractions,
        endpoint_types=[str(value.type) for value in group.bf16_outputs],
        source_semantics="complete_original_typed_DAG_with_all_casts_and_f32_intermediates",
        closure_only=True,
        numeric_certificate_installed=False,
    )


class SourceExactGroupPreparation:
    """An explicit DeviceRouting.prepared_transform source-exact control.

    Every eligible group is selected from its arithmetic/type/closure proof.
    Normal upstream compilation still executes the outlined source functions.
    Guarded external writers remain disabled until a separately qualified provider
    supplies its complete endpoint, source fallback, effect and ABI proofs.
    """

    def __init__(self, *, expected_source_sha256=None):
        self.expected_source_sha256 = expected_source_sha256
        self.receipt = None

    def __call__(self, source_path, workdir):
        source_path, workdir = Path(source_path), Path(workdir)
        source_bytes = source_path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if self.expected_source_sha256 is not None and source_sha != self.expected_source_sha256:
            raise ValueError("prepared source changed before group binding")
        module = parse_mlir_text(source_bytes.decode())
        selected = []
        for group in analyze_ordered_fma_groups(module):
            if not group.closed_bf16_endpoints:
                continue
            if any(op.inputs[0].type.element_type != bf16 for op in group.contractions):
                continue
            selected.append((group, describe_source_group(module, group, source_sha)))
        if not selected:
            raise ValueError("no closed ordered BF16 group in the current prepared source")
        # Derive every source selection before mutating the original module.
        records = []
        for index, (group, record) in enumerate(selected):
            binding = _digest(record)
            symbol = "ordered_bf16_source_" + binding[:16]
            outlined = outline_ordered_fma_group(module, group, symbol)
            for operation in (outlined.function, outlined.call):
                operation.attributes["merlin.source_group_binding_sha256"] = StringAttr(binding)
                operation.attributes["merlin.source_group_source_sha256"] = StringAttr(source_sha)
            records.append(
                record
                | dict(
                    binding_data=record,
                    binding_sha256=binding,
                    symbol=symbol,
                    function_sha256=hashlib.sha256(text(outlined.function).encode()).hexdigest(),
                    function_context_sha256=_context_digest(outlined.function),
                    call_context_sha256=_context_digest(outlined.call),
                    actual_call_count=1,
                    input_types=[str(value.type) for value in outlined.inputs],
                    internalized_constant_tensors=len(outlined.internalized_constants),
                    omitted_unread_initializers=len(outlined.omitted_unread_initializers),
                    implementation_kind="ordinary_source_cpu",
                    target_product_provider_installed=False,
                )
            )
        module.verify()
        workdir.mkdir(parents=True, exist_ok=True)
        prepared = workdir / "source_groups.mlir"
        prepared.write_text(text(module))
        self.receipt = dict(
            schema="ordered_bf16_group_source_preparation_v1",
            original_prepared_path=str(source_path.resolve()),
            original_prepared_sha256=source_sha,
            selected_path=str(prepared.resolve()),
            selected_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
            source_groups=len(records),
            source_contractions=sum(len(record["contractions"]) for record in records),
            records=records,
            scope="Actual ordinary source calls; no guarded numeric or target product provider enabled",
        )
        (workdir / "source_group_calls.json").write_text(json.dumps(self.receipt, indent=2) + "\n")
        verify_group_call_coverage(prepared, self.receipt)
        return prepared


def verify_group_call_coverage(source_path, receipt):
    """Recheck actual call/declaration types and source identities, never shapes alone."""
    module = parse_mlir_text(Path(source_path).read_text())
    return verify_group_module_coverage(module, receipt)


def verify_group_module_coverage(module, receipt):
    """Validate the same retained witnesses on an already parsed live module."""
    module.verify()
    expected = {record["symbol"]: record for record in receipt["records"]}
    if len(expected) != len(receipt["records"]) or receipt["source_groups"] != len(expected):
        raise ValueError("source group distinct binding count changed")
    declarations = {
        operation.sym_name.data: operation
        for operation in module.walk()
        if operation.name == "func.func" and operation.sym_name.data in expected
    }
    calls = [
        operation
        for operation in module.walk()
        if operation.name == "func.call" and operation.callee.root_reference.data in expected
    ]
    if set(declarations) != set(expected) or Counter(call.callee.root_reference.data for call in calls) != Counter(
        {symbol: 1 for symbol in expected}
    ):
        raise ValueError("source group actual call/declaration coverage changed")
    for symbol, record in expected.items():
        if _digest(record["binding_data"]) != record["binding_sha256"]:
            raise ValueError("source group binding record changed")
        if hashlib.sha256(text(declarations[symbol]).encode()).hexdigest() != record["function_sha256"]:
            raise ValueError("source group retained function changed")
    for operation in [*declarations.values(), *calls]:
        symbol = operation.sym_name.data if operation.name == "func.func" else operation.callee.root_reference.data
        record = expected[symbol]
        if (
            getattr(operation.attributes.get("merlin.source_group_binding_sha256"), "data", None)
            != record["binding_sha256"]
            or getattr(operation.attributes.get("merlin.source_group_source_sha256"), "data", None)
            != record["source_sha256"]
        ):
            raise ValueError("source group exact binding changed")
        context = "function_context_sha256" if operation.name == "func.func" else "call_context_sha256"
        if _context_digest(operation) != record[context]:
            raise ValueError("source group enclosing numeric context changed")
        inputs = (
            operation.function_type.inputs
            if operation.name == "func.func"
            else [value.type for value in operation.arguments]
        )
        outputs = (
            operation.function_type.outputs
            if operation.name == "func.func"
            else [value.type for value in operation.results]
        )
        if [str(typ) for typ in inputs] != record["input_types"] or [str(typ) for typ in outputs] != record[
            "endpoint_types"
        ]:
            raise ValueError("source group typed ABI changed")
    return dict(
        groups=len(expected),
        actual_calls=len(calls),
        source_contractions=sum(len(record["contractions"]) for record in expected.values()),
        target_product_provider_installed=False,
    )
