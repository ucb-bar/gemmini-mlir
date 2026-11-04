"""Explicit external writer binding for a completely retained source group.

This module checks source, typed ABI and supplied proof identities. The caller
owns the mathematical certificate and complete source fallback permission. Neither
closure nor an implementation hash proves numerical equivalence. There is no
implicit routing or target ABI choice; installation is always explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from merlin.llvmlower.fresh_tensor_writer import (
    FreshTensorWriterContract,
    PrivateWorkspaceContract,
    rewrite_fresh_tensor_writers,
    validate_private_workspaces,
)
from merlin.llvmlower.ordered_bf16_group_binding import (
    _context_digest,
    verify_group_module_coverage,
)
from merlin.xdsl_dialects._common import text


@dataclass(frozen=True)
class ClosedGroupWriterContract:
    """Separately supplied endpoint/effect proof and physical writer identity.

    The full source semantic fingerprint includes all scalar properties, casts,
    operand dependencies and types. Provenance is excluded; source instances
    remain independently bound by their retained function/context witnesses.
    A borrowed ranked-C writer must fully write its destination, preserve all
    inputs, never retain/release buffers, and provide source fallback on every
    input outside its certificate. Allocation alignment is supplied by its owner.
    """

    source_semantic_sha256: str
    provider_symbol: str
    provider_implementation_sha256: str
    numerical_witness_sha256: str
    numerical_policy: str
    effect_witness_sha256: str
    allocation_alignment: int
    fully_writes_result: bool
    preserves_inputs: bool
    borrowed_buffers: bool
    complete_source_fallback: bool
    fenv_policy: str
    qualification_scope: str
    private_workspaces: tuple[PrivateWorkspaceContract, ...] = ()
    source_fallback_symbol: str | None = None


def source_function_semantic_sha256(function):
    """Fingerprint the complete typed function independent of provenance/name.

    Registered operations and every numerical attribute/property are retained.
    Only prov.* metadata, source-instance tags and public C-interface visibility
    are removed. Input/output dimensions stay part of this specialization.
    """
    from xdsl.dialects.builtin import StringAttr

    canonical = function.clone()
    canonical.properties["sym_name"] = StringAttr("closed_group_source")
    canonical.properties["sym_visibility"] = StringAttr("private")
    for operation in canonical.walk():
        for name in tuple(operation.attributes):
            if (
                name.startswith("prov.")
                or name.startswith("merlin.source_group_")
                or operation is canonical
                and name == "llvm.emit_c_interface"
            ):
                del operation.attributes[name]
    return hashlib.sha256(text(canonical).encode()).hexdigest()


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def install_closed_group_writers(module, source_receipt, contracts):
    """Bind selected complete source bodies to explicitly proved fresh writers.

    Validate the entire selection before source mutation. Unselected groups keep
    their source implementation, including equal-shape groups. Physical writers
    share an implementation only with equal typed ABI and complete contract.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import (
        ArrayAttr,
        DictionaryAttr,
        NoneAttr,
        StringAttr,
        TensorType,
        UnitAttr,
    )
    from xdsl.ir import Block, Region

    verify_group_module_coverage(module, source_receipt)
    records = {record["symbol"]: record for record in source_receipt["records"]}
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    plans, providers, fallbacks = [], {}, {}
    reserved = {
        symbol.data
        for operation in module.body.block.ops
        if (symbol := operation.properties.get("sym_name", operation.attributes.get("sym_name"))) is not None
        and isinstance(symbol, StringAttr)
    }
    for symbol, contract in contracts.items():
        if symbol not in records or not isinstance(contract, ClosedGroupWriterContract):
            raise ValueError("writer requires a retained source group and explicit contract")
        function = functions[symbol]
        validate_private_workspaces(contract.private_workspaces)
        if contract.source_semantic_sha256 != source_function_semantic_sha256(function):
            raise ValueError("complete typed source semantics differ from writer certificate")
        if (
            any(
                not _pin(value)
                for value in (
                    contract.source_semantic_sha256,
                    contract.provider_implementation_sha256,
                    contract.numerical_witness_sha256,
                    contract.effect_witness_sha256,
                )
            )
            or not contract.provider_symbol
            or not contract.numerical_policy
            or not contract.qualification_scope
            or contract.fenv_policy != "rne_returned_values"
            or any(
                value is not True
                for value in (
                    contract.fully_writes_result,
                    contract.preserves_inputs,
                    contract.borrowed_buffers,
                    contract.complete_source_fallback,
                )
            )
        ):
            raise ValueError("complete endpoint, fallback, effect and RNE witnesses required")
        alignment = contract.allocation_alignment
        if type(alignment) is not int or alignment <= 0 or alignment >= (1 << 63) or alignment & (alignment - 1):
            raise ValueError("explicit positive power-of-two allocation alignment required")
        inputs, outputs = tuple(function.function_type.inputs), tuple(function.function_type.outputs)
        if len(outputs) != 1 or any(
            not isinstance(typ, TensorType) or typ.encoding != NoneAttr() or any(dim <= 0 for dim in typ.get_shape())
            for typ in (*inputs, *outputs)
        ):
            raise ValueError("one static unencoded tensor endpoint and static tensor inputs required")
        provider = contract.provider_symbol
        identity = (contract, inputs, outputs)
        if provider in providers and providers[provider] != identity:
            raise ValueError("shared physical writer has unequal source, ABI or proof identity")
        if provider not in providers:
            if provider in reserved or provider + "_borrowed" in reserved:
                raise ValueError("writer symbol collides with a source or planned declaration")
            reserved.update((provider, provider + "_borrowed"))
        providers[provider] = identity
        fallback = contract.source_fallback_symbol
        if fallback is not None:
            if not isinstance(fallback, str) or not fallback:
                raise ValueError("retained source fallback requires an explicit nonempty symbol")
            fallback_identity = (contract.source_semantic_sha256, inputs, outputs, _context_digest(function))
            if fallback in fallbacks:
                if fallbacks[fallback][0] != fallback_identity:
                    raise ValueError("shared source fallback has unequal semantics, ABI or context")
                fallbacks[fallback][2].append(symbol)
            else:
                if fallback in reserved or "_mlir_ciface_" + fallback in reserved:
                    raise ValueError("source fallback symbol collides with a source or planned declaration")
                reserved.update((fallback, "_mlir_ciface_" + fallback))
                fallbacks[fallback] = (fallback_identity, function, [symbol])
        plans.append((symbol, function, contract, inputs, outputs))

    fallback_reports = []
    for fallback, (identity, function, source_symbols) in fallbacks.items():
        retained = function.clone()
        retained.properties["sym_name"] = StringAttr(fallback)
        # An external provider calls this compiler-emitted helper. Exporting it
        # retains its body through ordinary symbol DCE; the model entry ABI stays
        # unchanged. Normal buffer-results lowering appends destination params.
        retained.properties.pop("sym_visibility", None)
        retained.attributes["llvm.emit_c_interface"] = UnitAttr()
        for name in tuple(retained.attributes):
            if name.startswith("merlin.source_group_"):
                del retained.attributes[name]
        module.body.block.add_op(retained)
        fallback_reports.append(
            dict(
                symbol=fallback,
                c_interface_symbol="_mlir_ciface_" + fallback,
                source_symbols=source_symbols,
                source_semantic_sha256=identity[0],
                input_types=[str(typ) for typ in identity[1]],
                output_types=[str(typ) for typ in identity[2]],
                source_context_sha256=identity[3],
                function_sha256=hashlib.sha256(text(retained).encode()).hexdigest(),
                retained_semantic_sha256=source_function_semantic_sha256(retained),
                source_body="complete_original_typed_function",
                external_retention="exported_c_interface",
                redispatch_source_binding=False,
            )
        )
    reports = []
    for symbol, function, contract, inputs, outputs in plans:
        body = Block(arg_types=inputs)
        empty = tensor.EmptyOp([], outputs[0])
        call = func.CallOp(contract.provider_symbol, [*body.args, empty.tensor], outputs)
        body.add_ops([empty, call, func.ReturnOp(*call.results)])
        function.detach_region(0)
        function.add_region(Region(body))
        function.attributes["merlin.closed_group_writer_witness"] = StringAttr(contract.numerical_witness_sha256)
        reports.append(
            dict(
                source_symbol=symbol,
                binding_sha256=records[symbol]["binding_sha256"],
                source_function_sha256=records[symbol]["function_sha256"],
                writer_function_sha256=hashlib.sha256(text(function).encode()).hexdigest(),
                writer_context_sha256=_context_digest(function),
                contract=_contract_record(contract),
            )
        )
    fresh = []
    for provider, (contract, inputs, outputs) in providers.items():
        args = (*inputs, outputs[0])
        access = ("read",) * len(inputs) + ("write",)
        declaration = func.FuncOp(
            provider,
            (args, outputs),
            Region(),
            visibility="private",
            arg_attrs=ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(mode)}) for mode in access]),
        )
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        module.body.block.add_op(declaration)
        fresh.append(
            FreshTensorWriterContract(
                provider,
                len(inputs),
                (len(inputs),),
                provider + "_borrowed",
                contract.allocation_alignment,
                private_workspaces=contract.private_workspaces,
            )
        )
    writer_report = rewrite_fresh_tensor_writers(module, fresh) if fresh else []
    module.verify()
    result = dict(
        schema="closed_group_writer_installation_v1",
        records=reports,
        source_groups=len(reports),
        physical_writers=len(providers),
        fresh_writer_report=writer_report,
        scope="Explicit supplied proof/ABI binding; no numerical or target qualification inferred",
    )
    if fallback_reports:
        result["retained_source_fallbacks"] = fallback_reports
    return result


def writer_contract_sha256(contract):
    return hashlib.sha256(json.dumps(_contract_record(contract), sort_keys=True).encode()).hexdigest()


def _contract_record(contract):
    record = asdict(contract)
    if not contract.private_workspaces:
        del record["private_workspaces"]
    if contract.source_fallback_symbol is None:
        del record["source_fallback_symbol"]
    return record
