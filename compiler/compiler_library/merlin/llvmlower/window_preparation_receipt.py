"""Checked, static placement evidence for original-input window transport.

This describes emitted host work; it grants no accelerator-placement exemption
or target capability. Address arithmetic is host work even though payload bytes
are never arithmetically transformed. Consumer completion remains the caller's
checked packed-raw schedule obligation.
"""

from dataclasses import asdict
import hashlib
import json

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8
from xdsl.ir import BlockArgument

from merlin.compile.scheduling.spatial_packed_raw import PackedRawConvolution
from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel


def _pointer_root(value):
    while isinstance(value.owner, llvm.GEPOp):
        value = value.owner.operands[0]
    return value


def emit_checked_window_preparation(builder, adapter, source, destination,
                                    fields, *, transport="window_bytes"):
    """Emit and check one static preparation site from its original typed plan.

    A site may execute many times inside counted loops. Neither this receipt nor
    its operation counts are an execution trace or a cycle measurement.
    """
    if type(adapter) is not PackedRawConvolution:
        raise ValueError("window preparation requires the original packed-raw adapter")
    adapter.__post_init__()
    window = adapter.original.window
    if (fields["adapter_identity"] != adapter.identity or fields["window"] != window
            or fields["workspace_bytes"] != adapter.workspace_bytes
            or not 0 < fields["rows"] <= adapter.original.row_tile
            or fields["panel_bytes"] != fields["rows"] * adapter.reduction):
        raise ValueError("window preparation differs from its checked source or storage")
    # This proof covers explicit byte-copy loops. The separate unit-window
    # memcpy/memset implementation requires its own intrinsic-effects adapter.
    if window.kernel == (1, 1):
        raise ValueError("window preparation receipt requires explicit window-copy effects")
    allocation = destination.owner
    count = allocation.operands[0].owner if isinstance(allocation, llvm.AllocaOp) else None
    extent = getattr(count, "value", None)
    if (not isinstance(allocation, llvm.AllocaOp)
            or allocation.parent_block() is not builder.entry
            or allocation.elem_type != i8
            or not isinstance(extent, IntegerAttr)
            or extent.value.data < adapter.workspace_bytes
            or _pointer_root(source) is destination):
        raise ValueError("window preparation requires disjoint bounded entry storage")

    before = {op for block in builder.region.blocks for op in block.ops}
    emitter = emit_nhwc_window_panel
    if transport == "contiguous_taps":
        from merlin.llvmlower.nhwc_preparation_runs import emit_nhwc_window_runs
        emitter = emit_nhwc_window_runs
    elif transport != "window_bytes":
        raise ValueError("unknown checked original-input preparation transport")
    transport = emitter(builder, window, source, destination,
        fields["output_row"], fields["rows"], source_destination_disjoint=True,
        maximum_panel_bytes=fields["workspace_bytes"])
    emitted = [op for block in builder.region.blocks for op in block.ops if op not in before]
    emitted_set = set(emitted)
    allowed = {"llvm.mlir.constant", "llvm.getelementptr", "llvm.add", "llvm.sub",
               "llvm.mul", "llvm.udiv", "llvm.urem", "llvm.and", "llvm.icmp",
               "llvm.select", "llvm.br", "llvm.cond_br", "llvm.load", "llvm.store"}
    loads = stores = zeros = 0
    for op in emitted:
        if op.name not in allowed:
            raise ValueError("unproved effect in host window preparation: " + op.name)
        if isinstance(op, llvm.LoadOp):
            loads += 1
            uses = tuple(op.results[0].uses)
            if (_pointer_root(op.operands[0]) is not _pointer_root(source)
                    or op.results[0].type != i8
                    or len(uses) != 1
                    or any(use.operation not in emitted_set
                           or not isinstance(use.operation, llvm.StoreOp)
                           or use.operation.operands[0] is not op.results[0]
                           for use in uses)):
                raise ValueError("window preparation transforms payload or reads another storage")
        elif isinstance(op, llvm.StoreOp):
            stores += 1
            value = op.operands[0]
            origin = value.owner
            zero = getattr(origin, "value", None)
            is_zero = isinstance(zero, IntegerAttr) and zero.value.data == 0
            if (_pointer_root(op.operands[1]) is not destination or value.type != i8
                    or not (is_zero or isinstance(origin, llvm.LoadOp) and origin in emitted_set)):
                raise ValueError("window preparation writes nontransport payload or another storage")
            zeros += int(is_zero)
    if not loads or stores != loads + zeros:
        raise ValueError("window preparation does not establish byte-copy and zero-pad effects")
    original = asdict(window)
    source_root = _pointer_root(source)
    return dict(schema="merlin.host_window_preparation.v1", lane="host",
        reason="original_input_window_materialization", evidence_scope="static_emission_site",
        original_source_identity=adapter.original.identity, adapter_identity=adapter.identity,
        window=original, window_sha256=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        input_shape=list(window.input_shape), input_layout="nhwc", input_access="read_only",
        source_argument_index=(source_root.index if isinstance(source_root, BlockArgument)
                               and source_root.owner is builder.entry else None),
        output_row_mapping="original logical output row SSA plus counted panel-row index",
        destination=dict(ownership="private", lifetime="invocation", storage="entry_alloca",
            allocation_bytes=extent.value.data, declared_bytes=adapter.workspace_bytes,
            written_bytes=fields["panel_bytes"], panel_shape=[fields["rows"], adapter.reduction]),
        semantics="original kh,kw,channel byte copies with guarded source reads and zero padding",
        numeric_payload_arithmetic=False, host_address_and_control_arithmetic=True,
        effects_check=dict(load_origin="original_input", store_destination="private_panel",
            stored_payload="unchanged_i8_load_or_i8_zero", loads=loads, stores=stores, zero_stores=zeros),
        storage_checker="PackedRawConvolution.__post_init__/prepare/emit_prepared",
        completion_obligation="publish host stores before device load; finish raw tile before panel overwrite",
        target_preparation_admission="UNKNOWN; placement receipt grants no capability",
        transport=transport)
