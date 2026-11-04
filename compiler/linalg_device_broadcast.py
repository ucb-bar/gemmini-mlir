"""Exact nonzero/runtime scalar splat/fill through original scalar storage."""
from dataclasses import asdict
from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase, i8
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_broadcast_transfer import SourceBroadcastTransfer
from linalg_device_zero import _metadata_only, _literal_zero
from linalg_device_layout import resource_admission as layout_admission
from raw_storage_transfer import emit_raw_run, storage_receipt

ENABLED=True


def plan(op):
    name=op_name(op)
    if (name not in ('tensor.splat','linalg.fill') or len(op.results)!=1
        or not _metadata_only(op,('operandSegmentSizes',) if name=='linalg.fill' else ())):
        return None
    result=op.results[0]
    if not isinstance(result.type,TensorType) or not isinstance(result.type.encoding,NoneAttr):
        return None
    dtype=str(result.type.element_type)
    if dtype not in ('i8','i32','f32') or not op.operands or str(op.operands[0].type)!=dtype:
        return None
    if name=='tensor.splat':
        if len(op.operands)!=1 or op.regions:
            return None
    else:
        if len(op.operands)!=2 or op.operands[1].type!=result.type or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:
            return None
        segment=op.properties.get('operandSegmentSizes')
        if segment is not None and (not isinstance(segment,DenseArrayBase) or str(segment.elt_type)!='i32' or tuple(segment.iter_values())!=(1,1)):
            return None
        block=op.regions[0].block;body=tuple(block.ops)
        if (len(block.args)!=2 or any(str(v.type)!=dtype for v in block.args)
            or block.args[1].uses or len(body)!=1 or op_name(body[0])!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],) or body[0].regions or not _metadata_only(body[0])):
            return None
    # Keep the selected NULL-source zero implementation as the incumbent.
    if _literal_zero(op.operands[0],dtype):
        return None
    try:
        schedule=SourceBroadcastTransfer(tuple(tensor_shape(result.type)),dtype,16)
        if layout_admission(schedule.movement) is None:
            return None
        return schedule
    except (TypeError,ValueError):
        return None


def try_lower(host,op):
    if not ENABLED:
        return False
    schedule=plan(op)
    if schedule is None:
        return False
    result=op.results[0];scalar=op.operands[0]
    if (not result.uses or result not in host.ptrs or scalar not in host.vals
        or result in getattr(host,'aliasable_results',()) or result in getattr(host,'storage_aliases',{})
        or any(v is not result and p is host.ptrs[result] for v,p in host.ptrs.items())):
        return False
    if op_name(op)=='linalg.fill' and (op.operands[1] not in host.ptrs or host.ptrs[op.operands[1]] is host.ptrs[result]):
        return False
    value=host.get(scalar)
    if not hasattr(value,'type'):
        return False
    enc=host.enc;fn=host.fp.fb
    # Existing host preparation pattern (backend.py) and original scalar store
    # APIs: one bounded private slot; no new input ABI or payload computation.
    allocation=llvm.AllocaOp(fn.const(schedule.itemsize),i8,alignment=64)
    terminator=fn.entry.last_op
    if terminator is not None and terminator.has_trait(llvm.IsTerminator):
        fn.entry.insert_op_before(allocation,terminator)
    else:
        fn.entry.add_op(allocation)
    pointer=allocation.results[0]
    enc.fence()
    if schedule.dtype in ('i8','i32'):
        fn.store_i64(value,pointer,fn.const(0),schedule.dtype)
    else:
        host.fp.store(value,pointer,fn.const(0),schedule.dtype)
    schedule.emit(enc.index_builder,lambda run:emit_raw_run(enc,pointer,host.ptrs[result],run,schedule.dtype))
    enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_scalar_broadcast',None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_broadcast_transfer.v1',plan=asdict(schedule),
            resources=layout_admission(schedule.movement),storage_view=storage_receipt(schedule.dtype),
            preparation='one original scalar stored through existing host scalar store API in private entry alloca',
            scalar_storage_bytes=schedule.itemsize,scalar_payload_operations=0,
            observers='original scalar and fill seed unchanged; fresh complete result',
            completion='entry, each load, each store, final full drains; stack slot live through final drain',timing='UNKNOWN')))
    return True
