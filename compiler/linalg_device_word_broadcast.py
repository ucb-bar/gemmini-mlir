"""Exact i64 splat/fill through two original adjacent raw32 storage words."""
from dataclasses import asdict
from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase, i8
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_word_broadcast_transfer import SourceWordBroadcastTransfer
from linalg_device_zero import _metadata_only
from linalg_device_layout import resource_admission as layout_admission
from raw_storage_transfer import emit_raw_run, storage_receipt, raw_storage_admission
from encoding import FACTS
from pathlib import Path
from hashlib import sha256

ENABLED=True


def resource_admission(schedule):
    # i32's ordinary route alone is not a general bit-pattern proof. Reuse the
    # selected structural raw32 law already required by exact f32 movement.
    raw=raw_storage_admission('f32',FACTS)
    resources=layout_admission(schedule.movement)
    if raw is None or resources is None:
        return None
    return dict(resources,raw_word_preservation=raw,words_per_source_element=2,
        source_word_offsets_bytes=[0,4],output_element_stride_bytes=8,
        word_order='same increasing byte-address order on both sides; endian-independent',
        native_scalar_store='FnBuilder.store_i64: original i64 SSA store, no cast',
        payload_arithmetic=False)


def refusal_reason(op):
    if op.operands and str(op.operands[0].type)=='f64':
        return 'native-width f64 scalar owner unavailable: FpBuilder load/store supports f32/bf16; f64 constants materialize as f32'
    return None


def plan(op):
    name=op_name(op)
    if (name not in ('tensor.splat','linalg.fill') or len(op.results)!=1
        or not _metadata_only(op,('operandSegmentSizes',) if name=='linalg.fill' else ())):
        return None
    result=op.results[0]
    if not isinstance(result.type,TensorType) or not isinstance(result.type.encoding,NoneAttr):
        return None
    dtype=str(result.type.element_type)
    if dtype != 'i64' or not op.operands or str(op.operands[0].type)!=dtype:
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
    # Existing NULL-source initializer has no i64 route. Literal zero uses
    # this same original-width scalar store, with no new zero-source law.
    try:
        schedule=SourceWordBroadcastTransfer(tuple(tensor_shape(result.type)),dtype,16)
        if resource_admission(schedule) is None:
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
    # Reject any scalar representation that already lost original i64 bits.
    if not hasattr(value,'type') or str(value.type)!='i64':
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
    fn.store_i64(value,pointer,fn.const(0),'i64')
    schedule.emit(enc.index_builder,lambda run:emit_raw_run(enc,pointer,host.ptrs[result],run,'i32'))
    enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_i64_word_broadcast',None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_word_broadcast_transfer.v1',plan=asdict(schedule),
            resources=resource_admission(schedule),storage_view=storage_receipt('i32'),
            source_owners={p:sha256((Path(__file__).parent/p).read_bytes()).hexdigest() for p in (
                'linalg_device_word_broadcast.py','encoding.py','raw_storage_transfer.py','data/raw_storage_contract.json',
                'compiler_library/merlin/compile/scheduling/source_word_broadcast_transfer.py',
                'compiler_library/merlin/compile/scheduling/source_broadcast_transfer.py',
                'compiler_library/merlin/compile/scheduling/layout_transfer.py',
                'compiler_library/merlin/llvmlower/host_builder.py')},
            preparation='one original scalar stored through existing host scalar store API in private entry alloca',
            scalar_storage_bytes=schedule.itemsize,scalar_payload_operations=0,
            observers='original scalar and fill seed unchanged; fresh complete result',
            completion='entry, each load, each store, final full drains; stack slot live through final drain',timing='UNKNOWN')))
    return True
