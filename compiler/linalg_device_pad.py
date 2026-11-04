"""Exact static tensor.pad through selected raw accelerator transfers."""
from dataclasses import asdict
from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseArrayBase,IntegerAttr,FloatAttr,i8
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_pad_transfer import SourcePadTransfer
from linalg_device_slice import resource_admission as slice_admission
from raw_storage_transfer import emit_raw_run,storage_receipt

ENABLED=True


def _metadata(op,properties=()):
    return (not op.successors and not set(op.properties)-set(properties)
        and all(k.startswith('prov.') for k in op.attributes))


def plan(op):
    if (op_name(op)!='tensor.pad' or len(op.operands)!=1 or len(op.results)!=1
        or len(op.regions)!=1 or len(op.regions[0].blocks)!=1
        or not _metadata(op,('static_low','static_high','operandSegmentSizes','nofold'))):
        return None
    source=op.operands[0];result=op.results[0]
    if any(not isinstance(v.type,TensorType) or not isinstance(v.type.encoding,NoneAttr) for v in (source,result)):
        return None
    dtype=str(source.type.element_type)
    if dtype not in ('i8','i32','f32') or str(result.type.element_type)!=dtype:
        return None
    rank=len(tensor_shape(source.type));bounds=[]
    for key in ('static_low','static_high'):
        attr=op.properties.get(key)
        if not isinstance(attr,DenseArrayBase) or str(attr.elt_type)!='i64':return None
        bounds.append(tuple(attr.iter_values()))
    segment=op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment,DenseArrayBase) or str(segment.elt_type)!='i32' or tuple(segment.iter_values())!=(1,0,0)):
        return None
    block=op.regions[0].block;body=tuple(block.ops)
    if (len(block.args)!=rank or any(str(a.type)!='index' or a.uses for a in block.args)
        or not body or op_name(body[-1])!='tensor.yield' or len(body[-1].operands)!=1
        or body[-1].regions or not _metadata(body[-1])):
        return None
    scalar=body[-1].operands[0]
    if str(scalar.type)!=dtype:return None
    if len(body)==2:
        constant=body[0]
        if (op_name(constant)!='arith.constant' or constant.operands or len(constant.results)!=1
            or scalar is not constant.results[0] or constant.regions or not _metadata(constant,('value',))
            or not isinstance(constant.properties.get('value'),(IntegerAttr,FloatAttr))):
            return None
    elif len(body)!=1 or scalar in block.args or getattr(scalar.owner,'parent',None) is block:
        return None
    try:
        schedule=SourcePadTransfer(tuple(tensor_shape(source.type)),*bounds,dtype,16)
        if tuple(tensor_shape(result.type))!=schedule.output_shape or resource_admission(schedule) is None:return None
        return schedule
    except (TypeError,ValueError,IndexError):
        return None


def resource_admission(schedule,facts=None):
    admissions=[slice_admission(s,facts) for s in (schedule.interior,*schedule.borders)]
    return None if any(a is None for a in admissions) else admissions


def try_lower(host,op):
    if not ENABLED:return False
    schedule=plan(op)
    if schedule is None:return False
    source=op.operands[0];result=op.results[0];scalar=op.regions[0].block.last_op.operands[0]
    local=len(tuple(op.regions[0].block.ops))==2
    if (not result.uses or any(v not in host.ptrs for v in (source,result))
        or result in getattr(host,'aliasable_results',()) or result in getattr(host,'storage_aliases',{})
        or any(v is not result and p is host.ptrs[result] for v,p in host.ptrs.items())
        or not local and scalar not in host.vals):
        return False
    enc=host.enc;fn=host.fp.fb
    if local and schedule.dtype=='f32':
        # Retain the original typed FloatAttr, bypassing Python-float constant
        # memoization (which identifies +0 and -0). No scalar conversion.
        value=fn.add(llvm.ConstantOp(scalar.owner.properties['value'],scalar.type)).results[0]
    else:
        value=host._const(scalar.owner) if local else host.get(scalar)
    if not hasattr(value,'type'):return False
    enc.fence()
    if schedule.borders:
        allocation=llvm.AllocaOp(fn.const(schedule.itemsize),i8,alignment=64)
        terminator=fn.entry.last_op
        if terminator is not None and terminator.has_trait(llvm.IsTerminator):
            fn.entry.insert_op_before(allocation,terminator)
        else:fn.entry.add_op(allocation)
        pointer=allocation.results[0]
        if schedule.dtype=='f32':host.fp.store(value,pointer,fn.const(0),schedule.dtype)
        else:fn.store_i64(value,pointer,fn.const(0),schedule.dtype)
    else:pointer=None
    schedule.emit(enc.index_builder,
        lambda run:emit_raw_run(enc,pointer,host.ptrs[result],run,schedule.dtype),
        lambda run:emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,schedule.dtype))
    enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result);host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_source_pad_transfer',None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_pad_transfer.v1',plan=asdict(schedule),
            resources=resource_admission(schedule),storage_view=storage_receipt(schedule.dtype),
            scalar_payload_operations=0,scalar_storage_bytes=schedule.itemsize if schedule.borders else 0,
            observers='complete original source and scalar unchanged; fresh complete result',
            partition='disjoint first-outside-axis border slabs plus source interior; every element written once',
            completion='entry, each raw load and store, final full drain; scalar slot live through completion',timing='UNKNOWN')))
    return True
