"""Optional exact integer and positive-f32-zero initialization via NULL-source MVIN."""
from dataclasses import asdict
from math import prod
import hashlib
from pathlib import Path
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase, IntegerAttr, FloatAttr
from xdsl.ir import Operation
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_zero_transfer import SourceZeroTransfer
from linalg_device_layout import resource_admission as layout_resource_admission

ENABLED = True
F32_ENABLED = True


def _metadata_only(op, properties=()):
    return (not op.successors and all(k.startswith('prov.') for k in op.attributes)
        and not set(op.properties)-set(properties))


def _literal_zero(value,dtype):
    owner=value.owner
    if (not isinstance(owner,Operation) or op_name(owner) != 'arith.constant'
        or len(owner.results) != 1 or owner.results[0] is not value
        or owner.operands or owner.regions or not _metadata_only(owner,('value',))
        or str(value.type) != dtype):
        return False
    literal=owner.properties.get('value')
    if dtype=='f32':
        from raw_zero_admission import positive_zero
        return isinstance(literal,FloatAttr) and str(literal.type)=='f32' and positive_zero(literal.value.data)
    return isinstance(literal,IntegerAttr) and str(literal.type)==dtype and literal.value.data==0


def resource_admission(schedule,facts=None):
    """Bind the same selected raw transfer encoding used by existing zero paths.

    dense_pool.emit_dense_pool clears ACC with NULL-source MVIN, while window
    schedules use the same Encoder.ptr_zero/load pair. Native development tests
    preserve these actual instructions for both shrink settings and raw stores.
    Geometry facts alone are not a proof of NULL's behavioral semantics.
    """
    resources=layout_resource_admission(schedule.movement,facts)
    if resources is None:
        return None
    if schedule.dtype=='f32':
        from raw_zero_admission import zero_admission
        from encoding import FACTS
        law=zero_admission(FACTS if facts is None else facts)
        if law is None:
            return None
        resources['raw_zero_law']=law
    owner=Path(__file__).with_name('dense_pool.py')
    resources.update(null_source_address=0,accumulator_local_base=1 << 31,
        accumulate=False,load_stride_bytes=0,load_scale=1,store_scale=1,
        shrunk=schedule.dtype=='i8',readout='raw_full_i32' if schedule.dtype in ('i32','f32') else 'exact_zero_i8',
        zero_mechanism='existing Encoder.ptr_zero/load as used by dense_pool ACC zero padding',
        existing_zero_owner_sha256=hashlib.sha256(owner.read_bytes()).hexdigest(),
        authority='selected same-Encoder native zero-source mechanism; development execution checked separately')
    return resources


def plan(op):
    name=op_name(op)
    if (name not in ('tensor.splat','linalg.fill') or len(op.results)!=1
        or not _metadata_only(op,('operandSegmentSizes',) if name=='linalg.fill' else ())):
        return None
    result=op.results[0]
    if not isinstance(result.type,TensorType) or not isinstance(result.type.encoding,NoneAttr):
        return None
    dtype=str(result.type.element_type)
    if dtype not in ('i8','i32','f32') or dtype=='f32' and not F32_ENABLED:
        return None
    if name=='tensor.splat':
        if len(op.operands)!=1 or op.regions:
            return None
    else:
        if len(op.operands)!=2 or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:
            return None
        seed=op.operands[1]
        if seed.type!=result.type:
            return None
        segment=op.properties.get('operandSegmentSizes')
        if segment is not None and (not isinstance(segment,DenseArrayBase)
            or str(segment.elt_type)!='i32' or tuple(segment.iter_values())!=(1,1)):
            return None
        block=op.regions[0].block;body=tuple(block.ops)
        if (len(block.args)!=2 or any(str(a.type)!=dtype for a in block.args)
            or block.args[1].uses or len(body)!=1 or op_name(body[0])!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],) or body[0].regions
            or not _metadata_only(body[0])):
            return None
    if not _literal_zero(op.operands[0],dtype):
        return None
    try:
        factory=SourceZeroTransfer
        if dtype=='f32':
            from raw_zero_admission import Raw32ZeroTransfer
            factory=Raw32ZeroTransfer
        schedule=factory(tuple(tensor_shape(result.type)),dtype,16)
    except (ValueError,TypeError):
        return None
    if resource_admission(schedule) is None:
        return None
    return schedule


def try_lower(host,op):
    if not ENABLED:
        return False
    schedule=plan(op)
    if schedule is None:
        return False
    result=op.results[0]
    # Virtual zero groups have already omitted their pointers before dispatch.
    # A dead source value does not need a materialized zero allocation either.
    if (not result.uses or result not in host.ptrs
        or result in getattr(host,'aliasable_results',())
        or result in getattr(host,'storage_aliases',{})):
        return False
    if any(value is not result and ptr is host.ptrs[result] for value,ptr in host.ptrs.items()):
        return False
    if op_name(op)=='linalg.fill':
        seed=op.operands[1]
        if seed not in host.ptrs or host.ptrs[seed] is host.ptrs[result]:
            return False
    enc=host.enc
    from merlin.compile.scheduling.raw_storage_view import raw_storage_view
    storage=raw_storage_view(schedule.dtype)
    from raw_transfer_admission import raw_pair_admission
    ordering=dict(raw_pair_admission())
    ordering['source_run_count']=(prod(schedule.shape)+schedule.tile-1)//schedule.tile
    ordering['selected']=ordering['admitted'] and ordering['source_run_count']>2
    ordering['selection_reason']=('replace more per-run drains than added entry/retirement drains'
        if ordering['selected'] else ordering.get('reason') or 'at most two runs; retain original drains')
    if ordering['selected']:
        enc.fence()
    def emit(run):
        enc.load(enc.ptr_zero(),1 << 31,run.rows,run.columns,0,shrunk=schedule.dtype=='i8')
        if not ordering['selected']:
            enc.fence()
        enc.store(enc.ptr_offset(host.ptrs[result],enc.mul(run.output_element,schedule.itemsize)),
            0,run.rows,run.columns,run.output_stride*schedule.itemsize,dtype=storage.storage_dtype)
        enc.fence()
    schedule.emit(enc.index_builder,emit)
    if ordering['selected']:
        enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result,try_lower)
    host.last_lowering=('device',('exact_static_raw32_positive_zero_fill' if schedule.dtype=='f32' else 'exact_static_integer_zero_fill'),None)
    source_storage={}
    if schedule.dtype=='f32':
        from raw_zero_admission import zero_storage_receipt
        source_storage['storage_view']=zero_storage_receipt()
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_zero_transfer.v1',plan=asdict(schedule),**source_storage,
            resources=resource_admission(schedule),observers='original scalar and fill seed unchanged; complete result written',
            external_buffers='complete distinct original named ABI result',completion=('selected stream0 ACC RAW completes each zero load before store; per-store and entry/final full drains'
                if ordering['selected'] else 'original fence after each load and store'),
            ordering_admission=ordering,timing='UNKNOWN')))
    return True
