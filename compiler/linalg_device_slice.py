"""Optional original static integer slice copies on accelerator load/store."""
from dataclasses import asdict
from math import prod
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_slice_transfer import SourceSliceTransfer
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission as layout_resource_admission

from raw_storage_transfer import emit_raw_run, storage_receipt

ENABLED = True
F32_ENABLED = True


def resource_admission(schedule, facts=None):
    """Reuse selected raw ACC/field pin and check original slice source pitches."""
    resources = layout_resource_admission(LayoutTransfer((16,), (0,), schedule.dtype, schedule.tile), facts)
    if resources is None:
        return None
    _, step, contiguous = schedule.run_geometry
    maximum = max(schedule.tile,step)*schedule.itemsize
    if maximum >= 1 << resources['conservative_stride_bits']:
        return None
    resources['maximum_stride_bytes'] = maximum
    # Reuse the existing movement emitter when the complete slice is packed.
    # It treats the original result's final axis as columns, hence its pitch.
    columns = schedule.result_shape[-1] if schedule.result_shape else 1
    resources['route'] = ('existing_emit_movement' if schedule.dtype != 'f32' and schedule.is_contiguous
        and columns*schedule.itemsize < 1 << resources['conservative_stride_bits']
        else 'counted_source_slice_transfer')
    if resources['route'] == 'existing_emit_movement':
        resources['maximum_stride_bytes'] = max(maximum,columns*schedule.itemsize)
    return resources


def plan(op):
    if (op_name(op) != 'tensor.extract_slice' or len(op.operands) != 1
        or len(op.results) != 1 or op.regions or op.successors
        or any(not k.startswith('prov.') for k in op.attributes)
        or set(op.properties)-{'static_offsets','static_sizes','static_strides','operandSegmentSizes'}):
        return None
    source, result = op.operands[0], op.results[0]
    if any(not isinstance(v.type,TensorType) or not isinstance(v.type.encoding,NoneAttr) for v in (source,result)):
        return None
    dtype = str(source.type.element_type)
    if dtype == 'f32' and not F32_ENABLED:
        return None
    if dtype not in ('i8','i32','f32') or str(result.type.element_type) != dtype:
        return None
    arrays = [op.properties.get(k) for k in ('static_offsets','static_sizes','static_strides')]
    if any(not isinstance(v,DenseArrayBase) or str(v.elt_type) != 'i64' for v in arrays):
        return None
    segment = op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment,DenseArrayBase)
        or str(segment.elt_type) != 'i32' or tuple(segment.iter_values()) != (1,0,0,0)):
        return None
    try:
        schedule = SourceSliceTransfer(tuple(tensor_shape(source.type)),
            *(tuple(v.iter_values()) for v in arrays),tuple(tensor_shape(result.type)),dtype,16)
    except (ValueError,TypeError):
        return None
    if resource_admission(schedule) is None:
        return None
    return schedule


def try_lower(host,op):
    if not ENABLED:
        return False
    schedule = plan(op)
    if schedule is None:
        return False
    source,result = op.operands[0],op.results[0]
    if (any(v not in host.ptrs for v in (source,result))
        or result in getattr(host,'aliasable_results',())
        or result in getattr(host,'storage_aliases',{})
        or host.ptrs[result] is host.ptrs[source]):
        return False
    enc = host.enc
    resources = resource_admission(schedule)
    if resources['route'] == 'existing_emit_movement' and schedule.dtype != 'f32':
        from elementwise import emit_movement
        emit_movement(enc,enc.ptr_offset(host.ptrs[source],schedule.first_element*schedule.itemsize),
            host.ptrs[result],schedule.result_shape or (1,),src_dtype=schedule.dtype,output_dtype=schedule.dtype)
    else:
        def emit(run):
            emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,schedule.dtype)
        schedule.emit(enc.index_builder,emit)
    host.vals[result] = host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result,try_lower)
    host.last_lowering = ('device',('exact_static_raw32_extract_slice' if schedule.dtype=='f32' else 'exact_static_integer_extract_slice'),None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_slice_transfer.v1',plan=asdict(schedule), storage_view=storage_receipt(schedule.dtype),resources=resources,
            observers='complete original source unmodified; complete distinct result written',
            external_buffers='original full named ABI buffers; result disjoint from source',
            completion='fence after each load and store',timing='UNKNOWN')))
    return True
