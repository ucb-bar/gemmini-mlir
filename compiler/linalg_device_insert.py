"""Static tensor.insert_slice on selected raw accelerator load/store storage."""
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseArrayBase
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_insert_transfer import SourceInsertTransfer
from linalg_device_slice import resource_admission as slice_resource_admission
from raw_storage_transfer import emit_raw_run,storage_receipt

ENABLED=True

def resource_admission(schedule,facts=None):
    resources=slice_resource_admission(schedule.patch,facts)
    if resources is None or slice_resource_admission(schedule.seed,facts) is None:
        return None
    resources['route']='complete_seed_copy_then_reversed_source_slice_patch'
    return resources

def plan(op):
    if (op_name(op)!='tensor.insert_slice' or len(op.operands)!=2 or len(op.results)!=1
        or op.regions or op.successors or any(not k.startswith('prov.') for k in op.attributes)
        or set(op.properties)-{'static_offsets','static_sizes','static_strides','operandSegmentSizes'}):
        return None
    source,seed=op.operands;result=op.results[0]
    values=(source,seed,result)
    if any(not isinstance(v.type,TensorType) or not isinstance(v.type.encoding,NoneAttr) for v in values):
        return None
    dtype=str(source.type.element_type)
    if dtype not in ('i8','i32','f32') or any(str(v.type.element_type)!=dtype for v in values):
        return None
    if tuple(tensor_shape(seed.type))!=tuple(tensor_shape(result.type)):
        return None
    arrays=[op.properties.get(k) for k in ('static_offsets','static_sizes','static_strides')]
    if any(not isinstance(v,DenseArrayBase) or str(v.elt_type)!='i64' for v in arrays):
        return None
    segment=op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment,DenseArrayBase) or str(segment.elt_type)!='i32'
        or tuple(segment.iter_values())!=(1,1,0,0,0)):
        return None
    try:
        schedule=SourceInsertTransfer(tuple(tensor_shape(source.type)),tuple(tensor_shape(seed.type)),
            *(tuple(v.iter_values()) for v in arrays),dtype,16)
    except (ValueError,TypeError):
        return None
    return schedule if resource_admission(schedule) is not None else None

def try_lower(host,op):
    if not ENABLED:
        return False
    schedule=plan(op)
    if schedule is None:
        return False
    source,seed=op.operands;result=op.results[0]
    if (any(v not in host.ptrs for v in (source,seed,result))
        or result in getattr(host,'aliasable_results',())
        or result in getattr(host,'storage_aliases',{})
        or any(host.ptrs[result] is host.ptrs[v] for v in (source,seed))):
        return False
    enc=host.enc
    enc.fence()
    schedule.emit(enc.index_builder,
        lambda run:emit_raw_run(enc,host.ptrs[seed],host.ptrs[result],run,schedule.dtype),
        lambda run:emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,schedule.dtype))
    enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_'+('raw32' if schedule.dtype=='f32' else 'integer')+'_insert_slice',None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_insert_transfer.v1',plan=asdict(schedule),
            storage_view=storage_receipt(schedule.dtype),resources=resource_admission(schedule),
            adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            schedule_sha256=sha256((Path(__file__).parent/'compiler_library/merlin/compile/scheduling/source_insert_transfer.py').read_bytes()).hexdigest(),
            observers='complete original source and seed unmodified; complete distinct result written',
            external_buffers='original full named ABI buffers; fresh result disjoint from both inputs',
            completion='fence after every load and store; complete seed copy precedes patch',timing='UNKNOWN')))
    return True
