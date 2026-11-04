"""Exact static native i64 insert_slice through complete seed copy and raw pair patch."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseArrayBase
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_word_insert_transfer import SourceWordInsertTransfer
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission as base_admission
from raw_storage_transfer import emit_raw_run,storage_receipt,raw_storage_admission

ENABLED=True


def resource_admission(schedule,facts=None):
    if facts is None:
        from encoding import FACTS
        facts=FACTS
    raw=raw_storage_admission('f32',facts)
    resources=base_admission(LayoutTransfer((16,),(0,),'i32',16),facts)
    if raw is None or resources is None:return None
    _,step,_=schedule.movement.patch.run_geometry
    stride=max(schedule.tile,step)*8
    if stride>=1<<resources['conservative_stride_bits']:return None
    return dict(resources,maximum_stride_bytes=stride,raw_word_preservation=raw,
        logical_tile=8,maximum_raw_rows=8,maximum_raw_columns=16,words_per_element=2,
        word_order='unchanged increasing byte-address order; endian-independent',payload_arithmetic=False)


def plan(op):
    if (op_name(op)!='tensor.insert_slice' or len(op.operands)!=2 or len(op.results)!=1
            or op.regions or op.successors or any(not k.startswith('prov.')for k in op.attributes)
            or set(op.properties)-{'static_offsets','static_sizes','static_strides','operandSegmentSizes'}):return None
    source,seed=op.operands;result=op.results[0]
    if any(not isinstance(v.type,TensorType)or not isinstance(v.type.encoding,NoneAttr)
            or str(v.type.element_type)!='i64'for v in (source,seed,result)):return None
    if tuple(tensor_shape(seed.type))!=tuple(tensor_shape(result.type)):return None
    arrays=[op.properties.get(k)for k in ('static_offsets','static_sizes','static_strides')]
    if any(not isinstance(v,DenseArrayBase)or str(v.elt_type)!='i64'for v in arrays):return None
    segment=op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment,DenseArrayBase)
            or str(segment.elt_type)!='i32' or tuple(segment.iter_values())!=(1,1,0,0,0)):return None
    try:
        schedule=SourceWordInsertTransfer(tuple(tensor_shape(source.type)),tuple(tensor_shape(seed.type)),
            *(tuple(v.iter_values())for v in arrays))
        schedule.__post_init__()
    except (TypeError,ValueError):return None
    if resource_admission(schedule)is None:return None
    return schedule


def try_lower(host,op):
    if not ENABLED or host.row_alignment!=1:return False
    schedule=plan(op)
    if schedule is None:return False
    source,seed=op.operands;result=op.results[0]
    if (any(v not in host.ptrs for v in (source,seed,result))
            or result in getattr(host,'aliasable_results',())
            or result in getattr(host,'storage_aliases',{})
            or any(v is not result and ptr is host.ptrs[result]for v,ptr in host.ptrs.items())):return False
    for value in (source,seed):
        tensor=host.vals.get(value)
        if (value not in host.materialized_results
                and getattr(getattr(tensor,'elems',None),'ptr',None)is not host.ptrs[value]):return False
    if (source is seed or host.ptrs[source] is host.ptrs[seed]
            or any(v in getattr(host,'storage_aliases',{})for v in (source,seed))):return False
    schedule.emit(host.enc.index_builder,
        lambda run:emit_raw_run(host.enc,host.ptrs[seed],host.ptrs[result],run,'i32'),
        lambda run:emit_raw_run(host.enc,host.ptrs[source],host.ptrs[result],run,'i32'))
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result);host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_i64_word_insert_slice',None)
    root=Path(__file__).parent
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_word_insert_transfer.v1',plan=asdict(schedule),
            source_operation_sha256=sha256(str(op).encode()).hexdigest(),
            storage_view=storage_receipt('i32'),resources=resource_admission(schedule),
            observers='complete original source and seed unchanged; complete distinct native i64 output written including outside patch',
            proof='existing complete logical insertion and unit-axis rank expansion proven before expansion to adjacent raw32 word pairs in unchanged address order',
            external_buffers='packed original pointer-backed source and seed; fresh complete output disjoint from every known pointer and alias record',
            completion='existing fence after every raw32 load and store; no ACC reuse before completion',
            source_owners={p:sha256((root/p).read_bytes()).hexdigest()for p in (
                'linalg_device_word_insert.py','compiler_library/merlin/compile/scheduling/source_word_insert_transfer.py',
                'compiler_library/merlin/compile/scheduling/source_insert_transfer.py',
                'compiler_library/merlin/compile/scheduling/source_slice_transfer.py','raw_storage_transfer.py','encoding.py')},timing='UNKNOWN')))
    return True
