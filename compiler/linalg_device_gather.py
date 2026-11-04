"""Original runtime-index row gather using selected exact raw32 device transfers."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from xdsl.dialects.builtin import TensorType,NoneAttr,IndexType
from xdsl.ir.affine import AffineDimExpr
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.llvmlower.dynamic_tensor_extract import load_element
from merlin.compile.scheduling.source_gather_transfer import SourceRowGather
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission
from raw_storage_transfer import emit_raw_run,storage_receipt

ENABLED=True


def _identity(mapping,rank):
    return (mapping.num_dims==rank and not mapping.num_symbols and len(mapping.results)==rank
        and all(isinstance(e,AffineDimExpr)and e.position==i for i,e in enumerate(mapping.results)))


def plan(op):
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
            or len(op.regions)!=1 or len(op.regions[0].blocks)!=1 or op.successors
            or any(not k.startswith('prov.')for k in op.attributes)
            or set(op.properties)-{'indexing_maps','iterator_types','operandSegmentSizes'}):return None
    indices,seed=op.operands;result=op.results[0]
    body=op.regions[0].block;ops=tuple(body.ops)
    if (len(body.args)!=2 or body.args[1].uses or len(ops)!=4
            or tuple(x.name for x in ops)!=('arith.index_cast','linalg.index','tensor.extract','linalg.yield')):return None
    cast,column,extract,yielded=ops
    if cast.properties or set(column.properties)!={'dim'} or extract.properties or yielded.properties:return None
    if (len(cast.operands)!=1 or cast.operands[0]!=body.args[0] or len(cast.results)!=1
            or str(cast.operands[0].type)!='i64' or not isinstance(cast.results[0].type,IndexType)
            or len(column.results)!=1 or not isinstance(column.results[0].type,IndexType)
            or column.operands or len(extract.operands)!=3 or len(extract.results)!=1
            or tuple(extract.operands[1:])!=(cast.results[0],column.results[0])
            or tuple(yielded.operands)!=(extract.results[0],)):return None
    source=extract.operands[0]
    if any(not isinstance(v.type,TensorType)or not isinstance(v.type.encoding,NoneAttr)
           for v in (source,indices,seed,result)):return None
    if (str(indices.type.element_type)!='i64' or any(str(v.type.element_type)!='f32'
            for v in (source,seed,result)) or str(extract.results[0].type)!='f32'):return None
    try:schedule=SourceRowGather(tuple(tensor_shape(source.type)),tuple(tensor_shape(indices.type)))
    except (TypeError,ValueError):return None
    rank=len(schedule.output_shape);maps=tuple(x.data for x in op.properties['indexing_maps'])
    segment=op.properties.get('operandSegmentSizes')
    if (segment is None or tuple(segment.get_values())!=(1,1)
            or tuple(tensor_shape(seed.type))!=schedule.output_shape
            or tuple(tensor_shape(result.type))!=schedule.output_shape
            or len(maps)!=2 or not _identity(maps[1],rank)
            or maps[0].num_dims!=rank or maps[0].num_symbols
            or len(maps[0].results)!=rank-1
            or any(not isinstance(e,AffineDimExpr)or e.position!=i for i,e in enumerate(maps[0].results))
            or len(op.properties['iterator_types'])!=rank
            or any(x.data.value!='parallel'for x in op.properties['iterator_types'])
            or int(column.properties['dim'].value.data)!=rank-1):return None
    if resource_admission(LayoutTransfer((schedule.source_shape[1],),(0,),'f32',16))is None:return None
    # Exactly the original cast/index/extract/yield; reject extra effectful state.
    for inner in ops:
        if inner.regions or inner.successors or any(not k.startswith('prov.')for k in inner.attributes):return None
    return schedule,source,cast


def _published(host,value):
    if value not in host.ptrs:return False
    tensor=host.vals.get(value)
    return value in host.materialized_results or getattr(getattr(tensor,'elems',None),'ptr',None)is host.ptrs[value]


def try_lower(host,op):
    if not ENABLED or host.row_alignment!=1:return False
    planned=plan(op)
    if planned is None:return False
    schedule,source,cast=planned;indices,seed=op.operands;result=op.results[0]
    if (any(v not in host.ptrs for v in (source,indices,seed,result))
            or not _published(host,source)or not _published(host,indices)
            or result in getattr(host,'aliasable_results',())
            or result in getattr(host,'storage_aliases',{})
            or any(host.ptrs[result]is host.ptrs[v]for v in (source,indices,seed))):return False
    enc=host.enc;builder=enc.index_builder;shape=schedule.index_shape
    def index(flat):
        # Use the shared exact runtime tensor extraction for the original index
        # tensor map; the original signed i64->index scalar op remains its owner.
        positions=[]
        from math import prod
        for i,size in enumerate(shape):
            positions.append(builder.op('urem',builder.op('udiv',flat,prod(shape[i+1:])),size))
        scalar=load_element(host.fp,host.ptrs[indices],shape,'i64',positions,row_alignment=host.row_alignment)
        return host._scalar_op(cast,[scalar])
    schedule.emit(builder,index,lambda run:emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,'f32'))
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result);host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_runtime_index_raw32_row_gather',None)
    root=Path(__file__).parent
    host.device_receipts.append(dict(source_op_index=host.current_source_op_index,
        emission=dict(schema='merlin.source_row_gather.v1',plan=asdict(schedule),
            source_operation_sha256=sha256(str(op).encode()).hexdigest(),
            storage_view=storage_receipt('f32'),resources=resource_admission(LayoutTransfer((schedule.source_shape[1],),(0,),'f32',16)),
            index_semantics='original signed i64 index_cast to 64-bit index; tensor.extract requires 0<=row<source rows; no wrapping/clamping/new OOB behavior',
            index_preparation='shared dynamic_tensor_extract.load_element of original i64 index tensor; original scalar index_cast',
            source_payload='original complete pointer-backed table; raw32 transfers only',
            observers='all original table/index/seed observers preserved; complete distinct output written',
            completion='fence after every raw load and store; scratch reused only after store completion',
            source_owners={p:sha256((root/p).read_bytes()).hexdigest()for p in ('linalg_device_gather.py','compiler_library/merlin/compile/scheduling/source_gather_transfer.py','raw_storage_transfer.py','compiler_library/merlin/llvmlower/dynamic_tensor_extract.py')},timing='UNKNOWN')))
    return True
