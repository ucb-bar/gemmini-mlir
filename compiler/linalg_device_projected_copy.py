"""Pure projected linalg.generic copy/broadcast on unchanged raw Encoder."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseArrayBase
from xdsl.ir.affine import AffineDimExpr,AffineConstantExpr
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_projected_transfer import SourceProjectedTransfer
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission as layout_admission
from raw_storage_transfer import emit_raw_run,storage_receipt

ENABLED=True


def resource_admission(schedule,facts=None):
    # Reuse the selected mesh/ACC/raw32 and register-field law; independently
    # account for this projection's potentially strided source rows.
    witness=LayoutTransfer((1,),(0,),schedule.dtype,schedule.tile)
    admitted=layout_admission(witness,facts)
    if admitted is None:
        return None
    maximum=schedule.maximum_stride_elements*schedule.itemsize
    if maximum >= 1 << admitted['conservative_stride_bits']:
        return None
    return dict(admitted,maximum_stride_bytes=maximum)


def _identity(mapping,rank):
    return (mapping.num_dims==rank and not mapping.num_symbols and len(mapping.results)==rank
        and all(isinstance(e,AffineDimExpr)and e.position==i for i,e in enumerate(mapping.results)))


def plan(op):
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
        or len(op.regions)!=1 or len(op.regions[0].blocks)!=1 or op.successors
        or any(not k.startswith('prov.')for k in op.attributes)
        or set(op.properties)-{'indexing_maps','iterator_types','operandSegmentSizes'}):
        return None
    source,seed=op.operands;result=op.results[0]
    values=(source,seed,result)
    if any(not isinstance(v.type,TensorType)or not isinstance(v.type.encoding,NoneAttr)for v in values):
        return None
    dtype=str(source.type.element_type)
    if dtype not in ('i8','i32','f32')or any(str(v.type.element_type)!=dtype for v in values):
        return None
    block=op.regions[0].block;body=tuple(block.ops)
    if (len(block.args)!=2 or any(str(v.type)!=dtype for v in block.args)
        or block.args[1].uses or len(body)!=1 or body[0].name!='linalg.yield'
        or tuple(body[0].operands)!=(block.args[0],)or body[0].properties
        or body[0].regions or body[0].successors
        or any(not k.startswith('prov.')for k in body[0].attributes)):
        return None
    segment=op.properties.get('operandSegmentSizes')
    if (segment is None or not isinstance(segment,DenseArrayBase)or str(segment.elt_type)!='i32'
        or tuple(segment.iter_values())!=(1,1)):
        return None
    try:
        maps=tuple(x.data for x in op.properties['indexing_maps'])
        shape=tuple(tensor_shape(result.type));rank=len(shape)
        if (tuple(tensor_shape(seed.type))!=shape or len(maps)!=2 or not _identity(maps[1],rank)
            or maps[0].num_dims!=rank or maps[0].num_symbols
            or len(maps[0].results)!=len(tensor_shape(source.type))
            or len(op.properties['iterator_types'])!=rank
            or any(x.data.value!='parallel'for x in op.properties['iterator_types'])):
            return None
        axes=[]
        for expr in maps[0].results:
            if isinstance(expr,AffineDimExpr):
                axes.append(expr.position)
            elif isinstance(expr,AffineConstantExpr)and expr.value==0:
                axes.append(None)
            else:
                return None
        schedule=SourceProjectedTransfer(tuple(tensor_shape(source.type)),shape,tuple(axes),dtype)
    except (KeyError,TypeError,ValueError,AttributeError):
        return None
    return schedule if resource_admission(schedule)is not None else None


def _published(host,value):
    tensor=host.vals.get(value)
    return (value in host.ptrs and (value in host.materialized_results
        or getattr(getattr(tensor,'elems',None),'ptr',None)is host.ptrs[value]))


def try_lower(host,op):
    if not ENABLED or host.row_alignment!=1:
        return False
    schedule=plan(op)
    if schedule is None:
        return False
    source,seed=op.operands;result=op.results[0]
    if (any(v not in host.ptrs for v in (*op.operands,*op.results))or not _published(host,source)
        or result in getattr(host,'aliasable_results',())or result in getattr(host,'storage_aliases',{})
        or any(v is not result and ptr is host.ptrs[result]for v,ptr in host.ptrs.items())):
        return False
    enc=host.enc
    enc.fence()
    schedule.emit(enc.index_builder,lambda run:emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,schedule.dtype))
    enc.fence()
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result);host.record_full_writer(result,try_lower)
    host.last_lowering=('device','exact_static_projected_raw_copy',None)
    root=Path(__file__).parent
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_projected_transfer.v1',plan=asdict(schedule),
            source_operation_sha256=sha256(str(op).encode()).hexdigest(),
            storage_view=storage_receipt(schedule.dtype),resources=resource_admission(schedule),
            source_coordinates='original projected output dimensions or literal zero; distinct dimensions; bounds checked',
            observers='original full input and seed unmodified; complete distinct output written',
            completion='entry/final drains; fence after every load and store before scratch reuse',
            source_owners={p:sha256((root/p).read_bytes()).hexdigest()for p in ('linalg_device_projected_copy.py','compiler_library/merlin/compile/scheduling/source_projected_transfer.py','compiler_library/merlin/compile/scheduling/layout_transfer.py','raw_storage_transfer.py')},
            timing='UNKNOWN')))
    return True
