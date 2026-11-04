"""Exact integer concatenation through disjoint inverse slice DMA projections."""
from dataclasses import dataclass,replace
from math import prod
from merlin.sched.ir import Kernel,Loop,TensorArg
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
from . import isa
from .source_slice import _kernel as slice_kernel,_insert_kernel,plan_source_slice,SourceSliceError

class IntegerMovementError(ValueError):
    pass


def _geometry(shapes,axis,dtype):
    shapes=tuple(tuple(shape) for shape in shapes)
    if not shapes or dtype not in ('i8','i32'):
        raise IntegerMovementError('concat needs at least one signed i8/i32 source')
    rank=len(shapes[0])
    if (not 1<=rank<=6 or type(axis) is not int or not 0<=axis<rank
            or any(len(s)!=rank or any(type(n) is not int or n<=0 for n in s) for s in shapes)
            or any(s[d]!=shapes[0][d] for s in shapes for d in range(rank) if d!=axis)):
        raise IntegerMovementError('concat needs static positive equal-rank shapes and equal non-axis dimensions')
    output=list(shapes[0]);output[axis]=sum(s[axis] for s in shapes)
    intervals=[];start=0
    for shape in shapes:
        intervals.append((start,start+shape[axis]));start+=shape[axis]
    return shapes,tuple(output),tuple(intervals)


def _kernel(shapes,axis,dtype):
    shapes,output,intervals=_geometry(shapes,axis,dtype);rank=len(output);ones=(1,)*rank
    # The existing insert schedule is a complete base copy followed by the
    # inverse of a checked slice. Concatenation consumes only that inverse:
    # every output element is supplied by exactly one source, so no base exists.
    base=slice_kernel(output,(0,)*rank,output,ones,dtype);body=[]
    for index,(shape,(start,end)) in enumerate(zip(shapes,intervals,strict=True)):
        offsets=tuple(start if d==axis else 0 for d in range(rank))
        insertion=_insert_kernel(shape,output,offsets,ones,dtype)
        def base_copy(ops):
            mapping={'slice_mvin':'insert_base_mvin','slice_mvout':'insert_mvout',
                'slice_drain':'insert_drain','slice_config_load':'insert_config_load',
                'slice_config_store':'insert_config_store'}
            return tuple(replace(op,body=base_copy(op.body)) if isinstance(op,Loop) else
                replace(op,instr=mapping[op.instr]) for op in ops)
        if insertion.body[:len(base.body)]!=base_copy(base.body):
            raise IntegerMovementError('insert schedule no longer begins with the exact independent full base copy')
        def rename(ops):
            return tuple(replace(op,body=rename(op.body)) if isinstance(op,Loop) else
                replace(op,instr={'insert_src_mvin':f'concat_mvin_{index}',
                    'insert_mvout':'concat_mvout','insert_drain':'concat_drain',
                    'insert_config_load':'concat_config_load','insert_config_store':'concat_config_store'}[op.instr]) for op in ops)
        body.extend(rename(insertion.body[len(base.body):]))
    return Kernel('source_integer_concat',tuple(TensorArg(f'src{i}',s,dtype,'read') for i,s in enumerate(shapes))+
        (TensorArg('dst',output,dtype,'write'),),tuple(body))


def check_concat(kernel,shapes,axis,dtype):
    shapes,output,intervals=_geometry(shapes,axis,dtype);rank=len(output)
    regions=[]
    for shape,(start,end) in zip(shapes,intervals,strict=True):
        offsets=tuple(start if d==axis else 0 for d in range(rank))
        region=plan_source_slice(output,offsets,shape,(1,)*rank,dtype)
        regions.append(region.revalidate())
    if kernel!=_kernel(shapes,axis,dtype):
        raise IntegerMovementError('concat source association, inverse projection, descriptors or completion changed')
    if (intervals[0][0]!=0 or intervals[-1][1]!=output[axis]
            or any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:]))
            or sum(prod(s) for s in shapes)!=prod(output)):
        raise IntegerMovementError('concat source intervals do not partition the complete output')
    width=1 if dtype=='i8' else 4
    return dict(schema='submission.static_integer_concat.v1',kernel_sha256=kernel.digest(),
        input_shapes=[list(s) for s in shapes],output_shape=list(output),axis=axis,dtype=dtype,
        axis_intervals=[list(interval) for interval in intervals],inverted_slice_projections=regions,
        source_address_bounds=[[0,prod(s)-1] for s in shapes],destination_address_bounds=[0,prod(output)-1],
        writes=prod(output),write_multiplicity=1,requested_load_bytes=prod(output)*width,
        requested_store_bytes=prod(output)*width,base_copy_bytes=0,
        proof_method='actual canonical inverse-slice streams; cumulative disjoint axis intervals and Cartesian full/tail partitions',
        numeric='signed i8 identity readout' if dtype=='i8' else 'raw full-width i32 readout',
        completion='each load/store completes before accumulator reuse and subsequent source')


@dataclass(frozen=True)
class SourceConcatPlan:
    shapes:tuple
    axis:int
    dtype:str
    kernel:Kernel
    proof:dict
    def revalidate(self):
        result=check_concat(self.kernel,self.shapes,self.axis,self.dtype)
        if result!=self.proof:raise IntegerMovementError('concat proof receipt changed')
        return result


def plan_source_concat(shapes,axis,dtype):
    shapes,_,_=_geometry(shapes,axis,dtype)
    try:kernel=_kernel(shapes,axis,dtype);proof=check_concat(kernel,shapes,axis,dtype)
    except SourceSliceError as exc:raise IntegerMovementError(str(exc)) from exc
    return SourceConcatPlan(shapes,axis,dtype,kernel,proof)


def prove_source_concat(op):
    from xdsl.dialects.builtin import TensorType,NoneAttr
    if op.name!='tensor.concat' or len(op.results)!=1 or not op.operands:
        raise IntegerMovementError('source is not one tensor concatenation')
    types=[v.type for v in (*op.operands,op.results[0])]
    if any(not isinstance(t,TensorType) or not isinstance(t.encoding,NoneAttr) for t in types):
        raise IntegerMovementError('concat requires explicit dense ranked tensor boundaries')
    dtype=str(types[-1].element_type)
    if any(str(t.element_type)!=dtype for t in types):raise IntegerMovementError('concat cannot convert element types')
    axis=op.dim.value.data
    plan=plan_source_concat([t.get_shape() for t in types[:-1]],axis,dtype)
    if tuple(types[-1].get_shape())!=tuple(plan.proof['output_shape']):
        raise IntegerMovementError('concat result shape differs from its original source interval partition')
    return {f'src{i}':v for i,v in enumerate(op.operands)},dict(axis=axis,output_dtype=dtype),plan.revalidate()


def lower_source_concat(ctx,operands,attributes):
    if set(attributes)!={'axis','output_dtype'}:
        raise IntegerMovementError('concat has no arithmetic, epilogue or dynamic addressing contract')
    n=len(operands)-1
    if n<1 or set(operands)!={'dst',*(f'src{i}' for i in range(n))}:
        raise IntegerMovementError('concat operands must be consecutive source indices and one destination')
    sources=[operands[f'src{i}'] for i in range(n)];dst=operands['dst']
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if dst in sources or dst in aliases or any(src in aliases for src in sources):
        raise IntegerMovementError('concat requires a distinct materialized destination; repeated sources are allowed')
    buffers=getattr(ctx,'buffers',{})
    if dst in buffers and any(src in buffers and buffers[src] is buffers[dst] for src in sources):
        raise IntegerMovementError('concat destination aliases a registered source buffer')
    specs=[ctx.spec(v) for v in (*sources,dst)];dtype=attributes['output_dtype']
    if any(s['dtype']!=dtype for s in specs):raise IntegerMovementError('concat source and destination dtype changed')
    plan=plan_source_concat([s['shape'] for s in specs[:-1]],attributes['axis'],dtype)
    if list(specs[-1]['shape'])!=plan.proof['output_shape']:
        raise IntegerMovementError('concat destination does not match the proved source intervals')
    proof=plan.revalidate();bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        if name=='concat_drain':ctx.fence();return
        if name=='concat_config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=dtype=='i8'));return
        if name=='concat_config_store':ctx.emit(*isa.config_st(args['stride']));return
        values={k:bridge.value(v) for k,v in args.items()}
        descriptor=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name.startswith('concat_mvin_'):
            src=sources[int(name.removeprefix('concat_mvin_'))]
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,values['offset']),descriptor)
        elif name=='concat_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,values['offset']),descriptor)
        else:raise IntegerMovementError('unrecognized concat DMA instruction')
    receipt=dict(proof=proof,emission=emit_index_program(plan.kernel,bridge,emit))
    ctx.metadata.setdefault('source_integer_concats',[]).append(receipt)
    return receipt
