"""Exact finite-domain residual quantizers through integer weighted products.

The source's three f32 scales and stage order determine a complete 256x256
output table. A bounded rational proposal is accepted only when the actual
integer weighted product and binary32 readout match every entry exactly.
"""
from functools import lru_cache
from fractions import Fraction
from math import ceil, isfinite, prod

from merlin.llvmlower.requantization import f32


class WeightedResidualError(ValueError):
    pass


@lru_cache(maxsize=128)
def _prove_weighted_residual(scales, relu=False):
    import numpy as np
    scales=tuple(f32(s) for s in scales)
    if len(scales)!=3 or any(not isfinite(s) or s<=0 for s in scales):
        raise WeightedResidualError('weighted residual needs its three finite positive binary32 source scales')
    left,right,quant=scales
    reciprocal=f32(1.0/quant)
    if not isfinite(reciprocal) or reciprocal<=0:
        raise WeightedResidualError('source quantizer reciprocal is not finite positive binary32')
    a=np.arange(-128,128,dtype=np.float32)[:,None]
    b=np.arange(-128,128,dtype=np.float32)[None,:]
    source=np.add(np.multiply(a,np.float32(left),dtype=np.float32),
        np.multiply(b,np.float32(right),dtype=np.float32),dtype=np.float32)
    if not np.isfinite(source).all():
        raise WeightedResidualError('source residual overflows binary32 before readout')
    if relu:source=np.maximum(source,np.float32(0))
    source=np.clip(np.rint(np.multiply(source,np.float32(reciprocal),dtype=np.float32)),-128,127)
    witnesses=[];seen=set()
    # This bounded proposal search is independent of tensor shape/workload.
    # No proposed coefficient or scale grants admission without all-pair proof.
    for denominator in (127,512,4096,32767):
        ratio=Fraction(left/right).limit_denominator(denominator)
        x,y=ratio.numerator,ratio.denominator
        if (x,y) in seen:continue
        seen.add((x,y))
        columns=ceil(x/127)+ceil(y/127)
        if columns>256 or 128*(x+y)>=(1<<31):continue
        scale=f32(f32(right/quant)/y)
        if not isfinite(scale) or scale<=0:continue
        acc=a.astype(np.int64)*x+b.astype(np.int64)*y
        if relu:acc=np.maximum(acc,0)
        target=np.clip(np.rint(np.multiply(acc.astype(np.float32),np.float32(scale),dtype=np.float32)),-128,127)
        different=np.argwhere(source!=target)
        if len(different):
            i,j=map(int,different[0]);witnesses.append(dict(coefficients=[x,y],
                lhs=i-128,rhs=j-128,source=int(source[i,j]),target=int(target[i,j])))
            continue
        weights=[]
        for coefficient in (x,y):
            q,r=divmod(coefficient,127);weights.extend([127]*q)
            if r:weights.append(r)
        return dict(schema='submission.exact_weighted_residual.v1',source_scales=list(scales),
            source_reciprocal=reciprocal,coefficients=[x,y],acc_scale=scale,relu=bool(relu),
            weights=weights,lhs_columns=ceil(x/127),rhs_columns=ceil(y/127),
            accumulator_min=-128*(x+y),accumulator_max=127*(x+y),
            proof='exhaustive original ordered binary32 source versus integer product/readout',
            pairs_checked=65536,max_error_lsb=0,
            source_order=['i8_to_f32','multiply_each_source_scale_f32','add_f32']+
                (['relu_f32'] if relu else [])+['multiply_reciprocal_f32','roundeven','clamp_i8'],
            device_order=['exact_i8_weighted_i32_sum']+(['relu_i32'] if relu else [])+
                ['i32_to_f32','multiply_readout_scale_f32','roundeven','clamp_i8'],
            refused_proposals=witnesses)
    raise WeightedResidualError('no exact weighted residual within 256 signed-i8 product columns; '+str(witnesses))


def prove_weighted_residual(scales,relu=False):
    from copy import deepcopy
    return deepcopy(_prove_weighted_residual(tuple(scales),bool(relu)))


def packing_kernel(m,lhs_columns,rhs_columns):
    """Counted identity DMA expansion; every packed matrix lane is written once."""
    from merlin.sched.ir import Kernel,TensorArg,Call,Loop,Const,Var,as_expr
    from . import isa
    if any(type(n) is not int or n<=0 for n in (m,lhs_columns,rhs_columns)):
        raise WeightedResidualError('packing needs positive static matrix geometry')
    k=lhs_columns+rhs_columns
    if k>256 or m*k>=1<<63:raise WeightedResidualError('weighted residual packing exceeds its checked bounded geometry')
    def call(name,**args):return Call(name,tuple((key,as_expr(v)) for key,v in args.items()))
    def columns(row,rows,name,start,count):
        offset=row*k+start+Var('column')
        return Loop('column',count,(call(name,offset=row,addr=isa.acc_addr(0),rows=rows,cols=1),
            call('drain'),call('store',offset=offset,addr=isa.acc_addr(0),rows=rows,cols=1),call('drain')))
    def rows(row,count):return (columns(row,count,'load_lhs',0,lhs_columns),
        columns(row,count,'load_rhs',lhs_columns,rhs_columns))
    full,tail=divmod(m,isa.DIM);body=[]
    if full:body.append(Loop('row',full,rows(Var('row')*isa.DIM,isa.DIM)))
    if tail:body.extend(rows(Const(full*isa.DIM),tail))
    return Kernel('source_weighted_residual_pack',(TensorArg('lhs',(m,),'i8','read'),
        TensorArg('rhs',(m,),'i8','read'),TensorArg('packed',(m,k),'i8','write')),
        (call('drain'),call('config_load',stride=1),call('config_store',stride=k),*body))


def check_packing(kernel,m,lhs_columns,rhs_columns):
    if kernel!=packing_kernel(m,lhs_columns,rhs_columns):
        raise WeightedResidualError('weighted residual counted packing stream changed')
    k=lhs_columns+rhs_columns
    return dict(kernel_sha256=kernel.digest(),shape=[m,k],
        proof='canonical counted-stream equality; disjoint lhs/rhs column ranges and full/tail row partition',
        source_address_bounds=[0,m-1],destination_address_bounds=[0,m*k-1],
        writes=m*k,write_multiplicity=1,
        address_law='packed[r,c] = lhs[r] for c<lhs_columns, otherwise rhs[r]')


def lower_weighted_residual(ctx,operands,attributes):
    """Pack only on device, then consume the shared exact dense schedule."""
    from . import isa
    from .dense import lower_dense
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
    lhs,rhs,dst=(operands[k] for k in ('lhs','rhs','dst'))
    specs=[ctx.spec(v) for v in (lhs,rhs,dst)]
    if (attributes.get('output_dtype')!='i8'
            or attributes.get('epilogue',[]) not in ([],['relu'])):
        raise WeightedResidualError('weighted residual readout attributes changed')
    if (any(s['dtype']!='i8' for s in specs) or any(s['shape']!=specs[0]['shape'] for s in specs)
            or dst in (lhs,rhs) or not specs[0]['shape'] or any(n<=0 for n in specs[0]['shape'])):
        raise WeightedResidualError('weighted residual needs matching distinct static signed-i8 boundaries')
    proof=prove_weighted_residual(tuple(attributes['source_scales']),'relu' in attributes.get('epilogue',()))
    if proof!=attributes['numeric_proof']:
        raise WeightedResidualError('weighted residual complete source-domain proof changed')
    m=prod(specs[0]['shape']);k=len(proof['weights']);tag=f'_weighted_residual_{len(ctx.tensors)}'
    packed,weights,out=tag+'_packed',tag+'_weights',tag+'_output'
    ctx.alloc(packed,[m,k],'i8');ctx.constant(weights,[k,1],'i8',proof['weights'])
    ctx.register(out,ctx.buffer(dst),[m,1],'i8')
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    packing_selection=knob('WEIGHTED_PACKING_EPOCHS',default='off',values=('off','on'),
        root=Path(__file__).resolve().parents[1])
    selected_kernel,selected_check=packing_kernel,check_packing
    if packing_selection=='on':
        from .weighted_packing_epochs import packing_kernel as selected_kernel,check_packing as selected_check
    kernel=selected_kernel(m,proof['lhs_columns'],proof['rhs_columns'])
    packing=selected_check(kernel,m,proof['lhs_columns'],proof['rhs_columns'])
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,args):
        if name=='drain':ctx.fence();return
        if name=='config_load':ctx.emit(*isa.config_ld(args['stride'],shrink=True));return
        if name=='config_store':ctx.emit(*isa.config_st(args['stride']));return
        values={key:bridge.value(value) for key,value in args.items()}
        descriptor=isa.pack_matrix(values['addr'],values['rows'],values['cols'],fb=ctx.fb)
        if name in ('load_lhs','load_rhs'):
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(lhs if name=='load_lhs' else rhs,values['offset']),descriptor)
        elif name=='store':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(packed,values['offset']),descriptor)
        else:raise WeightedResidualError('unknown weighted residual packing instruction')
    packing_emission=emit_index_program(kernel,bridge,emit)
    dense=lower_dense(ctx,packed,weights,out,m=m,k=k,n=1,attributes={
        'epilogue':['acc_scale']+(['relu'] if proof['relu'] else []),
        'acc_scale':proof['acc_scale'],'output_dtype':'i8'})
    receipt=dict(numeric_proof=proof,packed_shape=[m,k],packing_proof=packing,packing_emission=packing_emission,
        packing='device i8 identity DMA; each row repeats original lhs/rhs in disjoint columns',
        packing_selection=packing_selection,
        packing_bytes=m*k,packing_requested_load_bytes=packing.get('requested_load_bytes',m*k),packing_requested_store_bytes=m*k,
        packing_dma_descriptors=packing.get('source_load_descriptors',k*ceil(m/isa.DIM))+packing.get('store_descriptors',k*ceil(m/isa.DIM)),packing_columns_limit=256,
        packing_accumulator_rows=min(m,isa.DIM),packing_accumulator_columns=1,
        prefix_accumulator_absolute_bound=128*sum(proof['weights']),
        dense_leaf=dense,completion='full fences before all row reuse and dense consumption')
    ctx.metadata.setdefault('weighted_residuals',[]).append(receipt)
    return receipt
