"""Checked contiguous row copies with full-width accumulator DMA tiles.

This only changes identity movement geometry. Every actual source/destination
byte and staging lifetime is checked by the existing transpose projection and
ownership checker before the common ISA encoder receives the counted program.
"""
from dataclasses import replace
from merlin.sched.ir import Loop,Var,Const
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
from . import isa
from .transpose import plan_transpose,TransposePlan,check_projection,_batched_kernel,_call


def plan_copy(elements,dtype):
    base=plan_transpose((1,elements),(0,1),dtype,schedule='baseline')
    width=1 if dtype=='i8' else 4
    def tile(offset,rows,cols):
        return (_call('transpose_mvin',offset=offset,addr=isa.acc_addr(0),rows=rows,cols=cols),
                _call('transpose_drain'),
                _call('transpose_mvout',offset=offset,addr=isa.acc_addr(0,full=dtype=='i32'),rows=rows,cols=cols),
                _call('transpose_drain'))
    d=isa.DIM;full,tail=divmod(elements,d*d);body=[]
    if full:body.append(Loop('copy_tile',full,tile(Var('copy_tile')*(d*d),d,d)))
    rows,cols=divmod(tail,d)
    if rows:body.extend(tile(Const(full*d*d),rows,d))
    if cols:body.extend(tile(Const(full*d*d+rows*d),1,cols))
    configs=(_call('transpose_drain'),_call('transpose_config_load',stride=d*width),
             _call('transpose_config_store',stride=d*width))
    kernel=replace(base.kernel,body=configs+tuple(body))
    check_projection(kernel,base.shape,base.permutation,dtype)
    kernel=_batched_kernel(kernel)
    proof=check_projection(kernel,base.shape,base.permutation,dtype)
    return TransposePlan(base.shape,base.permutation,dtype,kernel,proof)


def emit_copy(ctx,src,dst,plan):
    """Emit a prevalidated full identity partition through the common encoder."""
    if (ctx.spec(src)['dtype']!=plan.dtype or ctx.spec(dst)['dtype']!=plan.dtype
            or ctx.spec(src)['shape']!=list(plan.shape)
            or ctx.spec(dst)['shape']!=list(plan.shape)):
        raise ValueError('identity DMA copy boundary changed')
    plan.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='transpose_drain':ctx.fence();return
        if name=='transpose_config_load':
            ctx.emit(*isa.config_ld(arguments['stride'],shrink=plan.dtype=='i8'));return
        if name=='transpose_config_store':
            ctx.emit(*isa.config_st(arguments['stride']));return
        a={k:bridge.value(v) for k,v in arguments.items()}
        packed=isa.pack_matrix(a['addr'],a['rows'],a['cols'],fb=ctx.fb)
        if name=='transpose_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(src,a['offset']),packed)
        elif name=='transpose_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,a['offset']),packed)
        else:raise ValueError('unknown identity copy operation')
    emission=emit_index_program(plan.kernel,bridge,emit)
    return dict(proof=plan.proof,emission=emission,kernel_sha256=plan.kernel.digest(),
                mechanism='full_row_identity_dma',complete_elements=plan.shape[1])
