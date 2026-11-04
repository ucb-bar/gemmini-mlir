"""Default-off bounded receptive-field panels, dense Q1, then hardware pool.

Input preparation uses checked window runs and signed DMA/identity readout.
Every contraction and Q1 readout uses dense.py; the original quantized image is
written once and consumed by the existing checked hardware spatial-max route.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from math import prod, isfinite
from pathlib import Path
from functools import lru_cache

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import Window2D

VALUES = ('off', 'packed_dense_panels', 'packed_dense_dma_epochs', 'dma_epochs_small_geometry')
DECLARATION = dict(values=list(VALUES), mechanism='bounded NHWC preparation panels into shared dense contraction and quantized-once hardware pool',
    note='Off incumbent. Explicit DMA modes remain available. dma_epochs_small_geometry selects only the bounded one-column-tile class using frozen completed RTL pairs; all other shapes decline before allocation.')


def selected():
    return knob('conv_pool_panel', default='off', values=VALUES, root=Path(__file__).parent)


@dataclass(frozen=True)
class PanelPlan:
    window: object
    channels: int
    rows: int
    lhs: str
    rhs: str
    output: str
    bias: str | None
    attrs: dict
    pool: object

    @property
    def reduction(self):return prod(self.window.kernel)*self.window.input_shape[-1]
    @property
    def output_rows(self):return self.window.input_shape[0]*prod(self.window.output_spatial)
    @property
    def panel_bytes(self):return self.rows*self.reduction
    @property
    def q1_bytes(self):return self.output_rows*self.channels


def prepare(command, tensors, pointers, *, residents=None, panel_rows=None):
    """Return a source-bound plan, or None before allocation/target emission."""
    try:
        return _prepare(command,tensors,pointers,residents=residents,panel_rows=panel_rows)
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):
        return None


def _prepare(command, tensors, pointers, *, residents=None, panel_rows=None):
    from convolution import _capacities
    from conv_pool_policy import plan_quantized_once
    if command.get('opcode') not in ('CONV','CONV2D'):return None
    a=command.get('attributes',{});o=command.get('operands',{})
    if set(o)-{'ifm','input','weight','dst','bias'}:return None
    if 'ifm' in o and 'input' in o and o['ifm']!=o['input']:return None
    if 'bias' in a and 'bias' in o and a['bias']!=o['bias']:return None
    # This adapter proves only the explicit integer-zero NHWC contract. Unknown
    # attributes may carry a numerical or layout obligation: decline them.
    known={'kernel','stride','padding','dilation','layout','output_dtype',
        'epilogue','acc_scale','acc_scale_by_column','bias','pool_in_dims',
        'pool_size','pool_stride','pool_padding','pool_pad_value',
        'input_zero_point','weight_zero_point','output_zero_point'}
    if set(a)-known:return None
    if any(type(a[k]) is not int or a[k]!=0 for k in
        ('input_zero_point','weight_zero_point','output_zero_point') if k in a):return None
    if (str(a.get('layout','nhwc')).lower()!='nhwc' or a.get('output_dtype')!='i8'
        or 'maxpool' not in a.get('epilogue',()) or a.get('acc_scale_by_column') is not None
        or 'requant_shift' in a or any(a.get(k,False) for k in ('transpose','rhs_transposed','a_transpose','b_transpose'))):return None
    stages=['bias_add' if s=='bias' else s for s in a.get('epilogue',())]
    canonical=['bias_add','acc_scale','relu','maxpool']
    if stages!=[s for s in canonical if s in stages]:return None
    if not isfinite(float(a.get('acc_scale',1.0))) or float(a.get('acc_scale',1.0))<=0:return None
    if float(a.get('acc_scale',1.0))!=1.0 and 'acc_scale' not in stages:return None
    lhs=o.get('ifm',o.get('input'));rhs=o.get('weight');out=o.get('dst')
    seen=set()
    while rhs in (residents or {}):
        if rhs in seen:return None
        seen.add(rhs);rhs=residents[rhs]
        if isinstance(rhs,dict):rhs=rhs.get('src',rhs.get('source'))
    if any(x not in tensors or x not in pointers for x in (lhs,rhs,out)):return None
    shape=tuple(tensors[lhs]['shape']);kernel=tuple(a.get('kernel',()))
    if len(shape)!=4 or len(kernel) not in (2,4):return None
    co=tensors[out]['shape'][-1];ci=shape[-1]
    if any(type(d) is not int or d<=0 for d in (*shape,*kernel,co)):return None
    if len(kernel)==4 and kernel[2:]!=(ci,co):return None
    if tensors[lhs]['dtype']!='i8' or tensors[rhs]['dtype']!='i8' or tensors[out]['dtype']!='i8':return None
    if tuple(tensors[rhs]['shape']) not in ((prod(kernel[:2])*ci,co),(*kernel[:2],ci,co)):return None
    bias=a.get('bias',o.get('bias'))
    if bool(bias)!=('bias_add' in stages):return None
    if any(s in a.get('epilogue',()) for s in ('bias','bias_add')) and not bias:return None
    if bias and (bias not in pointers or bias not in tensors or tensors[bias]['dtype']!='i32' or prod(tensors[bias]['shape'])!=co):return None
    try:
        w=Window2D(shape,kernel[:2],tuple(a.get('stride',(1,1))),tuple(a.get('dilation',(1,1))),tuple(a.get('padding',(0,0,0,0))))
        ps=tuple(a['pool_size']);st=tuple(a['pool_stride'])
        if len(ps)!=2 or len(st)!=2 or ps[0]!=ps[1] or st[0]!=st[1]:return None
        pool=plan_quantized_once(batches=shape[0],input_hw=w.output_spatial,channels=co,
            pool_size=ps[0],pool_stride=st[0],padding=tuple(a.get('pool_padding',(0,0,0,0))),
            pad_value=a.get('pool_pad_value',-128),relu='relu' in a.get('epilogue',()))
    except (ValueError,KeyError,TypeError):return None
    if a.get('pool_in_dims') and tuple(a['pool_in_dims'])!=w.output_spatial:return None
    if tuple(tensors[out]['shape']) not in (
        (shape[0]*prod(pool.source.output_hw),co),
        (shape[0],*pool.source.output_hw,co)):return None
    edge,operand_rows,accumulator_rows=_capacities()
    if accumulator_rows<edge:return None
    # One half of the selected operand address space bounds host panel capacity.
    # This does not assign that SP memory to the host; storage is private BSS.
    budget=operand_rows*edge//2
    limit=min(edge*edge,budget//(prod(w.kernel)*ci))//edge*edge
    if limit<edge:return None
    rows=limit if panel_rows is None else panel_rows
    if type(rows) is not int or rows<=0 or rows>limit or rows%edge:return None
    return PanelPlan(w,co,rows,lhs,rhs,out,bias,deepcopy(a),pool)


def _packing_function(builder, plan, rows):
    """Legacy CPU transport for local comparison; never used by emit().

    This host byte-copy path failed accelerator-group placement. Its native
    execution test remains useful as an independent transport comparison only.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import i64,IntegerAttr,StringAttr
    from xdsl.printer import Printer
    from merlin.llvmlower.host_builder import FnBuilder,PTR
    from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel
    owner=(builder.task_index,builder.source_op_index,builder.region_id)
    symbol='conv_pool_pack_'+sha256(repr((plan.window,rows,owner)).encode()).hexdigest()[:20]
    cache=getattr(builder,'_conv_pool_pack_functions',None)
    if cache is None:builder._conv_pool_pack_functions=cache={}
    if symbol in cache:return symbol,cache[symbol]
    host=FnBuilder([PTR,PTR,i64]);source,dest,start=host.entry.args
    proof=emit_nhwc_window_panel(host,plan.window,source,dest,start,rows,
        source_destination_disjoint=True,maximum_panel_bytes=plan.panel_bytes)
    fn=llvm.FuncOp(symbol,llvm.LLVMFunctionType([PTR,PTR,i64],llvm.LLVMVoidType()),body=host.finish())
    # Preparation remains attached to the original source owner, including
    # its ordinary CFG. Distinct owners get distinct helpers rather than
    # borrowing another source operation's attribution on a cache hit.
    for operation in fn.walk():
        if operation is fn or operation.name=='llvm.return':continue
        if builder.task_index is not None:operation.attributes['merlin.global_task']=IntegerAttr(builder.task_index,i64)
        if builder.source_op_index is not None:operation.attributes['merlin.source_op_index']=IntegerAttr(builder.source_op_index,i64)
        if builder.region_id is not None:operation.attributes['prov.region_id']=StringAttr(str(builder.region_id))
    fn.verify();stream=StringIO();Printer(stream=stream).print_op(fn)
    builder.globals.append(stream.getvalue());cache[symbol]=proof
    return symbol,proof


def emit_dma_panel(builder, plan, source, destination, start, rows, zero, *, private_scratchpad=False):
    """Pack original ky/kx/ci slices using checked runs and real DMA only.

    The private ACC tile is borrowed between completed dense panels. Its input
    bytes are sign-extended, then read out with exact identity quantization.
    Each source slice writes its final panel bytes once; no unfinished zero
    fill is overwritten. Source payloads never enter host SSA.
    """
    from dma_window_transport import emit_if_enabled
    byte_transport=emit_if_enabled(builder,plan,source,destination,start,rows,zero,private_scratchpad=private_scratchpad)
    if byte_transport is not None:return byte_transport
    from convolution import _capacities
    from isa import ACC
    from merlin.llvmlower.requantization import quantized
    edge,_,accumulator_rows=_capacities()
    if not 0<rows<=plan.rows or edge>accumulator_rows:raise ValueError('DMA panel exceeds admitted private tile')
    if any(quantized(x,(1.0,),False)!=x for x in range(-128,128)):
        raise ValueError('identity readout does not preserve signed i8')
    ci=plan.window.input_shape[-1];kh,kw=plan.window.kernel
    tile=min(edge,plan.window.output_spatial[1])
    builder.fence()
    builder.config_st(plan.reduction,activation=0,scale=1.0)
    def rectangle(offset,count):
        logical=builder.op('add',start,offset)
        def ky_body(ky):
            def kx_body(kx):
                def channel_body(channel,depth):
                    reduction=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',ky,kw),kx),ci),channel)
                    def transfer(item,local_offset,nrows,padded):
                        builder.config_ld(0 if padded else item['row_stride_elements'],shrink=True,id=2)
                        pointer=zero if padded else builder.ptr_offset(source,item['first_element'])
                        builder.mvin(pointer,ACC,cols=depth,rows=nrows,id=2)
                        builder.fence()
                        target=builder.ptr_offset(destination,builder.op('add',builder.op('mul',builder.op('add',offset,local_offset),plan.reduction),reduction))
                        builder.mvout(target,ACC,cols=depth,rows=nrows)
                        builder.fence()
                    run=plan.window.emit_row_run(builder,logical,count,ky,kx,channel,depth)
                    def boundary():
                        def one(i):
                            item=plan.window.emit_row_run(builder,builder.op('add',logical,i),1,ky,kx,channel,depth)
                            builder.branch(item['predicate'],lambda:transfer(item,i,1,False),lambda:transfer(item,i,1,True))
                        builder.loop(0,count,1,one)
                    builder.branch(run['predicate'],lambda:transfer(run,0,count,False),boundary)
                whole,tail=divmod(ci,edge)
                if whole:builder.loop(0,whole,1,lambda c:channel_body(builder.op('mul',c,edge),edge))
                if tail:channel_body(whole*edge,tail)
            builder.loop(0,kw,1,kx_body)
        builder.loop(0,kh,1,ky_body)
    whole,tail=divmod(rows,tile)
    if whole:builder.loop(0,whole,1,lambda i:rectangle(builder.op('mul',i,tile),tile))
    if tail:rectangle(whole*tile,tail)
    builder.fence()
    return dict(route='window_row_runs_dma_identity_panel',source_window=repr(plan.window),
        rows=rows,row_tile=tile,channel_tile=edge,private_accumulator_rows=edge,
        original_reduction_order='kernel_y,kernel_x,input_channel',
        transport='signed i8 ACC DMA then identity i8 readout into K-strided rows',
        padding='guarded original source run or immutable integer-zero DMA; one final write per byte',
        completion='load completed before readout; readout completed before ACC reuse; panel completed before dense',
        host_payload_operations=0)


@lru_cache(maxsize=64)
def _dma_epoch_schedule(window, rows, edge, accumulator_rows):
    """Admit actual counted phase IR with the shared private-effect checker."""
    from merlin.sched.ir import Kernel,Call,Loop,Const,Var,Scaled,Sum,instances,concretize
    from merlin.compile.scheduling.temporary_storage import (StorageAdapter,
        StorageEffects,TemporaryObject,verify_temporary_storage)
    from merlin.compile.scheduling.program_limits import EncodedAccess
    from merlin.semantic_compiler.allocation_types import StorageBank
    from merlin.llvmlower.requantization import quantized
    if any(quantized(x,(1.0,),False)!=x for x in range(-128,128)):
        raise ValueError('identity readout does not preserve signed i8')
    ci=window.input_shape[-1]
    if ci>edge and ci%edge:raise ValueError('epoch slices require a single static channel depth')
    depth=min(ci,edge);chunks=(ci+edge-1)//edge
    tile=min(edge,window.output_spatial[1],rows)
    total=prod(window.kernel)*chunks
    slots=min(total,accumulator_rows//tile,128)
    if slots<2 or total>2048:raise ValueError('DMA epochs lack bounded disjoint capacity')
    body=[]
    for first in range(0,total,slots):
        count=min(slots,total-first)
        args=(('slice',Sum((Const(first),Var('slot')))),('slot',Var('slot')))
        body.extend((Loop('slot',count,(Call('stage_slice',args),)),
            Call('complete_loads',(('first',Const(first)),('count',Const(count)))),
            Loop('slot',count,(Call('store_slice',args),)),
            Call('complete_stores',(('first',Const(first)),('count',Const(count))))))
    kernel=Kernel('dma_window_panel_phases',(),tuple(body))
    # These effects decode the actual macro operands. Each stage expands to
    # either one full-run DMA or disjoint one-row DMAs into precisely these
    # lanes. The subsequent real phase fence completes all those commands.
    def project(call):
        f=concretize(call,{})
        if call.instr.startswith('complete_'):
            prefix='L' if call.instr=='complete_loads' else 'S'
            return StorageEffects(call.instr,completion='wait',waits=tuple(prefix+str(i) for i in range(f['first'],f['first']+f['count'])))
        write=call.instr=='stage_slice';base=f['slot']*tile*edge*4
        accesses=tuple(EncodedAccess('acc',base+r*edge*4,depth*4,write) for r in range(tile))
        return StorageEffects(repr((call.instr,f['slice'],depth,tile)),accesses,
            completion='asynchronous',token=('L' if write else 'S')+str(f['slice']))
    flat=Kernel(kernel.name,(),tuple(Call(call.instr,tuple((k,Const(v)) for k,v in concretize(call,env).items())) for call,env in instances(kernel)))
    objects=tuple(TemporaryObject(f'slot{s}_row{r}','acc',(s*tile+r)*edge*4,depth*4,4)
        for s in range(slots) for r in range(tile))
    facts=(Path(__file__).parent/'facts.json').read_bytes()
    adapter=StorageAdapter(project,lambda call,accesses:call,
        'selected-gemmini:'+sha256(facts).hexdigest(),'signed-i8-identity-v1',
        'window-run-DMA-macros-with-full-phase-fences-v1')
    report=verify_temporary_storage(flat,flat,objects,
        (StorageBank('acc','physical_accumulator',accumulator_rows*edge*4,'bytes',4),),adapter,
        private_ownership=True,max_calls=8192,max_objects=2048)
    return kernel,dict(report, count_ir_sha256=kernel.digest(),slots=slots,
        tile_rows=tile,depth=depth,slices=total,phases=(total+slots-1)//slots,
        configuration='fixed data/zero load configurations and identity store; complete before issue',
        ordering='disjoint pending writes/reads only; complete load and store phase fences',
        scope='shared private ACC effects/initialization/completion; source byte cover checked separately')


def emit_dma_panel_epochs(builder, plan, source, destination, start, rows, zero):
    """Consume the admitted shared IR: all loads, drain, all stores, drain."""
    from convolution import _capacities
    from isa import ACC
    from merlin.compile.scheduling.llvm_program import emit_index_program
    edge,_,accumulator_rows=_capacities()
    kernel,admission=_dma_epoch_schedule(plan.window,rows,edge,accumulator_rows)
    tile=admission['tile_rows'];ci=plan.window.input_shape[-1]
    depth=admission['depth'];chunks=(ci+edge-1)//edge;kw=plan.window.kernel[1]
    # No changing configuration is pending during either data phase.
    builder.fence()
    builder.config_ld(ci*plan.window.stride[1],shrink=True,id=0)
    builder.config_ld(0,shrink=True,id=2)
    builder.config_st(plan.reduction,activation=0,scale=1.0)
    builder.fence()
    def rectangle(offset,count):
        logical=builder.op('add',start,offset)
        def lower(name,a):
            if name.startswith('complete_'):
                builder.fence();return
            tap=builder.op('udiv',a['slice'],chunks)
            ky=builder.op('udiv',tap,kw);kx=builder.op('urem',tap,kw)
            channel=builder.op('mul',builder.op('urem',a['slice'],chunks),edge)
            local=builder.op('add',ACC,builder.op('mul',a['slot'],tile))
            if name=='stage_slice':
                def transfer(item,i,n,padded):
                    pointer=zero if padded else builder.ptr_offset(source,item['first_element'])
                    builder.mvin(pointer,builder.op('add',local,i),cols=depth,rows=n,id=2 if padded else 0)
                run=plan.window.emit_row_run(builder,logical,count,ky,kx,channel,depth)
                def boundary():
                    def one(i):
                        item=plan.window.emit_row_run(builder,builder.op('add',logical,i),1,ky,kx,channel,depth)
                        builder.branch(item['predicate'],lambda:transfer(item,i,1,False),lambda:transfer(item,i,1,True))
                    builder.loop(0,count,1,one)
                builder.branch(run['predicate'],lambda:transfer(run,0,count,False),boundary)
            elif name=='store_slice':
                reduction=builder.op('add',builder.op('mul',tap,ci),channel)
                target=builder.ptr_offset(destination,builder.op('add',builder.op('mul',offset,plan.reduction),reduction))
                builder.mvout(target,local,cols=depth,rows=count)
            else:raise ValueError('unknown admitted DMA phase effect')
        emit_index_program(kernel,builder,lower)
    whole,tail=divmod(rows,tile)
    if whole:builder.loop(0,whole,1,lambda i:rectangle(builder.op('mul',i,tile),tile))
    if tail:
        # Tail uses a separately admitted exact row footprint, never unwritten
        # rows of a full-sized slot. Its ordinary phase emitter is recursive.
        emit_dma_panel_epochs(builder,plan,source,
            builder.ptr_offset(destination,whole*tile*plan.reduction),
            builder.op('add',start,whole*tile),tail,zero)
    builder.fence()
    return dict(route='window_dma_disjoint_phase_epochs',admission=admission,
        source_window=repr(plan.window),rows=rows,host_payload_operations=0,
        completed_before_dense=True)


def emit(builder, command, tensors, pointers, *, residents=None, panel_rows=None,
         dense_schedule_options=None, enabled=None, packing_epochs=None):
    """Emit actual prepared dense panels and original Q1→hardware max chain."""
    mode=selected()
    if enabled is None:enabled=mode!='off'
    if not enabled:return None
    plan=prepare(command,tensors,pointers,residents=residents,panel_rows=panel_rows)
    if plan is None:return None
    if mode=='dma_epochs_small_geometry':
        from conv_pool_panel_policy import select
        choice=select(plan,emit_route=lambda:_emit_prepared(builder,plan,pointers,
            dense_schedule_options=dense_schedule_options,packing_epochs=True))
        actual=choice.pop('actual')
        if actual is None:return None
        actual['selection']=choice
        return actual
    if packing_epochs is None:packing_epochs=mode=='packed_dense_dma_epochs'
    if packing_epochs:
        try:
            from convolution import _capacities
            edge,_,accumulator_rows=_capacities()
            _dma_epoch_schedule(plan.window,min(plan.rows,plan.output_rows),edge,accumulator_rows)
        except ValueError:
            packing_epochs=False
    return _emit_prepared(builder,plan,pointers,
        dense_schedule_options=dense_schedule_options,packing_epochs=packing_epochs)


def _emit_prepared(builder,plan,pointers,*,dense_schedule_options,packing_epochs):
    from dense import lower_dense
    from conv_pool_policy import emit_quantized_once
    panel=builder.global_buffer(plan.panel_bytes,'i8')
    image=builder.global_buffer(plan.q1_bytes,'i8')
    from convolution import _capacities
    zero=builder.global_buffer(_capacities()[0],'i8')
    attrs={k:deepcopy(v) for k,v in plan.attrs.items() if not k.startswith('pool_')}
    attrs['epilogue']=['bias_add' if s=='bias' else s for s in attrs.get('epilogue',()) if s!='maxpool']
    transport=[];products=[]
    def producer(q1):
        def run(start,rows):
            pack=emit_dma_panel_epochs if packing_epochs else emit_dma_panel
            transport.append(pack(builder,plan,pointers[plan.lhs],panel,start,rows,zero))
            output=builder.ptr_offset(q1,builder.op('mul',start,plan.channels))
            products.append(lower_dense(builder,panel,pointers[plan.rhs],output,rows,plan.reduction,plan.channels,
                attrs=attrs,bias_ptr=pointers[plan.bias] if plan.bias else None,
                schedule_options=dense_schedule_options))
            # Complete all DMA reads before overwriting the bounded host panel.
            builder.fence()
        whole,tail=divmod(plan.output_rows,plan.rows)
        if whole:builder.loop(0,whole,1,lambda i:run(builder.op('mul',i,plan.rows),plan.rows))
        if tail:run(whole*plan.rows,tail)
        return dict(route='library_prepared_dense_panels',source_window=repr(plan.window),
            original_attributes=plan.attrs,panel_rows=plan.rows,panel_bytes=plan.panel_bytes,
            original_reduction_order='kernel_y,kernel_x,input_channel',
            products=products,transport=transport,
            source_partition=dict(rows=plan.output_rows,whole=whole,tail=tail),
            preparation='checked window runs; DMA sign extension and exact identity readout; no host payload arithmetic')
    receipt=emit_quantized_once(builder,plan.pool,intermediate=image,output=pointers[plan.output],
        workspace_bytes=plan.q1_bytes,emit_producer=producer)
    # A literal completion after the final loop's exit dominates kernel return.
    # Completion is therefore visible even to consumers that do not prove a
    # counted loop's positive trip count or inspect its nested store fence.
    builder.fence()
    return dict(route='conv_pool_packed_dense_panels',panel=receipt,
        source_quantization='unchanged original bias/f32 scale/rounding/relu once per producer cell before maximum',
        workspace_bytes=plan.panel_bytes+plan.q1_bytes+_capacities()[0],target_execution=False,timing=None)
