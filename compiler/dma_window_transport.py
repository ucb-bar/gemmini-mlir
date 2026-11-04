"""Default-off exact source-window/halo transport through one private SP tile.

Admits only a complete kernel1 halo preparation before its consumer begins.
No owner is invalidated: a live retained scratchpad context refuses this route.
"""
from functools import lru_cache
from math import prod
from pathlib import Path

VALUES=('off','scratchpad_bytes')
DECLARATION=dict(values=list(VALUES),mechanism='exact source-owned signed-i8 halo transport through private scratchpad rows0..15',note='Default off; complete kernel1 preparation only; refuse live retained scratchpad state; existing row guards and every completion retained.')

def selected():
    from merlin.compile.tuning_knobs import knob
    return knob('dma_window_transport',default='off',values=VALUES,root=Path(__file__).parent)

@lru_cache(maxsize=256)
def storage_proof(rows=16,columns=16):
    if type(rows) is not int or type(columns) is not int or not 0<rows<=16 or not 0<columns<=16:raise ValueError('SP descriptor exceeds selected tile extents')
    from elementwise_tiles import _selected
    from merlin.sched.ir import Kernel,Call,Const
    from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
    from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
    from merlin.compile.scheduling.program_limits import EncodedAccess
    from merlin.semantic_compiler.allocation_types import StorageBank
    spaces,ordering,facts=_selected(str(Path(__file__).parent))
    # Actual descriptor subset of one16-column,16-row tile.
    TileLoadCoalescing(rows,columns,16,16,0,16,spaces['scratchpad']['bytes']//16)
    kernel=Kernel('private_window_byte_tile',(),(Call('load',(('rows',Const(rows)),('columns',Const(columns)))),Call('load_complete',()),Call('store',(('rows',Const(rows)),('columns',Const(columns)))),Call('store_complete',())))
    def project(call):
        if call.instr.endswith('_complete'):
            return StorageEffects(call.instr,completion='wait',waits=(call.instr.split('_')[0],))
        write=call.instr=='load'
        return StorageEffects(call.instr,tuple(EncodedAccess('sp',r*16,columns,write) for r in range(rows)),completion='asynchronous',token=call.instr)
    adapter=StorageAdapter(project,lambda call,accesses:call,'selected:'+facts,'i8-byte-identity','selected-SP-block-law/full-fences:'+ordering)
    proof=verify_temporary_storage(kernel,kernel,tuple(TemporaryObject('row'+str(r),'sp',r*16,columns,1) for r in range(rows)),
        (StorageBank('sp','physical_scratchpad',spaces['scratchpad']['bytes'],'bytes',16),),adapter,private_ownership=True)
    return dict(storage=proof,ir_sha256=kernel.digest(),facts_sha256=facts,ordering_sha256=ordering)

def admit(builder,plan,source,destination,start,rows,zero,*,private_scratchpad=False):
    if private_scratchpad is not True:raise ValueError('window SP requires explicit source-stage private ownership before any RHS load')
    from convolution import _capacities
    edge,operand_rows,_=_capacities()
    if edge!=16 or operand_rows<16:raise ValueError('window SP tile requires selected16-row capacity')
    if plan.window.kernel!=(1,1) or start!=0 or type(rows) is not int or rows!=plan.output_rows or plan.rows<rows:
        raise ValueError('window SP ownership requires complete kernel1 halo preparation at start0 before consumer')
    if plan.window.input_shape[-1]>16:raise ValueError('window SP halo requires one primitive channel slice')
    if source==destination or zero==destination:raise ValueError('window SP source, immutable zero and destination must be disjoint')
    if max(prod(plan.window.input_shape),rows*plan.reduction)>=1<<63:raise ValueError('window SP tensor byte range exceeds signed i64')
    owner=builder;seen=set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if getattr(owner,'dense_residency_state',None) is not None:
            raise ValueError('window SP tile ownership conflicts with live retained RHS')
        if not hasattr(owner,'builder'):break
        owner=owner.builder
    tile=min(edge,plan.window.output_spatial[1]);depth=plan.window.input_shape[-1]
    proof=[storage_proof(n,depth) for n in sorted({1,tile,rows%tile}-{0})]
    return dict(schema='authored.source_window_scratchpad_transport.v1',source_window=repr(plan.window),
        source_bytes=prod(plan.window.input_shape),destination_bytes=rows*plan.reduction,
        external_windows={'source':{'base':repr(source),'bytes':prod(plan.window.input_shape)},'destination':{'base':repr(destination),'bytes':rows*plan.reduction},'immutable_zero':{'base':repr(zero),'bytes':edge}},
        source_owner=dict(task=getattr(builder,'task_index',None),source_op=getattr(builder,'source_op_index',None),region=getattr(builder,'region_id',None)),
        ownership='complete kernel1 preparation before consumer; no retained RHS; full fence on entry/exit; rows0..15 private during each copy',
        proof=proof,quantization='none; original signed-i8 bytes and exact zero unchanged',cycles='UNKNOWN',physical_qualification='not_established')

def emit_if_enabled(builder,plan,source,destination,start,rows,zero,*,private_scratchpad=False):
    if selected()=='off':return None
    try:admit(builder,plan,source,destination,start,rows,zero,private_scratchpad=private_scratchpad)
    except ValueError as error:
        if not hasattr(builder,'dma_window_transport_refusals'):builder.dma_window_transport_refusals=[]
        builder.dma_window_transport_refusals.append(str(error));return None
    return emit(builder,plan,source,destination,start,rows,zero,private_scratchpad=private_scratchpad)

def emit(builder, plan, source, destination, start, rows, zero,*,private_scratchpad=False):
    """Pack original ky/kx/ci slices using checked runs and real DMA only.

    The explicit source-stage caller owns scratchpad rows0..15 before
    any consumer RHS load. Input bytes are stored without quantization.
    Each source slice writes its final panel bytes once; no unfinished zero
    fill is overwritten. Source payloads never enter host SSA.
    """
    admission = admit(builder, plan, source, destination, start, rows, zero,private_scratchpad=private_scratchpad)
    edge=16
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
                        builder.config_ld(0 if padded else item['row_stride_elements'],shrink=False,id=0)
                        pointer=zero if padded else builder.ptr_offset(source,item['first_element'])
                        builder.mvin(pointer,0,cols=depth,rows=nrows,id=0)
                        builder.fence()
                        target=builder.ptr_offset(destination,builder.op('add',builder.op('mul',builder.op('add',offset,local_offset),plan.reduction),reduction))
                        builder.mvout(target,0,cols=depth,rows=nrows)
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
    return dict(admission, route='window_row_runs_scratchpad_byte_panel',source_window=repr(plan.window),
        rows=rows,row_tile=tile,channel_tile=edge,private_scratchpad_rows=edge,
        original_reduction_order='kernel_y,kernel_x,input_channel',
        transport='exact signed-i8 scratchpad bytes; no ACC conversion or readout quantization',
        padding='guarded original source run or immutable integer-zero DMA; one final write per byte',
        completion='load completed before readout; readout completed before scratchpad reuse; panel completed before dense',
        host_payload_operations=0)
