"""Explicit packed source-K convolution through checked transport and dense IR.

The only new composition here is window DMA -> bounded row-major K panel ->
existing dense accelerator. No input payload is loaded into host arithmetic.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import prod, isfinite
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import Window2D

VALUES=('off','dma_packed_k')
DECLARATION=dict(values=list(VALUES),mechanism='checked window DMA packing into complete original K followed by shared dense accelerator',note='Default off. Source ordering and quantization unchanged; complete-call timing includes panel transport, dense work and original pool.')


def selected():
    return knob('conv_packed_k',default='off',values=VALUES,root=Path(__file__).parent)


@dataclass(frozen=True)
class PackedPlan:
    window: object
    channels: int
    rows: int
    lhs: str
    rhs: str
    output: str
    bias: str | None
    attrs: dict
    @property
    def reduction(self):return prod(self.window.kernel)*self.window.input_shape[-1]
    @property
    def output_rows(self):return self.window.input_shape[0]*prod(self.window.output_spatial)
    @property
    def panel_bytes(self):return self.rows*self.reduction


def prepare(command,tensors,pointers,*,residents=None):
    """Return an admitted raw/scaled source convolution, with specific refusal."""
    from convolution import _capacities
    if command.get('opcode') not in ('CONV','CONV2D'):raise ValueError('not a convolution')
    a=deepcopy(command.get('attributes',{}));o=command.get('operands',{})
    known={'kernel','stride','padding','dilation','layout','output_dtype','epilogue','acc_scale','acc_scale_by_column','bias','input_zero_point','weight_zero_point','output_zero_point'}
    # Provenance is metadata, never an additional numerical permission.
    provenance={'source_task_index','source_op_indices','source_primary_op_index','source_payload_index'}
    semantic=set(a)-provenance
    if semantic-known:raise ValueError('unmodeled source convolution attributes: '+str(sorted(semantic-known)))
    if set(o)-{'ifm','input','weight','dst','bias'}:raise ValueError('unmodeled convolution operands')
    if 'ifm' in o and 'input' in o and o['ifm']!=o['input']:raise ValueError('ambiguous source input')
    lhs=o.get('ifm',o.get('input'));rhs=o.get('weight');dst=o.get('dst')
    seen=set()
    while rhs in (residents or {}):
        if rhs in seen:raise ValueError('cyclic weight handle')
        seen.add(rhs);rhs=residents[rhs]
        if isinstance(rhs,dict):rhs=rhs.get('src',rhs.get('source'))
    if any(n not in tensors or n not in pointers for n in (lhs,rhs,dst)):raise ValueError('unmapped convolution value')
    shape=tuple(tensors[lhs]['shape']);kernel=tuple(a.get('kernel',()))
    if len(shape)!=4 or str(a.get('layout','nhwc')).lower()!='nhwc':raise ValueError('requires NHWC source')
    if len(kernel) not in (2,4):raise ValueError('requires static KH,KW[,CI,CO]')
    co=tensors[dst]['shape'][-1];ci=shape[-1]
    if any(type(v) is not int or v<=0 for v in (*shape,*kernel,co)):raise ValueError('requires positive static dimensions')
    if len(kernel)==4 and kernel[2:]!=(ci,co):raise ValueError('source kernel channels mismatch')
    if tensors[lhs]['dtype']!='i8' or tensors[rhs]['dtype']!='i8':raise ValueError('requires signed i8 operands')
    if tuple(tensors[rhs]['shape']) not in ((prod(kernel[:2])*ci,co),(*kernel[:2],ci,co)):raise ValueError('requires contiguous HWIO weights')
    if pointers[lhs]==pointers[dst] or pointers[rhs]==pointers[dst]:raise ValueError('source and output must be disjoint')
    if any(type(a[k]) is not int or a[k]!=0 for k in ('input_zero_point','weight_zero_point','output_zero_point') if k in a):raise ValueError('requires integer zero points zero')
    stages=['bias_add' if s=='bias' else s for s in a.get('epilogue',())]
    canonical=['bias_add','acc_scale','relu']
    if stages!=[s for s in canonical if s in stages]:raise ValueError('source epilogue order unsupported')
    bias=a.get('bias',o.get('bias'))
    if 'bias' in a and 'bias' in o and a['bias']!=o['bias']:raise ValueError('ambiguous source bias')
    if bool(bias)!=('bias_add' in stages):raise ValueError('bias stage and initializer must agree')
    if bias and (bias not in tensors or bias not in pointers or tensors[bias]['dtype']!='i32' or prod(tensors[bias]['shape'])!=co):raise ValueError('requires i32 per-column bias')
    dtype=tensors[dst]['dtype']
    if dtype not in ('i8','i32') or a.get('output_dtype',dtype)!=dtype:raise ValueError('requires original i8/i32 output')
    scales=a.get('acc_scale_by_column')
    if scales is not None and (len(scales)!=co or 'acc_scale' in a):raise ValueError('ambiguous per-column scale')
    scales=scales if scales is not None else [a.get('acc_scale',1.0)]
    if any(not isfinite(float(s)) for s in scales):raise ValueError('nonfinite source gain')
    if dtype=='i32' and (any(s in stages for s in ('acc_scale','relu')) or any(float(s)!=1.0 for s in scales)):raise ValueError('raw output cannot apply narrowing stages')
    if any(float(s)!=1.0 for s in scales) and 'acc_scale' not in stages:raise ValueError('source scale lacks epilogue stage')
    w=Window2D(shape,kernel[:2],tuple(a.get('stride',(1,1))),tuple(a.get('dilation',(1,1))),tuple(a.get('padding',(0,0,0,0))))
    if prod(tensors[dst]['shape'])!=shape[0]*prod(w.output_spatial)*co:raise ValueError('source output extent mismatch')
    edge,operand_rows,accumulator_rows=_capacities()
    if accumulator_rows<edge:raise ValueError('private packing ACC tile unavailable')
    reduction=prod(w.kernel)*ci
    # Bound the matrix panel independently of static output size; dense planning
    # separately checks actual SP/ACC reservations and every padded tail.
    rows=min(edge*edge,(operand_rows*edge//2)//reduction)//edge*edge
    if rows<edge:raise ValueError('one primitive panel exceeds bounded preparation budget')
    if ci>edge and ci%edge:raise ValueError('packing epoch requires uniform channel depth')
    from conv_pool_panel import _dma_epoch_schedule
    _dma_epoch_schedule(w,min(rows,shape[0]*prod(w.output_spatial)),edge,accumulator_rows)
    a['epilogue']=stages;a['output_dtype']=dtype
    return PackedPlan(w,co,rows,lhs,rhs,dst,bias,a)


def emit(builder,command,tensors,pointers,*,residents=None,enabled=None,dense_schedule_options=None):
    if enabled is None:enabled=selected()!='off'
    if not enabled:return None
    # Pool keeps the already checked original Q1 -> spatial maximum route.
    if 'maxpool' in command.get('attributes',{}).get('epilogue',()):
        # An admitted ring owns this complete consumer. Keep its incumbent
        # producer until it explicitly binds the packed-K producer seam.
        from spatial_readout_cache_adapter import admit_source
        if admit_source(command,tensors,pointers,residents=residents) is not None:return None
        from conv_pool_panel import prepare as prepare_pool,_emit_prepared,_dma_epoch_schedule
        plan=prepare_pool(command,tensors,pointers,residents=residents)
        if plan is None:return None
        from convolution import _capacities
        edge,_,accumulator_rows=_capacities()
        try:_dma_epoch_schedule(plan.window,min(plan.rows,plan.output_rows),edge,accumulator_rows)
        except ValueError:return None
        result=_emit_prepared(builder,plan,pointers,packing_epochs=True,dense_schedule_options=dense_schedule_options)
        result=dict(result,packed_k_producer=True)
    else:
        try:plan=prepare(command,tensors,pointers,residents=residents)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            if not hasattr(builder,'conv_packed_k_refusals'):builder.conv_packed_k_refusals=[]
            builder.conv_packed_k_refusals.append(str(error));return None
        from dense import lower_dense
        from conv_pool_panel import emit_dma_panel_epochs
        from convolution import _capacities
        panel=builder.global_buffer(plan.panel_bytes,'i8');zero=builder.global_buffer(_capacities()[0],'i8')
        transport=[];products=[]
        def run(start,rows):
            transport.append(emit_dma_panel_epochs(builder,plan,pointers[plan.lhs],panel,start,rows,zero))
            products.append(lower_dense(builder,panel,pointers[plan.rhs],builder.ptr_offset(pointers[plan.output],builder.op('mul',start,plan.channels*(4 if plan.attrs['output_dtype']=='i32' else 1))),rows,plan.reduction,plan.channels,attrs=plan.attrs,bias_ptr=pointers[plan.bias] if plan.bias else None,schedule_options=dense_schedule_options))
            builder.fence()
        whole,tail=divmod(plan.output_rows,plan.rows)
        if whole:builder.loop(0,whole,1,lambda i:run(builder.op('mul',i,plan.rows),plan.rows))
        if tail:run(whole*plan.rows,tail)
        builder.fence()
        result=dict(route='source_packed_k_dense',source_window=repr(plan.window),source_attributes=plan.attrs,panel_rows=plan.rows,panel_bytes=plan.panel_bytes,reduction_elements=plan.reduction,reduction_order='kernel_y,kernel_x,input_channel',padded_reduction_elements=((plan.reduction+_capacities()[0]-1)//_capacities()[0])*_capacities()[0],source_partition=dict(rows=plan.output_rows,whole=whole,tail=tail),transport=transport,products=products,host_payload_operations=0,completion='each packed panel completed before dense; dense completed before panel reuse',timing=None)
    if not hasattr(builder,'conv_pool_panel_receipts'):builder.conv_pool_panel_receipts=[]
    builder.conv_pool_panel_receipts.append(dict(mechanism='source_packed_k',actual=result))
    return result
