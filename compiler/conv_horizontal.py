"""Opt-in bounded horizontal source-K convolution with explicit completion.

The padded NHWC source is materialized once by existing DMA-only zero-halo
transport. The catalogue owns horizontal window runs, original K decomposition
and resident RHS. This first binding uses one M block: every array launch is
fresh and synchronous, so it grants neither weight-stay nor overlap semantics.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
from hashlib import sha256
from math import prod
from pathlib import Path
import json
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import Window2D,CountedConvolutionPlan
from merlin.compile.scheduling.horizontal_convolution import plan_horizontal_convolution_panels,emit_horizontal_convolution_panels
from isa import ACC,ACCUMULATE,FULL,GARBAGE,DIM

VALUES=('off','horizontal_fresh','window_ring_fresh')
DECLARATION=dict(values=list(VALUES),mechanism='catalogue horizontal_convolution source-K panels with selected multiblock addresses and synchronous fresh-array products',note='Default off. One source-owned row block; all halo DMA, Q1 materialization and original hardware pooling included. No timing claim or weight-stay authority.')
def selected():return knob('conv_horizontal',default='off',values=VALUES,root=Path(__file__).parent)

@lru_cache(maxsize=1)
def _load_law():
    root=Path(__file__).parent;s=json.loads((root/'command_ordering.json').read_text());r=s['ordering'];facts=(root/'facts.json').read_bytes()
    if sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()!=s['ordering_sha256'] or sha256(facts).hexdigest()!=s['base_facts']['sha256'] or r['status']!='derived':raise ValueError('horizontal load law requires intact selected facts/ordering')
    e=r['extents']['MVIN.spad_addr'];b=e['blocks']
    if e['status']!='derived' or e['rows']['offset']!=48 or e['rows']['width']!=5 or b['block']!=DIM or b['cols']['offset']!=32 or b['cols']['width']!=7 or b['stride']['offset']!=16 or b['stride']['width']!=16 or b['stride']['state']!={'operand':'rs1','offset':3,'width':2,'value':0}:
        raise ValueError('horizontal source load lacks selected MVIN multiblock extent/stride law')
    from dense_wide_loads import PUBLIC_REFERENCE_EVIDENCE
    return dict(ordering_sha256=s['ordering_sha256'],facts_sha256=sha256(facts).hexdigest(),decoded_load_extent=e,maximum_columns=64,public_reference=PUBLIC_REFERENCE_EVIDENCE,
        completion='each actual DMA and fresh PRELOAD/COMPUTE pair fully completes; no same/cross queue ordering inferred',array='one M block; fresh_weights always True; no stay operation emitted')

@dataclass(frozen=True)
class HorizontalPlan:
    source:object
    panels:object
    halo:object
    pool:object
    initializer:bool
    law:dict
    @property
    def padded_bytes(self):return prod(self.panels.padded_window.input_shape)
    def report(self):return dict(route='catalogue_horizontal_convolution_fresh',source_window=repr(self.source.window),source_attributes=self.source.attrs,panels=self.panels.summary(),padded_input_bytes=self.padded_bytes,initializer=self.initializer,load_law=self.law,host_payload_operations=0,timing='UNKNOWN')

def prepare(command,tensors,pointers,*,residents=None):
    from convolution import _capacities
    from conv_policy import accumulator_init_selection
    pool='maxpool' in command.get('attributes',{}).get('epilogue',())
    if pool:
        from conv_pool_panel import prepare as source_prepare
        source=source_prepare(command,tensors,pointers,residents=residents)
        if source is None:raise ValueError('original pooled source/numeric admission refused')
    else:
        from conv_packed_k import prepare as source_prepare
        source=source_prepare(command,tensors,pointers,residents=residents)
    if source.attrs.get('acc_scale_by_column') is not None:raise ValueError('horizontal scalar binding does not yet consume per-column exits')
    w=source.window;d,oprows,accrows=_capacities()
    if w.output_spatial[1]<d:raise ValueError('horizontal catalogue requires row_tile16 but original CountedConvolutionPlan forbids crossing output rows narrower than16')
    if w.dilation[1]!=1:raise ValueError('horizontal source K requires contiguous horizontal dilation1')
    if w.input_shape[-1]>d:raise ValueError('initial horizontal halo binding requires CI at most one primitive')
    if w.kernel[1]*w.input_shape[-1]>64:raise ValueError('horizontal source run exceeds selected/public64-byte transfer limit')
    init=accumulator_init_selection();explicit=bool(source.bias) or not source.bias and (init=='zero_accumulate' or init=='zero_single_slice' and source.reduction<=d)
    p=CountedConvolutionPlan(w,source.channels,d,d,d,min(d,w.input_shape[-1]),oprows,accrows,0,d,0,explicit)
    horizontal=plan_horizontal_convolution_panels(p,max_row_blocks=1,maximum_transfer_columns=64,private_rows_owned=True)
    if horizontal is None:raise ValueError('horizontal source geometry/private capacity refused')
    if horizontal.row_blocks!=1:raise ValueError('no selected multi-M fresh/stay law in this binding')
    halo_window=Window2D(w.input_shape,(1,1),(1,1),(1,1),w.padding)
    # Padding changes only address ownership; every byte is original source or
    # exact integer zero, with identity signed-i8 DMA readout.
    halo=replace(source,window=halo_window,rows=prod(horizontal.padded_window.input_shape[:3]))
    law=_load_law()
    return HorizontalPlan(source,horizontal,halo,source.pool if pool else None,explicit,law)

def _emit_product(builder,plan,pointers,padded,output):
    source=plan.source;co=source.channels;dtype=source.attrs['output_dtype'];width=4 if dtype=='i32' else 1
    scale=float(source.attrs.get('acc_scale',1.0));activation=int('relu' in source.attrs.get('epilogue',()))
    builder.fence();builder.config_ex(dataflow=1)
    if plan.initializer:builder.config_ld(0,id=2)
    builder.config_st(co*width,activation=activation,scale=scale)
    zero=builder.constant_i32_table([0]*DIM) if plan.initializer and not source.bias else None
    def lower(kind,f):
        if kind=='load_weight':
            builder.config_ld(co,id=1)
            ptr=builder.ptr_offset(pointers[source.rhs],builder.op('add',builder.op('mul',f['reduction_row'],co),f['output_column']))
            builder.mvin(ptr,f['local_row'],cols=f['columns'],rows=f['rows'],id=1)
        elif kind=='load_horizontal_input':
            if f['columns']>64 or f['block_stride_rows']!=DIM:raise ValueError('actual horizontal load leaves selected multiblock law')
            builder.config_ld(f['row_stride_elements'],block_stride=f['block_stride_rows'],id=0)
            builder.mvin(builder.ptr_offset(padded,f['first_element']),f['local_row'],cols=f['columns'],rows=f['rows'],id=0)
        elif kind=='initialize_bias':
            ptr=builder.ptr_offset(pointers[source.bias],builder.op('mul',f['output_column'],4)) if source.bias else zero
            builder.mvin(ptr,builder.op('or',ACC,f['accumulator_row']),cols=f['columns'],rows=f['rows'],id=2)
        elif kind=='compute':
            if f['fresh_weights'] is not True:raise ValueError('horizontal binding cannot emit unqualified array stay')
            continuation=plan.initializer or not f['first_reduction']
            acc=builder.op('or',ACC|(ACCUMULATE if continuation else 0),f['accumulator_row'])
            builder.preload(f['rhs_row'],acc,b_cols=f['columns'],b_rows=f['depth'],c_cols=f['columns'],c_rows=f['rows'])
            builder.compute(f['lhs_row'],GARBAGE,a_cols=f['depth'],a_rows=f['rows'],bd_cols=f['columns'],bd_rows=f['rows'])
        elif kind=='store':
            ptr=builder.ptr_offset(output,builder.op('mul',builder.op('add',builder.op('mul',f['output_row'],co),f['output_column']),width))
            builder.mvout(ptr,builder.op('or',ACC|(FULL if dtype=='i32' else 0),f['accumulator_row']),cols=f['columns'],rows=f['rows'])
        else:raise ValueError('unknown horizontal catalogue effect '+kind)
        builder.fence()
    result=emit_horizontal_convolution_panels(plan.panels,builder,lower,lambda kind,fields:builder.fence())
    builder.fence();return result

def emit(builder,command,tensors,pointers,*,residents=None):
    if selected()=='off':return None
    if selected()=='window_ring_fresh':
        from conv_window_reuse import emit as emit_reuse
        return emit_reuse(builder,command,tensors,pointers,residents=residents)
    try:plan=prepare(command,tensors,pointers,residents=residents)
    except (ValueError,KeyError,TypeError,IndexError) as error:
        if not hasattr(builder,'conv_horizontal_refusals'):builder.conv_horizontal_refusals=[]
        builder.conv_horizontal_refusals.append(str(error));return None
    from conv_pool_panel import emit_dma_panel
    padded=builder.global_buffer(plan.padded_bytes,'i8');zero=builder.constant_i32_table([0]*((DIM+3)//4))
    transport=emit_dma_panel(builder,plan.halo,pointers[plan.source.lhs],padded,0,plan.halo.rows,zero,private_scratchpad=True)
    if plan.pool is None:
        actual=_emit_product(builder,plan,pointers,padded,pointers[plan.source.output])
    else:
        from conv_pool_policy import emit_quantized_once
        qbytes=plan.source.output_rows*plan.source.channels;q1=builder.global_buffer(qbytes,'i8')
        actual=emit_quantized_once(builder,plan.pool,intermediate=q1,output=pointers[plan.source.output],workspace_bytes=qbytes,
            emit_producer=lambda intermediate:_emit_product(builder,plan,pointers,padded,intermediate))
    builder.fence();receipt=dict(plan.report(),transport=transport,actual=actual)
    owner=builder
    while hasattr(owner,'builder'):owner=owner.builder
    if not hasattr(owner,'conv_pool_panel_receipts'):owner.conv_pool_panel_receipts=[]
    owner.conv_pool_panel_receipts.append(dict(mechanism='conv_horizontal',actual=receipt))
    return receipt
