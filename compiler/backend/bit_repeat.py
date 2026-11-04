"""Exact repeated dense f32 blocks using one raw load and multiple raw stores.

One singleton axis is expanded. Row-axis expansion retains a single source row;
column-axis expansion retains a one-column strip with independent source and
destination row pitches. No payload arithmetic or CPU copies are permitted.
"""
from dataclasses import dataclass
from math import prod
from merlin.sched.ir import Kernel, Loop, TensorArg, Const, Var, Sum, Scaled, Call
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa
from .bit_slice import _call, _strides

class BitRepeatError(ValueError): pass

def _geometry(shape,axis,repeats):
    shape=tuple(shape)
    if (not 2<=len(shape)<=6 or any(type(d) is not int or d<=0 for d in shape)
        or type(axis) is not int or not 0<=axis<len(shape) or shape[axis]!=1
        or type(repeats) is not int or repeats<=1 or shape[-1]*4>=1<<32
        or prod(shape)*repeats*4>=1<<63):
        raise BitRepeatError('repeat requires static dense f32 and one singleton expansion')
    output=list(shape);output[axis]=repeats
    if output[-1]*4>=1<<32:
        raise BitRepeatError('repeat destination row pitch exceeds selected configuration field')
    return shape,tuple(output)


def _store_epoch_admission():
    """Consume the selected packaged queue facts; unknown facts retain drains."""
    import json,hashlib
    from pathlib import Path
    try:
        path=Path(__file__).with_name('conv_ordering.json');raw=path.read_bytes();record=json.loads(raw)
        ordering=record['ordering'];digest=hashlib.sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or ordering['status']!='derived'
            or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or ordering['same_queue']['store']['by_class']['MVOUT']!='completion'
            or ordering['completion_order']['store']['order']!='in_order'):return None
        queues={}
        for row in ordering['funct_queue']:
            if 'by_code' in row:queues.update({('config',int(k)):v for k,v in row['by_code'].items()})
            elif 'funct' in row:queues[row['funct']]=row['queue']
        entry=record['finite_entries']['store']
        if (queues.get(isa.FUNCT['mvout'])!='store' or queues.get(('config',2))!='store'
            or entry.get('status')!='derived' or type(entry.get('value')) is not int or entry['value']<1
            or not {'io_alloc_bits_cmd_inst_rs1','io_alloc_bits_cmd_inst_rs2'}.issubset(entry['evidence']['written_from'])):return None
        return {'ordering_sha256':digest,'packaged_record_sha256':hashlib.sha256(raw).hexdigest(),
            'base_facts_sha256':record['base_facts_sha256'],'store_queue_entries':entry['value'],
            'store_release':'completion','store_completion':'in_order',
            'configuration':'one original raw CONFIG_ST, same store lane before all repeated stores; no intervening reconfiguration',
            'finite_queue':'captured rs1/rs2 descriptors; ready/valid backpressure bounds pending stores without bounding total issued stores',
            'retained_lifetime':'real load completion and final store-epoch drain remain; ACC readonly until final drain',
            'consumed_kernel_sha256':None}
    except (OSError,KeyError,TypeError,ValueError):return None

def _kernel(shape,axis,repeats):
    store_epoch=_store_epoch_admission() is not None
    shape,output=_geometry(shape,axis,repeats)
    ss,ds=_strides(shape),_strides(output)
    src=sum((Var(f'd{i}')*ss[i] for i in range(len(shape)-2) if i!=axis),Const(0))
    dst=sum((Var(f'd{i}')*ds[i] for i in range(len(shape)-2) if i!=axis),Const(0))
    def tile(row,col,rows,cols):
        store=(_call('repeat_mvout',offset=dst+Var('rep')*ds[axis]+row*output[-1]+col,
                     addr=isa.acc_addr(0,full=True),rows=rows,cols=cols),)
        if not store_epoch:store+=(_call('repeat_drain'),)
        return (_call('repeat_mvin',offset=src+row*shape[-1]+col,addr=isa.acc_addr(0),rows=rows,cols=cols),
                _call('repeat_drain'),Loop('rep',repeats,store),_call('repeat_drain'))
    def cols(row,rows):
        full,tail=divmod(shape[-1],isa.DIM);body=[]
        if full:body.append(Loop('col',full,tile(row,Var('col')*isa.DIM,rows,isa.DIM)))
        if tail:body.extend(tile(row,Const(full*isa.DIM),rows,tail))
        return tuple(body)
    full,tail=divmod(shape[-2],isa.DIM);body=[]
    if full:body.append(Loop('row',full,cols(Var('row')*isa.DIM,isa.DIM)))
    if tail:body.extend(cols(Const(full*isa.DIM),tail))
    body=tuple(body)
    for i in reversed(range(len(shape)-2)):
        if i!=axis:body=(Loop(f'd{i}',shape[i],body),)
    return Kernel('raw_bit_repeat',(TensorArg('src',shape,'f32','read'),TensorArg('dst',output,'f32','write')),
                  (_call('repeat_drain'),_call('repeat_config'),*body))

def check_repeat(kernel,shape,axis,repeats):
    shape,output=_geometry(shape,axis,repeats)
    # Canonical schedule equality binds the receipt to every emitted call, loop,
    # address, descriptor and drain. The quotient/remainder partition below then
    # proves all events without enumerating tensor payloads.
    if kernel != _kernel(shape,axis,repeats):
        raise BitRepeatError('repeat schedule differs from the proved exact DMA partition')
    def linear(e):
        if isinstance(e,Const):return {'':e.value}
        if isinstance(e,Var):return {e.name:1}
        if isinstance(e,Scaled):return {k:v*e.factor for k,v in linear(e.expr).items()}
        if isinstance(e,Sum):
            out={}
            for term in e.terms:
                for k,v in linear(term).items():out[k]=out.get(k,0)+v
            return {k:v for k,v in out.items() if v}
        raise BitRepeatError('repeat address is not a proved affine expression')
    loaded=stored=sites=0
    def walk(body,scope,load=None):
        nonlocal loaded,stored,sites
        for op in body:
            if isinstance(op,Loop):walk(op.body,scope+((op.var,op.extent),),load)
            elif isinstance(op,Call) and op.instr=='repeat_mvin':
                load=op
                loaded+=prod(n for _,n in scope)*op.arg('rows').value*op.arg('cols').value
            elif isinstance(op,Call) and op.instr=='repeat_mvout':
                if load is None:raise BitRepeatError('repeat store lacks a dominating source tile')
                # Check scalar addresses inside the actual DMA descriptors too:
                # a column repeat uses different row pitches for source/dst.
                source=linear(load.arg('offset')+Var('_lane_row')*shape[-1]+Var('_lane_col'))
                delta=linear(op.arg('offset')+Var('_lane_row')*output[-1]+Var('_lane_col'))
                for k,v in source.items():delta[k]=delta.get(k,0)-v
                delta={k:v for k,v in delta.items() if v}
                if axis==len(shape)-1:
                    # Input width is one: every source address coefficient is
                    # before the expanded column except its zero column lane.
                    if load.arg('cols')!=Const(1):raise BitRepeatError('column repeat must load one column')
                    expected={k:(repeats-1)*v for k,v in source.items() if k!='_lane_col'}
                else:
                    expected={f'd{i}':(repeats-1)*_strides(shape)[i] for i in range(axis)}
                expected['rep']=_strides(output)[axis]
                expected={k:v for k,v in expected.items() if v}
                if delta!=expected:raise BitRepeatError('repeat actual DMA source/destination affine relation changed')
                stored+=prod(n for _,n in scope)*op.arg('rows').value*op.arg('cols').value
                sites+=1
    walk(kernel.body,())
    if loaded!=prod(shape) or stored!=prod(output):
        raise BitRepeatError('repeat actual DMA cardinalities differ from source and destination')
    ordering=_store_epoch_admission()
    if ordering is not None:ordering['consumed_kernel_sha256']=kernel.digest()
    return {'schema':'submission.raw_bit_repeat.v1','store_epoch_admission':ordering,'kernel_sha256':kernel.digest(),
        'source_shape':list(shape),'output_shape':list(output),'axis':axis,'repeats':repeats,
        'checked_actual_store_sites':sites,'actual_affine_dma_relation':'dst-src = sum(coordinate[i]*(repeats-1)*source_stride[i], i<axis)+rep*output_stride[axis], including DMA row/column lanes',
        'proof':'non-repeated indices and repeat index partition output coordinates; row/column quotient+remainder tiles partition each source block exactly once',
        'source_map':'same output coordinate with expanded axis replaced by zero',
        'address_bounds':'all coordinates bounded by static extents; independent source/destination dense row pitches fit selected fields',
        'source_row_pitch_bytes':shape[-1]*4,'destination_row_pitch_bytes':output[-1]*4,
        'output_coverage':'each output coordinate has exactly one repeat index, remaining source coordinate and tile/lane quotient+remainder; stores are disjoint and cover every output byte',
        'lifetime':'load completes before replicated reads; all disjoint stores complete before accumulator reuse',
        'requested_load_bytes':prod(shape)*4,'requested_store_bytes':prod(output)*4,
        'complete_call_cycles':'UNKNOWN','placement':'determined by source-region planner'}

@dataclass(frozen=True)
class BitRepeatPlan:
    shape:tuple
    axis:int
    repeats:int
    output_shape:tuple
    kernel:Kernel
    proof:dict
    def revalidate(self):return check_repeat(self.kernel,self.shape,self.axis,self.repeats)

def plan_repeat(shape,axis,repeats,dtype='f32'):
    if dtype!='f32':raise BitRepeatError('repeat requires unchanged f32 metadata')
    shape,output=_geometry(shape,axis,repeats);kernel=_kernel(shape,axis,repeats)
    return BitRepeatPlan(shape,axis,repeats,output,kernel,check_repeat(kernel,shape,axis,repeats))

def lower_repeat(ctx, operands, attributes):
    if (set(operands) != {'src', 'dst'} or set(attributes) != {'axis', 'repeats', 'output_dtype', 'storage_dtype', 'semantic'}
            or attributes.get('output_dtype') != 'f32' or attributes.get('storage_dtype') != 'i32'
            or attributes.get('semantic') != 'bit_preserving_repeat'):
        raise BitRepeatError('repeat requires exact source raw-bit contract without numeric attributes')
    src, dst = operands['src'], operands['dst']
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    if src == dst or src in aliases or dst in aliases:
        raise BitRepeatError('repeat requires separate materialized destination, without alias views')
    specs = [ctx.spec(src), ctx.spec(dst)]
    if any(s['dtype'] != 'f32' or s.get('strides') or s.get('offset', 0) for s in specs):
        raise BitRepeatError('repeat requires ordinary dense f32 storage boundaries')
    plan = plan_repeat(specs[0]['shape'], attributes['axis'], attributes['repeats'])
    if list(plan.output_shape) != specs[1]['shape']:
        raise BitRepeatError('repeat destination shape differs from exact expansion')
    from xdsl.dialects.builtin import IntegerType
    bridge = FnBuilderIndexAdapter(ctx.fb)

    def emit(name, arguments):
        if name == 'repeat_drain': ctx.fence(); return
        if name == 'repeat_config':
            ctx.emit(*isa.config_ld(plan.shape[-1] * 4, shrink=False))
            ctx.emit(*isa.config_st(plan.output_shape[-1] * 4)); return
        a = {key: bridge.value(value) for key, value in arguments.items()}
        tensor = src if name == 'repeat_mvin' else dst
        ptr = ctx.fb.gep(ctx.buffer(tensor), a['offset'], IntegerType(32))
        packed = isa.pack_matrix(a['addr'], a['rows'], a['cols'], fb=ctx.fb)
        ctx.emit(isa.FUNCT['mvin' if name == 'repeat_mvin' else 'mvout'], ptr, packed)

    emitted = emit_index_program(plan.kernel, bridge, emit)
    return {'proof': plan.proof, 'emission': emitted, 'kernel_sha256': plan.kernel.digest(),
            'storage_view': {'logical_dtype': 'f32', 'transfer_dtype': 'i32', 'element_bytes': 4,
                             'mapping': 'original ABI pointers and exact checked four-byte source/destination offsets'}}
