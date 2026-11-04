"""Opt-in static strided-row copies with exact byte and scratchpad effects.

Only contiguous inner runs become DMA columns. Original outer strides become
independent source/destination byte pitches; every other outer axis stays a
counted loop. All original offsets and destination holes are retained.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path
from hashlib import sha256
import json

from elementwise_tiles import plan_rectangles
from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Kernel,TensorArg,Call,Loop,Const,Var,select,check_structure,instances,concretize


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('movement_rowwise_tiles',default='off',values=('off','static_runs'),root=Path(__file__).resolve().parent)


def _nonoverlap(shape,strides):
    span=1
    for stride,extent in sorted((s,d) for s,d in zip(strides,shape) if d>1):
        if stride<span:return False
        span+=(extent-1)*stride
    return True


def _kernel(shape,ss,ds,so,do,width,source_bytes,destination_bytes,inner):
    run=prod(shape[inner:])*width
    outer=shape[:inner];souter=ss[:inner];douter=ds[:inner]
    rows=outer[-1];sp=souter[-1]*width;dp=douter[-1]*width
    blocks=(rows+15)//16;row=Var('row_block')*16
    height=select('row_block',blocks-1,(rows-1)%16+1,16)
    sbase=Const(so*width);dbase=Const(do*width)
    for axis in range(len(outer)-1):
        sbase=sbase+Var('outer_'+str(axis))*souter[axis]*width
        dbase=dbase+Var('outer_'+str(axis))*douter[axis]*width
    def effects(column,columns):
        args=(('source_byte',sbase+row*sp+column),('destination_byte',dbase+row*dp+column),
              ('rows',height),('columns',Const(columns)))
        return (Call('load_bytes',args),Call('complete',()),Call('store_bytes',args),Call('complete',()))
    cols=[];full,tail=divmod(run,64)
    if full:cols.append(Loop('column_block',full,effects(Var('column_block')*64,64)))
    if tail:cols.extend(effects(Const(full*64),tail))
    body=(Loop('row_block',blocks,tuple(cols)),)
    for axis in reversed(range(len(outer)-1)):
        body=(Loop('outer_'+str(axis),outer[axis],body),)
    return Kernel('gemmini_static_strided_row_copy',
        (TensorArg('source',(source_bytes,),'i8','read'),TensorArg('destination',(destination_bytes,),'i8','write')),body)


@dataclass(frozen=True)
class RowMovementPlan:
    source: str
    destination: str
    dtype: str
    shape: tuple
    source_strides: tuple
    destination_strides: tuple
    source_offset: int
    destination_offset: int
    source_elements: int
    destination_elements: int
    inner: int
    kernel: Kernel
    kernel_sha256: str

    @property
    def width(self):return 1 if self.dtype=='i8' else 4
    @property
    def run_bytes(self):return prod(self.shape[self.inner:])*self.width
    @property
    def source_pitch(self):return self.source_strides[self.inner-1]*self.width
    @property
    def destination_pitch(self):return self.destination_strides[self.inner-1]*self.width

    def revalidate(self):
        if self.dtype not in ('i8','i32') or self.source==self.destination:
            raise ValueError('row copy requires distinct same-width integer buffers')
        if (not self.shape or len(self.shape)!=len(self.source_strides) or len(self.shape)!=len(self.destination_strides)
            or any(type(v) is not int or v<=0 for v in self.shape+self.source_strides+self.destination_strides)):
            raise ValueError('row copy requires static positive shape and strides')
        if not 0<self.inner<len(self.shape):raise ValueError('row copy requires outer row and contiguous inner run')
        expected=1
        for axis in reversed(range(self.inner,len(self.shape))):
            if self.shape[axis]>1 and (self.source_strides[axis]!=expected or self.destination_strides[axis]!=expected):
                raise ValueError('inner copy run is not contiguous')
            expected*=self.shape[axis]
        if any(type(x) is not int or x<0 for x in (self.source_offset,self.destination_offset)):
            raise ValueError('row offsets must be static nonnegative')
        for strides,offset,extent in [(self.source_strides,self.source_offset,self.source_elements),(self.destination_strides,self.destination_offset,self.destination_elements)]:
            if type(extent) is not int or extent<=0 or offset+sum((d-1)*s for d,s in zip(self.shape,strides))>=extent:
                raise ValueError('row byte footprint exceeds original tensor')
            if not _nonoverlap(self.shape,strides):raise ValueError('row mapping has overlapping cells')
            if extent*self.width>=1<<63:raise ValueError('row offsets exceed signed i64')
        if max(self.source_pitch,self.destination_pitch)>=1<<32:
            raise ValueError('row pitches exceed DMA stride field')
        plan_rectangles(1024,'scratchpad').revalidate()
        TileLoadCoalescing(16,64,16,64,0,16,64)
        expected=_kernel(self.shape,self.source_strides,self.destination_strides,self.source_offset,self.destination_offset,self.width,self.source_elements*self.width,self.destination_elements*self.width,self.inner)
        errors=check_structure(self.kernel)
        if errors or self.kernel!=expected or self.kernel.digest()!=self.kernel_sha256:
            raise ValueError('row Kernel differs from exact source byte/effect partition')
        return self

    def report(self):
        self.revalidate()
        outer=self.shape[:self.inner]
        rectangles=prod(outer[:-1])*((outer[-1]+15)//16)*((self.run_bytes+63)//64)
        return dict(schema='authored.static_row_movement.v1',source=self.source,destination=self.destination,dtype=self.dtype,
            shape=self.shape,source_strides=self.source_strides,destination_strides=self.destination_strides,
            source_offset=self.source_offset,destination_offset=self.destination_offset,
            source_pitch_bytes=self.source_pitch,destination_pitch_bytes=self.destination_pitch,run_bytes=self.run_bytes,
            transferred_bytes=prod(self.shape)*self.width,rectangles=rectangles,
            kernel_sha256=self.kernel_sha256,private_scratchpad_rows=64,
            address_proof='canonical independent source/destination affine maps; bounded injective cells and contiguous inner runs',
            ordering='load complete, store complete before local reuse; original copy-task order retained',
            numeric='all source bytes preserved; no integer conversion, scaling or host tensor arithmetic',
            physical_qualification='not_established',cycles='UNKNOWN')


def plan_row_movement(cb,recipe):
    if 'opcode' in recipe:return None
    keys=('src','dst','shape','source_strides','destination_strides','source_offset','destination_offset')
    if not all(k in recipe for k in keys):return None
    src,dst=recipe['src'],recipe['dst'];s=cb['tensors'][src];d=cb['tensors'][dst]
    if s['dtype'] not in ('i8','i32') or s['dtype']!=d['dtype'] or src==dst:return None
    shape=tuple(recipe['shape']);ss=tuple(recipe['source_strides']);ds=tuple(recipe['destination_strides'])
    if not shape or len(shape)!=len(ss) or len(shape)!=len(ds):return None
    if any(type(x) is not int or x<=0 for x in shape+ss+ds):return None
    inner=len(shape);pitch=1
    for axis in reversed(range(len(shape))):
        if shape[axis]!=1 and (ss[axis]!=pitch or ds[axis]!=pitch):break
        inner=axis;pitch*=shape[axis]
    if not 0<inner<len(shape):return None
    for spec in (s,d):
        if any(type(x) is not int or x<=0 for x in spec['shape']):return None
    width=1 if s['dtype']=='i8' else 4
    try:
        kernel=_kernel(shape,ss,ds,recipe['source_offset'],recipe['destination_offset'],width,prod(s['shape'])*width,prod(d['shape'])*width,inner)
        return RowMovementPlan(src,dst,s['dtype'],shape,ss,ds,recipe['source_offset'],recipe['destination_offset'],prod(s['shape']),prod(d['shape']),inner,kernel,kernel.digest()).revalidate()
    except (ValueError,TypeError):return None


class _Control:
    def __init__(self,b):self.b=b
    def op(self,opcode,a,c):return getattr(self.b,opcode)(a,c)
    def cmp(self,p,a,c):return self.b.icmp(p,a,c)
    def select(self,c,a,d):return self.b.select(c,a,d)
    def loop(self,start,stop,step,callback):
        with self.b.counted_loop(start,stop,step) as i:callback(i)


def emit_row_movement(builder,plan,ptrs):
    p=plan.revalidate()
    if ptrs[p.source]==ptrs[p.destination]:raise ValueError('row copy refuses aliasing source/destination pointers')
    builder.fence();builder.config_ex()
    builder.config_ld(p.source_pitch,block_stride=16,id=0);builder.config_st(p.destination_pitch)
    def effect(name,a):
        if name=='complete':builder.fence()
        elif name=='load_bytes':builder.mvin(builder.ptr_offset(ptrs[p.source],a['source_byte']),0,a['columns'],a['rows'],id=0)
        elif name=='store_bytes':builder.mvout(builder.ptr_offset(ptrs[p.destination],a['destination_byte']),0,a['columns'],a['rows'])
        else:raise ValueError('unknown checked row movement effect')
    receipt=emit_index_program(p.kernel,_Control(builder),effect)
    builder.fence();p.revalidate()
    return dict(p.report(),emission=receipt)


def lower_row_movement(builder,cb,recipe,ptrs):
    p=plan_row_movement(cb,recipe)
    if p is None:return False
    from movement_pipeline import emit_if_enabled
    report=emit_if_enabled(builder,p,ptrs)
    if report is None: report=emit_row_movement(builder,p,ptrs)
    if not hasattr(builder,'movement_tile_receipts'):builder.movement_tile_receipts=[]
    builder.movement_tile_receipts.append(report)
    return True


def execute_row_movement(plan,source_bytes,destination_bytes):
    p=plan.revalidate();output=bytearray(destination_bytes);local={};pending=False;writes=set()
    for call,env in instances(p.kernel):
        a=concretize(call,env)
        if call.instr=='complete':pending=False;continue
        if pending:raise ValueError('row effect before completion')
        mapping=TileLoadCoalescing(a['rows'],a['columns'],16,64,0,16,64)
        for r in range(a['rows']):
            for c in range(a['columns']):
                cell=mapping.location(r,c)
                if call.instr=='load_bytes':local[cell]=source_bytes[a['source_byte']+r*p.source_pitch+c]
                elif call.instr=='store_bytes':
                    byte=a['destination_byte']+r*p.destination_pitch+c
                    if byte in writes:raise ValueError('row output byte written twice')
                    writes.add(byte);output[byte]=local[cell]
                else:raise ValueError('unknown row effect')
        pending=True
    if pending or len(writes)!=prod(p.shape)*p.width:raise ValueError('row copy dropped source cells')
    return bytes(output)
