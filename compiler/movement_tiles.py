"""Checked byte-preserving wide DMA adapter for contiguous integer copies.

An i32 copy is a scratchpad byte copy, never an accumulator conversion. This
preserves every bit, including negative values, without f32 or host arithmetic.
The parent may call this after source transport admission and before incumbent
movement lowering. No source grouping, placement or completion is reordered.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
import json

from elementwise_tiles import plan_rectangles
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Kernel, TensorArg, Loop, Call, Const, Var, check_structure, instances, concretize


def _kernel(byte_count):
    block=1024; bulk=byte_count//block*block
    def effects(offset,columns,rows):
        args=(('byte_offset',offset),('columns',Const(columns)),('rows',Const(rows)))
        return (Call('load_bytes',args),Call('complete',()),
                Call('store_bytes',args),Call('complete',()))
    body=[]
    if bulk: body.append(Loop('rectangle',bulk//block,effects(Var('rectangle')*block,64,16)))
    rows=(byte_count-bulk)//64
    if rows: body.extend(effects(Const(bulk),64,rows))
    tail=(byte_count-bulk)%64
    if tail: body.extend(effects(Const(bulk+rows*64),tail,1))
    return Kernel('gemmini_contiguous_copy_bytes',
        (TensorArg('source',(byte_count,),'i8','read'),TensorArg('destination',(byte_count,),'i8','write')),
        tuple(body))


@dataclass(frozen=True)
class MovementPlan:
    source: str
    destination: str
    elements: int
    dtype: str
    source_byte: int
    destination_byte: int
    kernel: Kernel
    kernel_sha256: str

    @property
    def byte_count(self): return self.elements*(1 if self.dtype=='i8' else 4)

    def revalidate(self):
        if self.dtype not in ('i8','i32') or type(self.elements) is not int or self.elements<=0:
            raise ValueError('contiguous movement requires positive static i8/i32 element count')
        if any(type(x) is not int or x<0 for x in (self.source_byte,self.destination_byte)):
            raise ValueError('contiguous copy offsets must be static nonnegative bytes')
        if max(self.source_byte,self.destination_byte)+self.byte_count>=1<<63:
            raise ValueError('copy address arithmetic exceeds signed i64')
        plan_rectangles(self.byte_count,'scratchpad').revalidate()
        errors=check_structure(self.kernel)
        if errors or self.kernel!=_kernel(self.byte_count) or self.kernel_sha256!=self.kernel.digest():
            raise ValueError('copy axes/effects differ from canonical exact byte partition: '+str(errors))
        return self

    def report(self):
        self.revalidate()
        return dict(schema='authored.contiguous_integer_copy.v1',source=self.source,destination=self.destination,
            dtype=self.dtype,elements=self.elements,source_byte=self.source_byte,destination_byte=self.destination_byte,
            bytes=self.byte_count,kernel_sha256=self.kernel_sha256,
            rectangle_plan=plan_rectangles(self.byte_count,'scratchpad').report(),
            arithmetic='none; scratchpad byte identity preserves all source integer bits',
            ordering='load; completion; store; completion per private rectangle',
            source_boundary='original live destination and original source retained',
            native_numerical_execution='not_established')


def _contiguous(shape,strides):
    if len(shape)!=len(strides): return False
    pitch=1
    for extent,stride in reversed(tuple(zip(shape,strides))):
        if type(extent) is not int or extent<=0 or type(stride) is not int: return False
        if extent!=1 and stride!=pitch: return False
        pitch*=extent
    return True


def plan_movement(cb,command):
    """Admit MOVEMENT or one existing source_transport_tasks copy recipe.

    Returns None for noncontiguous, conversion, or other semantic routes.
    Recipe offsets/strides are in elements, exactly as linalg_lowering emits.
    Tensor declarations bound every original source and destination byte.
    """
    if command.get('opcode')=='MOVEMENT':
        attrs=command.get('attributes',{})
        if attrs.get('semantic','copy') not in ('copy','mvin_mvout'): return None
        if any(k in attrs for k in ('acc_scale','acc_scale_by_column')) or attrs.get('epilogue'): return None
        ops=command['operands'];src,dst=ops['src'],ops['dst']
        source,destination=cb['tensors'][src],cb['tensors'][dst]
        shape=source['shape'];count=prod(shape);so=do=0
        if count!=prod(destination['shape']): return None
        if attrs.get('output_dtype',destination['dtype'])!=destination['dtype']: return None
    elif 'opcode' not in command and all(k in command for k in ('src','dst','shape','source_strides','destination_strides','source_offset','destination_offset')):
        src,dst=command['src'],command['dst'];source,destination=cb['tensors'][src],cb['tensors'][dst]
        shape=command['shape'];count=prod(shape);so,do=command['source_offset'],command['destination_offset']
        if not(_contiguous(shape,command['source_strides']) and _contiguous(shape,command['destination_strides'])): return None
    else: return None
    dtype=source['dtype']
    if dtype not in ('i8','i32') or dtype!=destination['dtype']: return None
    for spec in (source,destination):
        if any(type(d) is not int or d<=0 for d in spec['shape']): return None
    if any(type(d) is not int or d<=0 for d in shape): return None
    if any(type(offset) is not int or offset<0 for offset in (so,do)): return None
    if so+count>prod(source['shape']) or do+count>prod(destination['shape']): return None
    # An overlapping same-buffer recipe has different memmove hazards; retain
    # the incumbent rather than pretending byte identity proves temporal safety.
    if src==dst and max(so,do)<min(so+count,do+count) and so!=do: return None
    width=1 if dtype=='i8' else 4
    kernel=_kernel(count*width)
    return MovementPlan(src,dst,count,dtype,so*width,do*width,kernel,kernel.digest()).revalidate()


class _Control:
    def __init__(self,b): self.b=b
    def op(self,opcode,a,c): return getattr(self.b,opcode)(a,c)
    def cmp(self,p,a,c): return self.b.icmp(p,a,c)
    def select(self,c,a,d): return self.b.select(c,a,d)
    def loop(self,start,stop,step,callback):
        with self.b.counted_loop(start,stop,step) as index: callback(index)


def emit_movement(builder,plan,ptrs):
    """Emit the actual checked Kernel through the common terminal encoder.

    Parent owns placement/source attribution and invalidation of any retained
    scratchpad owner of rows0..63. Distinct logical buffers must have the same
    disjoint allocation contract as the existing source-copy transport path.
    """
    p=plan.revalidate()
    if ptrs[p.source]==ptrs[p.destination] and p.source_byte!=p.destination_byte:
        if max(p.source_byte,p.destination_byte)<min(p.source_byte+p.byte_count,p.destination_byte+p.byte_count):
            raise ValueError('wide movement refuses overlapping byte ranges')
    builder.fence();builder.config_ex()
    builder.config_ld(64,block_stride=16,id=0);builder.config_st(64)
    def effect(name,a):
        if name=='complete':builder.fence();return
        offset=a['byte_offset']
        if name=='load_bytes':builder.mvin(builder.ptr_offset(ptrs[p.source],builder.add(p.source_byte,offset)),0,a['columns'],a['rows'],id=0)
        elif name=='store_bytes':builder.mvout(builder.ptr_offset(ptrs[p.destination],builder.add(p.destination_byte,offset)),0,a['columns'],a['rows'])
        else:raise ValueError('unknown checked movement effect '+name)
    receipt=emit_index_program(p.kernel,_Control(builder),effect)
    builder.fence();p.revalidate()
    return dict(p.report(),emission=receipt)


def lower_movement_tiled(builder,cb,command,ptrs):
    """Return True after emitting;False leaves unsupported input unchanged.

    Accepted command is either a MOVEMENT CB command or one contiguous copy
    recipe from source_transport_tasks. No residual/conversion path is owned.
    """
    p=plan_movement(cb,command)
    if p is None:
        from movement_rows import setting as row_setting, lower_row_movement
        if row_setting() == 'static_runs':
            return lower_row_movement(builder,cb,command,ptrs)
        return False
    from movement_pipeline import emit_if_enabled
    report=emit_if_enabled(builder,p,ptrs)
    if report is None: report=emit_movement(builder,p,ptrs)
    if not hasattr(builder,'movement_tile_receipts'):builder.movement_tile_receipts=[]
    builder.movement_tile_receipts.append(report)
    return True


def execute_movement(plan,source_bytes,destination_bytes):
    """Development-only oracle interpreting the actual checked byte effects."""
    from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
    p=plan.revalidate();out=bytearray(destination_bytes);local={};pending=False
    for call,env in instances(p.kernel):
        a=concretize(call,env)
        if call.instr=='complete':pending=False;continue
        if pending:raise ValueError('uncompleted private byte rectangle')
        mapping=TileLoadCoalescing(a['rows'],a['columns'],16,64,0,16,64)
        for r in range(a['rows']):
            for c in range(a['columns']):
                byte=a['byte_offset']+r*64+c;cell=mapping.location(r,c)
                if call.instr=='load_bytes':local[cell]=source_bytes[p.source_byte+byte]
                elif call.instr=='store_bytes':out[p.destination_byte+byte]=local[cell]
                else:raise ValueError('unknown byte effect')
        pending=True
    if pending:raise ValueError('unterminated byte effect')
    return bytes(out)


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('movement_dma_tiles',default='off',values=('off','contiguous_bytes'),root=Path(__file__).resolve().parent)
