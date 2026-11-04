"""Opt-in bounded wide DMA for contiguous signed-byte movement/residuals.

The selected decoded block-address law admits64 columns, each group of16
occupying16 distinct local rows. No scheduling or completion permission is
inferred from the ISA packet's reserved16-bit extent fields. Full-width i32
loads remain primitive under the public64-byte transfer-row bound.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from isa import ACC, ACCUMULATE, DIM, pack_matrix
from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
from merlin.compile.scheduling.program_limits import AddressWindow


@lru_cache(maxsize=4)
def _selected(root):
    root=Path(root)
    facts_bytes=(root/'facts.json').read_bytes()
    selected=json.loads((root/'command_ordering.json').read_text())
    ordering=selected['ordering']
    if (sha256(facts_bytes).hexdigest()!=selected['base_facts']['sha256'] or
        sha256(json.dumps(ordering,sort_keys=True).encode()).hexdigest()!=selected['ordering_sha256']):
        raise ValueError('wide elementwise requires intact selected facts and ordering')
    spaces={s['name']:s for s in json.loads(facts_bytes)['facts']['memory_address_spaces']}
    if DIM!=16 or spaces['scratchpad']['row_elems']!=DIM or spaces['accumulator']['lanes']!=DIM:
        raise ValueError('wide elementwise requires selected16-lane local rows')
    for name,load_id in [('MVIN.spad_addr',0),('MVIN2.spad_addr',1),('MVOUT.acc_addr',None)]:
        e=ordering['extents'][name]; block=e['blocks']; stride=block['stride']
        if (e['status']!='derived' or e['rows']['offset']!=48 or e['rows']['width']!=5 or
            block['cols']['offset']!=32 or block['cols']['width']!=7 or block['block']!=DIM):
            raise ValueError('wide elementwise requires selected derived block extents')
        if load_id is None:
            if stride!={'constant':DIM}: raise ValueError('wide store must have fixed16-row block stride')
        elif (stride['offset']!=16 or stride['width']!=16 or
              stride['state']!={'operand':'rs1','offset':3,'width':2,'value':load_id}):
            raise ValueError('wide load must select its own configured16-row block stride')
    # Public gemmini_params.h MAX_BYTES64; the selected extent probes admit
    # four16-column blocks. Keep the conservative intersection, not127cols.
    return spaces,selected['ordering_sha256'],sha256(facts_bytes).hexdigest()


@dataclass(frozen=True)
class RectanglePlan:
    elements: int
    columns: int
    local_space: str
    facts_root: str

    def revalidate(self):
        if type(self.elements) is not int or not 0<self.elements<(1<<61):
            raise ValueError('positive bounded contiguous element extent required')
        if self.columns!=64 or self.local_space not in ('scratchpad','accumulator'):
            raise ValueError('wide standalone path supports only64-column signed-byte rectangles')
        spaces,_,_=_selected(self.facts_root)
        space=spaces[self.local_space]
        row_bytes=DIM if self.local_space=='scratchpad' else space['row_bytes']
        window=AddressWindow(self.local_space,0,64*row_bytes)
        if window.start+window.size>space['bytes']:
            raise ValueError('wide rectangle exceeds private local rows')
        # Library address proof: block-major16-row storage, no overlapping
        # complete local rows. Check actual max descriptors through encoder.
        TileLoadCoalescing(DIM,self.columns,DIM,self.columns,0,DIM,space['bytes']//row_bytes)
        pack_matrix(ACC if self.local_space=='accumulator' else 0,self.columns,DIM)
        return self

    def rectangles(self):
        """Concrete development oracle, never expanded during LLVM emission."""
        self.revalidate()
        block=self.columns*DIM; bulk=self.elements//block*block
        for offset in range(0,bulk,block): yield offset,self.columns,DIM
        rows=(self.elements-bulk)//self.columns
        if rows: yield bulk,self.columns,rows
        tail=(self.elements-bulk)%self.columns
        if tail: yield bulk+rows*self.columns,tail,1

    def emit(self,builder,callback):
        self.revalidate()
        block=self.columns*DIM; bulk=self.elements//block*block
        if bulk:
            with builder.counted_loop(0,bulk,block) as offset:
                callback(offset,self.columns,DIM)
        rows=(self.elements-bulk)//self.columns
        if rows: callback(bulk,self.columns,rows)
        tail=(self.elements-bulk)%self.columns
        if tail: callback(bulk+rows*self.columns,tail,1)

    def report(self):
        self.revalidate()
        _,ordering,facts=_selected(self.facts_root)
        block=self.columns*DIM; bulk,tail=divmod(self.elements,block)
        count=bulk+int(tail//self.columns>0)+int(tail%self.columns>0)
        return dict(mechanism='selected_wide_standalone_dma',elements=self.elements,
            max_columns=self.columns,max_rows=DIM,local_block_stride_rows=DIM,
            local_space=self.local_space,private_rows=64,
            rectangle_count=count,maximum_static_callback_sites=3,
            ordering_sha256=ordering,facts_sha256=facts,
            numerical_change='none; same lane arithmetic and scale literals',
            completion='overwrite complete; add complete; store complete',
            cycles='UNKNOWN',physical_qualification='not_established')


def plan_rectangles(elements,local_space,*,facts_root=None):
    return RectanglePlan(elements,64,local_space,str(Path(facts_root) if facts_root is not None else Path(__file__).resolve().parent)).revalidate()


def lower_small_residual(builder,cb,command,ptrs):
    """One private DMA panel; larger and movement calls keep their incumbent.

    Two original geometries have passing whole-call RTL pairs (917 to758 and
    672 to563 cycles). Neighbors remain unmeasured. Independent exact numerical,
    capacity and local-layout checks still run in the shared emitter below.
    """
    if command.get('opcode') != 'RESIDUAL_ADD':
        return False
    from elementwise import _spec
    dst = _spec(cb,command['operands']['dst'])
    if prod(dst['shape']) > 64*DIM:
        return False
    if not lower_elementwise_tiled(builder,cb,command,ptrs):
        return False
    builder.elementwise_tile_receipts[-1]['selection'] = {
        'class':'one_private_residual_dma_panel',
        'support':'two same-provider measured whole-call original RTL pairs',
        'evidence':['artifacts/experiments/elementwise_raw041/evidence.json',
                    'artifacts/experiments/elementwise_relu041/evidence.json'],
        'neighbor_timing':'UNKNOWN',
    }
    return True


def lower_elementwise_tiled(builder,cb,command,ptrs):
    """Opt-in before incumbent lower_elementwise;False means unchanged fallback.

    Only source-admitted i8 copies/residuals are changed. Existing residual
    numerical admission owns load/readout factorization and exhaustive byte
    proof. All effects use the common FunctionBuilder terminal encoder.
    """
    from elementwise import _spec,residual_plan
    opcode=command.get('opcode')
    if opcode not in ('MOVEMENT','RESIDUAL_ADD'): return False
    ops=command.get('operands',{});attrs=command.get('attributes',{})
    dst=_spec(cb,ops['dst'])
    if dst['dtype']!='i8' or attrs.get('output_dtype','i8')!='i8': return False
    size=prod(dst['shape'])
    # An opt-in issue-count admission: never increase the original rectangle
    # count for a small/tail-only tensor. This is not a cycle prediction.
    primitive=size//(DIM*DIM)+int(size%(DIM*DIM)//DIM>0)+int(size%DIM>0)
    wide=size//(64*DIM)+int(size%(64*DIM)//64>0)+int(size%64>0)
    if wide>=primitive: return False
    if opcode=='MOVEMENT':
        src=_spec(cb,ops['src'])
        if src['dtype']!='i8': return False
        if prod(src['shape'])!=size: raise ValueError('movement must preserve contiguous element count')
        if attrs.get('semantic','copy') not in ('copy','mvin_mvout'): return False
        p=plan_rectangles(size,'scratchpad')
        builder.fence();builder.config_ex()
        builder.config_ld(p.columns,block_stride=DIM,id=0)
        builder.config_st(p.columns)
        def effect(offset,cols,rows):
            builder.mvin(builder.ptr_offset(ptrs[ops['src']],offset),0,cols,rows,id=0)
            builder.fence()
            builder.mvout(builder.ptr_offset(ptrs[ops['dst']],offset),0,cols,rows)
            builder.fence()
    else:
        lhs,rhs=_spec(cb,ops['lhs']),_spec(cb,ops['rhs'])
        if lhs['dtype']!='i8' or rhs['dtype']!='i8' or not(lhs['shape']==rhs['shape']==dst['shape']):
            return False
        stages=attrs.get('epilogue',[])
        if any(stage!='relu' for stage in stages): return False
        relu='relu' in stages
        la,lb,factor,error=residual_plan(attrs['lhs_scale'],attrs['rhs_scale'],attrs['bound_lsb'],relu)
        p=plan_rectangles(size,'accumulator')
        builder.fence()
        builder.config_ld(p.columns,la,shrink=True,block_stride=DIM,id=0)
        builder.config_ld(p.columns,lb,shrink=True,block_stride=DIM,id=1)
        builder.config_ex(activation=int(relu),scale=factor)
        builder.config_st(p.columns,activation=int(relu),scale=factor)
        def effect(offset,cols,rows):
            builder.mvin(builder.ptr_offset(ptrs[ops['lhs']],offset),ACC,cols,rows,id=0)
            builder.fence()
            builder.mvin(builder.ptr_offset(ptrs[ops['rhs']],offset),ACC|ACCUMULATE,cols,rows,id=1)
            builder.fence()
            builder.mvout(builder.ptr_offset(ptrs[ops['dst']],offset),ACC,cols,rows)
            builder.fence()
    p.emit(builder,effect);builder.fence()
    if not hasattr(builder,'elementwise_tile_receipts'): builder.elementwise_tile_receipts=[]
    builder.elementwise_tile_receipts.append(p.report())
    return True
