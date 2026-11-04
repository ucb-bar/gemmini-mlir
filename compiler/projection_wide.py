"""Checked wide row-major loads composed with checked K-stream epochs.

Address coalescing stays inside an original epoch. Its checker expands ALL
physical bands against the source and hazards. The epoch consumer separately
checks those original bands. Porting its completion deletions by unchanged,
unique loop paths commutes with this address-only rewrite.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
import json
from pathlib import Path
from merlin.sched.ir import Call,Loop,Const,as_expr
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads,_offset_form
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing


def _drain_paths(kernel):
    paths=set()
    def visit(body,path):
        for s in body:
            if isinstance(s,Loop):visit(s.body,(*path,s.var))
            elif s.instr==DRAIN:
                if path in paths:raise ValueError('duplicate completion path')
                paths.add(path)
    visit(kernel.body,())
    return paths


def _port_completions(wide,source,epochs):
    before,after=_drain_paths(source),_drain_paths(epochs)
    if not after <= before or _drain_paths(wide)!=before:
        raise ValueError('wide rewrite changed original completion scopes')
    removed=before-after
    def walk(body,path=()):
        out=[]
        for s in body:
            if isinstance(s,Loop):out.append(replace(s,body=walk(s.body,(*path,s.var))))
            elif s.instr!=DRAIN or path not in removed:out.append(s)
        return tuple(out)
    result=replace(wide,body=walk(wide.body))
    if _drain_paths(result)!=after:raise ValueError('wide completion map changed')
    return result


@dataclass(frozen=True)
class WideEpochResult:
    original:object
    wide:object
    kernel:object
    realization:dict
    owner_sha256:str
    def revalidate(self):
        self.original.revalidate();self.wide.revalidate()
        if (self.kernel!=_port_completions(self.wide.kernel,self.wide.original.kernel,self.original.kernel)
                or sha256(Path(__file__).read_bytes()).hexdigest()!=self.owner_sha256):
            raise ValueError('wide epoch source, mapping or consumer changed')
    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        receipt=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate();return receipt


def apply_wide_epochs(epochs,problem,block,resources,storage,adapter,max_tiles=None):
    # Explicit original owner: no relabeling a wide or epoch result as a panel.
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochResult
    if type(epochs) is not DensePanelEpochResult:return epochs
    original=epochs._source()
    root=Path(__file__).parent
    selection=json.loads((root/'command_ordering.json').read_bytes())
    facts_bytes=(root/'rtl_facts/facts.json').read_bytes()
    facts_sha=sha256(facts_bytes).hexdigest()
    read_client=json.loads(facts_bytes)['facts']['timing_structure']['memory_clients']['clients']['dma_read']
    # One selected maximum reader request per row; final hardware column
    # bounds and complete-column coalescing still constrain every proposal.
    if max_tiles is None:
        max_tiles=read_client['max_request_bytes']//problem.tile_n
    if max_tiles < 2:
        return epochs
    if selection['base_facts']['sha256']!=facts_sha:return epochs
    record=selection['ordering']
    # Actual decoded hardware width, not the unused 16-bit packet slot.
    ex=record['extents']['MVIN.spad_addr'];cols=ex['blocks']['cols'];rows=ex['rows']
    if ex['status']!='derived' or ex['blocks']['block']!=problem.tile_n:return epochs
    edge=problem.tile_n
    max_columns=min(max_tiles*edge,((1<<cols['width'])-1)//edge*edge)
    identity='gemmini-exact-i8-row-load:'+facts_sha+':'+selection['ordering_sha256']
    def project(call):
        if call.instr not in ('gemmini_lhs_load','gemmini_weight_load'):return None
        op=adapter.project(call);left=op.role=='lhs';pitch=problem.k if left else problem.n
        law=TileLoadAddressLaw(edge,min(edge,(1<<rows['width'])-1),max_columns,1,pitch,edge,identity)
        return TileLoadMapping(op.role,storage.lhs.backing,
            op.dram_row*pitch+op.dram_col,op.row,op.rows,op.cols,law)
    def encode(first,columns):
        op=adapter.project(first)
        return adapter.encode(replace(op,cols=as_expr(columns)))
    def encode_band(call,band):
        op=adapter.project(call);old=project(call)
        delta,terms=_offset_form(band.source_byte+old.source_byte*(-1))
        if terms or delta%edge:raise ValueError('band does not preserve an original tile column')
        return adapter.encode(replace(op,block=(op.block[0],op.block[1]+delta//edge),
            dram_col=op.dram_col+delta,row=band.local_row,cols=band.columns))
    try:
        proposal,report=propose_tile_load_coalescing(original.kernel,project=project,
            encode=encode,max_tiles=max_tiles,max_static_nodes=4096,coalesce_tail=True)
        if not report['changes']:return epochs
        wide=admit_panel_tile_loads(original,proposal,problem,block,resources,storage,adapter,
            project=project,encode_band=encode_band,static_budget=128,timeout_ms=15000)
        kernel=_port_completions(wide.kernel,original.kernel,epochs.kernel)
        realization={**epochs.realization,'kernel_sha256':kernel.digest(),
            'wide_loads':report,'wide_check':wide.realization['tile_load_projection'],
            'composition':'exact wide-band consumer followed by original-band epoch consumer; unique unchanged completion paths'}
        result=WideEpochResult(epochs,wide,kernel,realization,sha256(Path(__file__).read_bytes()).hexdigest())
        result.revalidate();return result
    except ValueError as exc:
        import sys
        print('projection wide fallback: '+str(exc),file=sys.stderr)
        return epochs
