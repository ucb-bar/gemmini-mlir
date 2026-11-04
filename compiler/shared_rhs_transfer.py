"""Actual wide shared-member transfers with exact original band custody.

Only pure-load tail domains split. Existing library proposals coalesce them;
all final bands expand back to the complete original source/member/drain trace.
The wide kernel alone reaches the ordinary shared Encoder.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from pathlib import Path
import json
from merlin.sched.ir import Call,Loop,Const,Var,Select,Scaled,Sum,Ptr,add
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.dense_shared_rhs import DenseSharedRHSResult
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing,_substitute
from merlin.compile.scheduling.panel_tile_loads import expand_panel_tile_loads,_canonical_load_runs,_map_call,_offset_form
from dense_transfer import _trace_equal,_singleton_load_nests


MAX_TRANSFER_TILES=7


def _split_load_tails(kernel,project):
    """Use the library checked final peel for a pure-load tail domain."""
    from merlin.sched.ir import check_structure
    from merlin.sched.primitives import peel,Cursor
    if check_structure(kernel):raise ValueError('shared transfer requires unshadowed valid source binders')
    records=[]
    def pure(items):
        return all(pure(s.body) and not(s.iter_args or s.yields or s.results) if type(s)is Loop else type(s)is Call and project(s)is not None for s in items)
    def selects(value,var,end):
        if type(value)is Select:
            return (value.var==var and value.equals==end) or selects(value.then,var,end) or selects(value.other,var,end)
        if type(value)is Sum:return any(selects(t,var,end)for t in value.terms)
        if type(value)is Scaled:return selects(value.expr,var,end)
        if type(value)is Ptr:return selects(value.offset,var,end)
        return False
    def has(items,var,end):
        return any(has(s.body,var,end)if type(s)is Loop else any(selects(v,var,end)for _,v in s.args)for s in items)
    def find(items,path=()):
        for s in items:
            if type(s)is not Loop:continue
            if s.extent>1 and pure(s.body) and has(s.body,s.var,s.extent-1):return (*path,s.var),s.extent
            found=find(s.body,(*path,s.var))
            if found:return found
        return None
    while (found:=find(kernel.body)) is not None:
        path,extent=found
        kernel=peel(kernel,Cursor(loops=path),at='back')
        records.append(dict(path=path,extent=extent,partition=[extent-1,1],proof='library peel: complete actual instance operands including Ptr; original issue-order partition'))
    return kernel,records


@dataclass(frozen=True)
class SharedRHSTransferResult:
    original:object
    kernel:object
    realization:dict
    _guard:object
    @property
    def blocks(self):return self.original.blocks
    def revalidate(self):self._guard(self)
    emit_index_program=DenseSharedRHSResult.emit_index_program


def coalesce_shared_rhs(original,problems,storages,adapter,encode_member,project_member):
    original.revalidate()
    if type(MAX_TRANSFER_TILES)is not int or not 1<=MAX_TRANSFER_TILES<=7:raise ValueError('shared transfer tile bound must be in selected range 1..7')
    root=Path(__file__).parent
    facts_path=root/'data/command_ordering.json'
    extents=json.loads(facts_path.read_text())['ordering']['extents']
    laws={}
    for role,name,tile,stream in [('lhs','MVIN.spad_addr',problems[0].tile_k,0),('weight','MVIN2.spad_addr',problems[0].tile_n,1)]:
        e=extents[name];b=e['blocks']
        if e['status']!='derived' or b['block']!=tile or b['stride']['code']!=1 or b['stride']['state']['value']!=stream:
            raise ValueError('shared transfer lacks selected MVIN multi-block geometry')
        laws[role]=(tile,min(tile,(1<<e['rows']['width'])-1),((1<<b['cols']['width'])-1)//tile*tile)
    def project(call):
        if call.instr==DRAIN:return None
        member,op=project_member(call)
        if type(op)is not Load:return None
        if encode_member(member,op)!=call or op.step!=Const(1):raise ValueError('shared transfer changed actual original member source fields')
        p=problems[0 if member is None else member];storage=storages[0 if member is None else member]
        tile,rows,cols=laws[op.role]
        pitch=p.k if op.role=='lhs'else p.n
        law=TileLoadAddressLaw(tile,rows,cols,1,pitch,16,'selected-shared-mvin:'+sha256(facts_path.read_bytes()).hexdigest())
        ptr=dict(call.args)['ptr'];view=storage.lhs if op.role=='lhs'else storage.rhs
        return TileLoadMapping(ptr.tensor,view.backing,ptr.offset,op.row,op.rows,op.cols,law)
    def encode(call,columns):
        member,op=project_member(call)
        return encode_member(member,replace(op,cols=Const(columns)))
    def encode_band(call,band):
        member,op=project_member(call);mapping=project(call)
        delta=band.source_byte+mapping.source_byte*-1
        c,terms=_offset_form(delta);tile=mapping.law.tile_columns
        if c%tile or any(v%tile for v in terms.values()):raise ValueError('shared transfer band lost tile source alignment')
        bands=add(Const(c//tile),*(atom*(v//tile)for atom,v in terms.items()))
        return encode_member(member,replace(op,dram_col=op.dram_col+delta,row=band.local_row,cols=band.columns,block=(op.block[0],op.block[1]+bands)))
    split,partitions=_split_load_tails(original.kernel,project)
    proposal,changes=propose_tile_load_coalescing(split,project=project,encode=encode,max_tiles=min(MAX_TRANSFER_TILES,min(x[2]//x[0]for x in laws.values())),coalesce_tail=True)
    def normalize(k):
        k,_=_singleton_load_nests(k,project)
        return _canonical_load_runs(k,project)[0]
    baseline=normalize(split)
    def check(k):
        expanded,record=expand_panel_tile_loads(k,project=project,encode_band=encode_band)
        expanded=normalize(expanded)
        if replace(expanded,body=())!=replace(baseline,body=()) or not _trace_equal(baseline.body,expanded.body):
            raise ValueError('shared actual wide bands differ from original complete member/source/drain trace')
        return record
    projection=check(proposal)
    paths=('shared_rhs_transfer.py','shared_rhs_ready.py','shared_rhs_input_pipeline.py','resident_chain.py','encoding.py','dense.py','dense_transfer.py','data/command_ordering.json','compiler_library/merlin/compile/scheduling/tile_load_coalescing.py','compiler_library/merlin/compile/scheduling/panel_tile_loads.py','compiler_library/merlin/compile/scheduling/llvm_program.py')
    selected_tiles=MAX_TRANSFER_TILES
    owners={name:sha256((root/name).read_bytes()).hexdigest()for name in paths}
    def guard(result):
        original.revalidate()
        if MAX_TRANSFER_TILES!=selected_tiles:raise ValueError('shared transfer selected tile bound changed')
        if any(sha256((root/n).read_bytes()).hexdigest()!=h for n,h in owners.items()):raise ValueError('shared transfer selected proof/emission owner changed')
        rebuilt,splits=_split_load_tails(original.kernel,project)
        if rebuilt!=split or splits!=partitions or result.kernel!=proposal or result.realization['shared_rhs_transfer']['source_owners']!=owners:
            raise ValueError('shared transfer actual program or domain partition changed')
        if check(result.kernel)!=projection:raise ValueError('shared transfer complete actual band projection changed')
    result=SharedRHSTransferResult(original,proposal,{**original.realization,'kernel_sha256':proposal.digest(),'shared_rhs_transfer':dict(selected=bool(changes['changes']),original_kernel_sha256=original.kernel.digest(),pure_load_domain_partitions=partitions,proposal=changes,actual_band_projection=projection,source_owners=owners)},guard)
    result.revalidate()
    return result
