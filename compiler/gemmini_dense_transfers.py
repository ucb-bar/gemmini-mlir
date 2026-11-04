"""Final physical RHS transfer aggregation under the original ready owner.

No completion is removed. The shared address-only proposal is expanded into
ALL original tile bands and admitted by the original source/epoch owner before
its actual wide commands reach the unchanged terminal encoder.
"""
from dataclasses import dataclass, replace, asdict
import hashlib
import json
from pathlib import Path

from merlin.sched.ir import Const, Var, add
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.dense_ready_readout import admit_ready_readout_tile_loads
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing, _linear)
from merlin.transforms.dense_planning import DenseResources
from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
from gemmini_dense_readout import ReadoutUnavailable


@dataclass(frozen=True)
class DenseTransferResult:
    checked: object

    @property
    def kernel(self): return self.checked.kernel

    @property
    def realization(self): return self.checked.realization

    def revalidate(self): self.checked.revalidate()

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        receipt=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def materialize_scf(self): return self.checked.materialize_scf()


def plan_transfers(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False,
                   outer_order=('m','n')):
    """Aggregate only complete contiguous RHS column tiles, keeping all drains.

    The selected decoded extent bounds the physical MVIN2 columns. A single
    selected DMA request bounds the proposed row width; this is a legal initial
    candidate, not a cycle-derived preference. Narrow/ragged tails retain their
    original commands. All ordinary signed dense readout behavior is retained.
    """
    from gemmini_dense_segmented import plan_segmented_readout
    original,receipt=plan_segmented_readout(incumbent,adapter,problem,geometry,facts_path,
        bias=bias,initial=initial,outer_order=outer_order)
    facts_path=Path(facts_path);raw=facts_path.read_bytes();facts=json.loads(raw)
    pin=json.loads((facts_path.parent/'COMMAND_ORDERING_SELECTION.json').read_text())
    if pin['base_facts']['sha256']!=hashlib.sha256(raw).hexdigest():
        raise ReadoutUnavailable('wide transfer selected hardware facts differ')
    record=pin['ordering'];extent=record['extents']['MVIN2.spad_addr']
    blocks=extent.get('blocks',{});cols=blocks.get('cols',{});stride=blocks.get('stride',{})
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    if (extent.get('status')!='derived' or blocks.get('block')!=edge
            or cols.get('operand')!='rs2' or cols.get('offset')!=32
            or type(cols.get('width')) is not int
            or stride.get('code')!=1 or stride.get('operand')!='rs1'
            or stride.get('offset')!=16 or stride.get('state',{}).get('value')!=1):
        raise ReadoutUnavailable('wide MVIN2 decoded width/block/stream addressing is unknown')
    decoded_max=(1<<cols['width'])-1
    dma=facts['facts']['timing_structure']['memory_clients']['clients']['dma_read']
    request_bytes=dma.get('max_request_bytes')
    if type(request_bytes) is not int or request_bytes<edge*problem.rhs_bytes:
        raise ReadoutUnavailable('selected DMA request width cannot aggregate complete RHS tiles')
    max_tiles=min(decoded_max//edge,request_bytes//(edge*problem.rhs_bytes))
    if max_tiles<2:raise ReadoutUnavailable('selected decoded transfer width admits only one tile')
    law=TileLoadAddressLaw(edge,edge,decoded_max,problem.rhs_bytes,problem.n*problem.rhs_bytes,
        edge,'selected-command-ordering:'+pin['ordering_sha256']+':MVIN2-exact-tile-block-stride')
    storage=original._storage
    # Every final load must have a complete mapping, even though only RHS loads
    # are offered to the proposal. Original LHS commands remain byte-for-byte.
    lhs_law=replace(law,element_bytes=problem.lhs_bytes,source_pitch_bytes=problem.k*problem.lhs_bytes,
                    identity=law.identity+':unchanged-lhs')
    def project(call):
        if call.instr==DRAIN:return None
        op=adapter.project(call)
        if type(op) is not Load:return None
        left=op.role=='lhs';pitch=problem.k if left else problem.n
        width=problem.lhs_bytes if left else problem.rhs_bytes
        return TileLoadMapping(op.role,(storage.lhs if left else storage.rhs).backing,
            (op.dram_row*pitch+op.dram_col)*width,op.row,op.rows,op.cols,lhs_law if left else law)
    def rhs_project(call):
        return project(call) if call.instr=='gemmini.mvin.weight' else None
    def encode(call,columns):
        return replace(call,args=tuple((k,Const(columns) if k=='cols' else v) for k,v in call.args))
    def encode_band(call,mapping):
        op=adapter.project(call);pitch=problem.k if op.role=='lhs' else problem.n
        width=problem.lhs_bytes if op.role=='lhs' else problem.rhs_bytes
        # Source-row operands are unchanged; derive exact column from byte offset.
        c,ts=_linear(mapping.source_byte+op.dram_row*(-pitch*width))
        col=add(Const(c),*(Var(k)*v for k,v in ts.items()))
        if width!=1:raise ReadoutUnavailable('wide dense transfer requires selected i8 elements')
        constant,terms=_linear(col+op.dram_col*(-1))
        if constant%edge or any(v%edge for v in terms.values()):
            raise ReadoutUnavailable('wide checking band lost its original logical tile coordinate')
        delta=add(Const(constant//edge),*(Var(k)*(v//edge) for k,v in terms.items()))
        return adapter.encode(replace(op,block=(op.block[0],op.block[1]+delta),
            dram_col=col,row=mapping.local_row,rows=mapping.rows,cols=mapping.columns))
    proposal,changes=propose_tile_load_coalescing(original.kernel,project=rhs_project,encode=encode,
        max_tiles=max_tiles,coalesce_tail=True)
    if proposal==original.kernel:
        raise ReadoutUnavailable('no original complete RHS column load run admits aggregation')
    bm,bk,bn=receipt['block']
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=bm,max_k_blocks=bk,max_n_blocks=bn,
        partial_output_tiles=True,partial_output_blocks=True)
    block=next(b for b in dense_segmented_blocks(problem,resources,storage)
               if (b.m,b.k,b.n)==(bm,bk,bn))
    try:
        checked=admit_ready_readout_tile_loads(original,proposal,problem,block,resources,storage,adapter,
            project=project,encode_band=encode_band,static_budget=256,timeout_ms=60000)
    except ValueError as error:
        raise ReadoutUnavailable('original ready owner declined final wide transfers: '+str(error)) from error
    result=DenseTransferResult(checked);result.revalidate()
    receipt.update(mode='segmented_coalesced',before_transfer_schedule_sha256=original.kernel.digest(),
        transformed_schedule_sha256=result.kernel.digest(),transfer_address_law=asdict(law),
        proposal=changes,max_tiles=max_tiles,selected_dma_request_bytes=request_bytes,
        actual_transfer_projection=checked.realization['tile_load_projection'],qualification_identity=None)
    return result,receipt
