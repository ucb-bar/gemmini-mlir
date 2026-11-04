"""Experimental final RHS band coalescing; never grants completion semantics.

Pass order: checked retained/segmented/ready owner -> address-only proposal ->
full original causal-readout admission of EVERY decoded final band -> terminal
LLVM. The proposal changes RHS load width only. Existing drains and effects
stay in their original order. Source pitch is bytes; local stride is rows.

Selected command_ordering extent probes decode cols at rs2[38:32], rows at
rs2[52:48], and load-id-specific CONFIG_LD block stride rs1[31:16]. The public
reference gemmini.h block load layout is (k*J+j)*DIM, with contiguous source
columns; MAX_BYTES64 in gemmini_params.h bounds this experiment to four tiles.
Use the intersection (64 columns,16 rows), NOT reserved packet field widths.
This is a conditional exact-address adapter, not a synchronous transfer law.
"""
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
from merlin.sched.ir import Const,Call
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing
from merlin.compile.scheduling.panel_tile_loads import _offset_form
from merlin.compile.scheduling.dense_ready_readout import admit_ready_readout_tile_loads

PUBLIC_REFERENCE_EVIDENCE={'gemmini.h': {'path': 'reference-kernels/originals/upstream-exo/public-reference/headers/gemmini.h', 'sha256': '06988403de641bbf22fea84e76debd863ab3185498f2b48ceb4b475d4af0914e'}, 'gemmini_params.h': {'path': 'reference-kernels/originals/upstream-exo/public-reference/headers/gemmini_params.h', 'sha256': 'd6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5'}}

class WideRHSResult:
    """Emission owner for the real checked wide kernel, using caller's encoder."""
    def __init__(self,checked,receipt):
        self.checked=checked
        self.receipt=receipt
    @property
    def kernel(self): return self.checked.kernel
    @property
    def realization(self): return self.checked.realization
    def revalidate(self): return self.checked.revalidate()
    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        receipt=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        self.receipt["actual_emission_applied"]=True
        self.receipt["emitted_kernel_sha256"]=self.kernel.digest()
        if "lhs_loads" in self.receipt:
            self.receipt["lhs_loads"]["proposal_consumed"]=True
            self.receipt["lhs_loads"]["actual_emission_applied"]=bool(self.receipt["lhs_loads"]["changes"])
            self.receipt["lhs_loads"]["emitted_kernel_sha256"]=self.kernel.digest()
            receipt["wide_lhs_loads"]=self.receipt["lhs_loads"]
        receipt["wide_rhs_loads"]=self.receipt
        return receipt

class WideRHSAdapter:
    def __init__(self,problem,storage,adapter,*,facts_root=None):
        facts_root=Path(__file__).resolve().parent if facts_root is None else Path(facts_root)
        selection=json.loads((facts_root/"command_ordering.json").read_text())
        ordering=selection["ordering"]
        if sha256(json.dumps(ordering,sort_keys=True).encode()).hexdigest()!=selection["ordering_sha256"]:
            raise ValueError("selected ordering digest changed")
        if sha256((facts_root/"facts.json").read_bytes()).hexdigest()!=selection["base_facts"]["sha256"]:
            raise ValueError("selected facts digest changed")
        extent=ordering["extents"]["MVIN2.spad_addr"]; blocks=extent["blocks"]
        if (extent["status"]!="derived" or extent["rows"]["offset"]!=48 or extent["rows"]["width"]!=5
            or blocks["block"]!=16 or blocks["cols"]["offset"]!=32 or blocks["cols"]["width"]!=7
            or blocks["stride"]["offset"]!=16 or blocks["stride"]["width"]!=16
            or blocks["stride"]["state"]!={"operand":"rs1","offset":3,"width":2,"value":1}):
            raise ValueError("wide RHS requires independently decoded selected load extents/stride")
        self.evidence={"public_reference":PUBLIC_REFERENCE_EVIDENCE,"ordering_sha256":selection["ordering_sha256"],"decoded_extent":extent,"maximum_columns":64,"maximum_rows":16,"block_stride_rows":16}
        if (problem.tile_m,problem.tile_k,problem.tile_n)!=(16,16,16):
            raise ValueError('wide RHS requires ordinary selected16 square views')
        self.p,self.storage,self.adapter=problem,storage,adapter
        self.law=TileLoadAddressLaw(16,16,64,1,problem.n,16,
            'selected-derived-MVIN2-extents7col5row-config-stride16-reference-MAX_BYTES64-v1')
    def project(self,call):
        if not isinstance(call,Call) or call.instr not in ('weight_load','lhs_load'): return None
        a=dict(call.args); rhs=call.instr=='weight_load'
        law=self.law if rhs else replace(self.law,source_pitch_bytes=self.p.k,identity=self.law.identity+':lhs-primitive')
        return TileLoadMapping('weight' if rhs else 'lhs',self.storage.rhs.backing if rhs else self.storage.lhs.backing,a['r']*law.source_pitch_bytes+a['c'],a['local'],a['rows'],a['cols'],law)
    def eligible(self,call):
        return self.project(call) if isinstance(call,Call) and call.instr=='weight_load' else None
    def encode(self,call,columns):
        return replace(call,args=tuple((k,Const(columns) if k=='cols' else v) for k,v in call.args))
    def band(self,call,band):
        before=self.project(call)
        delta,terms=_offset_form(band.source_byte+before.source_byte*(-1))
        if terms or delta%16: raise ValueError('nonconstant final source band offset')
        a=dict(call.args)
        a.update(c=a['c']+delta,bj=a['bj']+delta//16,local=band.local_row,rows=band.rows,cols=band.columns)
        return replace(call,args=tuple((k,a[k]) for k,_ in call.args))
    def transform(self,original,block,resources,*,timeout_ms=20000):
        original.revalidate()
        proposal,receipt=propose_tile_load_coalescing(original.kernel,project=self.eligible,encode=self.encode,max_tiles=4,coalesce_tail=True)
        from dense_lhs_loads import propose_lhs
        proposal,lhs_receipt,lhs_codec=propose_lhs(proposal,self.p,self.storage,self)
        checked=admit_ready_readout_tile_loads(original,proposal,self.p,block,resources,self.storage,self.adapter,
            project=self.project if lhs_codec is None else lhs_codec.project,
            encode_band=self.band if lhs_codec is None else lhs_codec.band,static_budget=256,timeout_ms=timeout_ms)
        if lhs_receipt is not None:receipt["lhs_loads"]=lhs_receipt
        checked.revalidate()
        receipt["target_addressing_evidence"]=self.evidence
        receipt["actual_emission_applied"]=False
        return WideRHSResult(checked,receipt),receipt

KNOB_DECLARATION={'dense_wide_loads':{
    'values':['primitive','coalesced4'],
    'mechanism':'final address-only RHS tile coalescing under original ready-readout admission',
    'note':'primitive default; coalesced4 bounds each actual load to64 columns and retains every explicit completion. No cycle claim.'}}

def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('dense_wide_loads',default='primitive',values=('primitive','coalesced4'),root=Path(__file__).resolve().parent)

def maybe_coalesce(original,problem,resources,storage,adapter,info,*,mode=None,facts_root=None):
    """Final wide-builder hook; resources must be its actual bounded limits.

    No source transpose or numeric configuration is introduced. Existing common
    terminal callback owns identity i8 loads, including configured stride16.
    The descriptor cap64 is a conservative intersection of decoded7-bit width
    and public reference MAX_BYTES64; it is not a universal hardware maximum.
    Refusal propagates to the builder's existing checked fallback policy.
    """
    mode=setting() if mode is None else mode
    if mode=='primitive':
        from dense_lhs_loads import maybe_lhs
        return maybe_lhs(original,problem,resources,storage,adapter,info)
    if mode!='coalesced4': raise ValueError('unknown dense wide load mode')
    from merlin.transforms.dense_planning import DenseBlock
    owner,receipt=coalesce_ready(original,problem,DenseBlock(**info['block']),resources,storage,adapter,facts_root=facts_root)
    info=dict(info,wide_load_coalescing=receipt)
    if "lhs_loads" in receipt:info["lhs_load_coalescing"]=receipt["lhs_loads"]
    return owner,info


def coalesce_ready(original,problem,block,resources,storage,adapter,*,facts_root=None):
    """Explicit opt-in reuse on any original checked segmented-ready owner.

    No shape/name selector or new ordering permission. Supply the actual
    realized block and exact resources from the original builder. The library
    exact-owner gate prevents accidental second wrapping or widened premises.
    """
    return WideRHSAdapter(problem,storage,adapter,facts_root=facts_root).transform(original,block,resources)
