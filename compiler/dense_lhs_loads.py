"""Opt-in retained-LHS K-band DMA coalescing under the original ready owner.

Only contiguous original lhs_load runs are proposed for replacement. The
selected MVIN0 rectangle/stride decode and public MAX_BYTES64 intersection
bound four16-column bands; actual block-major local rows remain explicit.
The complete final proposal (including optional existing RHS coalescing) is
expanded back into EVERY original primitive band, then admitted through the
unchanged causal-ready verifier. Source axes, M/K tails, physical placement,
RAW/WAR, accumulator continuation and drains are never inferred from adjacency.
One common terminal effect encoder emits the verified actual wider calls.
"""
from dataclasses import replace
from pathlib import Path
from merlin.sched.ir import Call,Const
from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw,TileLoadMapping,propose_tile_load_coalescing
from merlin.compile.scheduling.panel_tile_loads import _offset_form
from merlin.compile.scheduling.dense_ready_readout import admit_ready_readout_tile_loads

KNOB_DECLARATION={'dense_lhs_loads':{
    'values':['off','coalesced4'],
    'mechanism':'actual retained-LHS K-band DMA coalescing under original complete ready-readout proof',
    'note':'off default; coalesced4 emits up to64 LHS columns with selected MVIN0 stride16, preserving all M/K tails, disjoint rows and original completions. No timing claim.'}}


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('dense_lhs_loads',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)


class LHSBandCodec:
    """Decode both operands; independently bind the selected LHS load-id law."""
    def __init__(self,problem,storage,rhs_codec):
        self.p,self.storage,self.rhs_codec=problem,storage,rhs_codec
        import json
        # RHS codec already checks selected ordering/facts digests. LHS fields
        # are separately checked rather than borrowing MVIN2's configuration.
        root=Path(__file__).resolve().parent
        selected=json.loads((root/'command_ordering.json').read_text())
        extent=selected['ordering']['extents']['MVIN.spad_addr'];blocks=extent['blocks']
        if (extent['status']!='derived' or extent['rows']['offset']!=48 or extent['rows']['width']!=5
            or blocks['block']!=16 or blocks['cols']['offset']!=32 or blocks['cols']['width']!=7
            or blocks['stride']['offset']!=16 or blocks['stride']['width']!=16
            or blocks['stride']['state']!={'operand':'rs1','offset':3,'width':2,'value':0}):
            raise ValueError('LHS bands require independently selected MVIN0 rectangle/stride decode')
        self.law=TileLoadAddressLaw(16,16,64,1,problem.k,16,
            'selected-derived-MVIN0-extents7col5row-config-stride16-reference-MAX_BYTES64-v1')
        self.evidence={'ordering_sha256':selected['ordering_sha256'],'decoded_extent':extent,
            'public_reference':rhs_codec.evidence['public_reference'],
            'maximum_columns':64,'maximum_rows':16,'block_stride_rows':16}

    def project(self,call):
        if not isinstance(call,Call) or call.instr!='lhs_load':return self.rhs_codec.project(call)
        a=dict(call.args)
        return TileLoadMapping('lhs',self.storage.lhs.backing,a['r']*self.p.k+a['c'],
            a['local'],a['rows'],a['cols'],self.law)

    def eligible(self,call):
        return self.project(call) if isinstance(call,Call) and call.instr=='lhs_load' else None

    def encode(self,call,columns):
        return replace(call,args=tuple((key,Const(columns) if key=='cols' else value) for key,value in call.args))

    def band(self,call,band):
        if call.instr!='lhs_load':return self.rhs_codec.band(call,band)
        before=self.project(call);delta,terms=_offset_form(band.source_byte+before.source_byte*(-1))
        if terms or delta%16:raise ValueError('LHS checking band requires exact constant K-tile offset')
        a=dict(call.args)
        a.update(c=a['c']+delta,bj=a['bj']+delta//16,local=band.local_row,rows=band.rows,cols=band.columns)
        return replace(call,args=tuple((key,a[key]) for key,_ in call.args))


def propose_lhs(kernel,problem,storage,rhs_codec,*,mode=None):
    """Return proposed actual kernel, diagnostic and codec; grants no execution."""
    mode=setting() if mode is None else mode
    if mode=='off':return kernel,None,None
    if mode!='coalesced4':raise ValueError('unknown retained LHS load mode')
    codec=LHSBandCodec(problem,storage,rhs_codec)
    normalized,normalization=simplify_singleton_domains(kernel)
    actual,receipt=propose_tile_load_coalescing(normalized,project=codec.eligible,
        encode=codec.encode,max_tiles=4,coalesce_tail=True)
    receipt.update(target_addressing_evidence=codec.evidence,actual_emission_applied=False,
        singleton_domain_normalization=normalization)
    return actual,receipt,codec


class LHSLoadsResult:
    def __init__(self,checked,receipt):self.checked,self.receipt=checked,receipt
    @property
    def kernel(self):return self.checked.kernel
    @property
    def realization(self):return self.checked.realization
    def revalidate(self):return self.checked.revalidate()
    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        report=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate();self.receipt['proposal_consumed']=True
        self.receipt['actual_emission_applied']=bool(self.receipt['changes'])
        self.receipt['emitted_kernel_sha256']=self.kernel.digest()
        report['wide_lhs_loads']=self.receipt
        return report


def maybe_lhs(original,problem,resources,storage,adapter,info,*,mode=None):
    """LHS-only path; the combined RHS path uses propose_lhs before ONE admit."""
    mode=setting() if mode is None else mode
    if mode=='off':return original,info
    from dense_wide_loads import WideRHSAdapter
    rhs_codec=WideRHSAdapter(problem,storage,adapter)
    actual,receipt,codec=propose_lhs(original.kernel,problem,storage,rhs_codec,mode=mode)
    from merlin.transforms.dense_planning import DenseBlock
    checked=admit_ready_readout_tile_loads(original,actual,problem,DenseBlock(**info['block']),
        resources,storage,adapter,project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
    checked.revalidate()
    return LHSLoadsResult(checked,receipt),dict(info,lhs_load_coalescing=receipt)


def simplify_singleton_domains(kernel):
    """Use the shared semantic normal form; original owner still admits loads."""
    from merlin.compile.scheduling.tile_load_coalescing import normalize_singleton_domains
    return normalize_singleton_domains(kernel)
