"""Coalesce original RHS loads while retaining strided LHS source indexing.

The RHS is the original contiguous CI/CO source. Its independently audited
MVIN law is identical to dense RHS loads. LHS rows have a different source
pitch and scanline discontinuities: this codec deliberately does not project
or rewrite them in the proposal. The checked expansion retains their original
logical matrix source map; lower_strided_pointwise alone binds those unchanged
logical LHS requests to exact Window2D source runs. Thus the wide audit applies
only to RHS, and never asserts a contiguous physical LHS matrix. M tiles may be any complete scanline divisor <= DIM because
only RHS has physical DIM-by-DIM tiles here.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
from .dense_wide_loads import DenseWideLoadCodec, DenseWideLoadResult, selected_addressing_evidence
from . import isa


@dataclass(frozen=True)
class PointwiseWeightCodec(DenseWideLoadCodec):
    def __post_init__(self):
        # Invoke the existing full codec validation on a DIM-row problem: the
        # only relaxed premise concerns LHS rows, which project() never admits.
        p=self.problem
        if not 0 < p.tile_m <= isa.DIM:
            raise ValueError('pointwise row tile must fit a single physical row block')
        DenseWideLoadCodec(replace(p,tile_m=isa.DIM),self.storage,self.adapter)

    def project(self,call):
        mapping=super().project(call)
        if mapping is not None and call.instr=='gemmini_mvin_lhs':
            # This is the original schedule's logical sampled matrix, not
            # an assertion that the terminal physical LHS has K-byte pitch.
            mapping=replace(mapping,law=replace(mapping.law,
                identity='pointwise-logical-sampled-matrix:unchanged-Window2D-terminal'))
        return mapping

    def project_weight(self,call):
        return self.project(call) if call.instr=='gemmini_mvin_weight' else None


def select_pointwise_weight_loads(original,problem,block,resources,storage,adapter):
    selection=knob('POINTWISE_STRIDED_WIDE',default='off',values=('off','on'),
                   root=Path(__file__).resolve().parents[1])
    if selection=='off':
        return original
    try:
        original.revalidate()
        facts=json.loads(Path(__file__).with_name('dense_load_facts.json').read_text())
        evidence=selected_addressing_evidence(facts)
        codec=PointwiseWeightCodec(problem,storage,adapter)
        proposal,report=propose_tile_load_coalescing(original.kernel,project=codec.project_weight,
            encode=codec.encode,max_tiles=4,max_static_nodes=4096,coalesce_tail=True)
        if not report['changes']:
            return replace(original,realization=dict(original.realization,pointwise_weight_wide={
                'applied':False,'refusal':'no contiguous full-tile RHS load-only loop','proposal':report}))
        checked=admit_panel_tile_loads(original,proposal,problem,block,resources,storage,adapter,
            project=codec.project,encode_band=codec.encode_band,static_budget=128,
            timeout_ms=30000,max_static_nodes=4096)
        receipt=dict(applied=True,addressing=evidence,proposal=report,
            tile_load_projection=checked.realization['tile_load_projection'],
            lhs='logical sampled-matrix projection only; exact Window2D terminal loads unchanged')
        return DenseWideLoadResult(checked,dict(original.realization,kernel_sha256=proposal.digest(),
            pointwise_weight_wide=receipt),facts,evidence['facts_document_sha256'])
    except (ValueError,TimeoutError) as error:
        return replace(original,realization=dict(original.realization,pointwise_weight_wide={
            'applied':False,'refusal':type(error).__name__+': '+str(error)}))
