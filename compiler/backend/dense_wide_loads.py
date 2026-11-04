"""Selected source-backed MVIN addressing and checked wide dense loads.

Source audit (the FIRRTL module bodies are identical in the selected target and
its aligned simulator, as recorded by facts.simulation_correspondence):
LoadController.scala 45/52/61-76/102-111/147-151 decodes rows/columns,
selects channel state and sends base+r*pitch, local+r, columns, block stride
and scale. DMA.scala 163/267-274 partitions each row's contiguous bytes;
BeatMerger.scala 48/57-83 places byte c at local+floor(c/16)*block_stride,
lane c%16 with exact masks. Scratchpad.scala 391-428/576-598 preserves those
addresses/masks and the original lane-wise scale path. PixelRepeater.scala
82-84 makes configured repeat=1 an ordinary unrepeated row. This adapter
retains that exact scale path and channel state, and never changes arithmetic.

The accepted width is conservatively capped at 64 bytes: StreamReaderCore's
bytesRequested register is six bits (DMA.scala 162). The seven decoded column
bits alone are insufficient to admit 127 columns. Full rows are at most DIM.
No synchronous completion law is invented; ALL actual wide bands are checked
by the original panel source/lifetime/epoch verifier before actual emission.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelPipelineResult
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult
from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads, admit_segmented_tile_loads
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, _linear, propose_tile_load_coalescing,
)
from merlin.sched.ir import Const

from . import isa

TARGET_CORE_SHA256 = '042bdccfd29e8cafd43fe1d4fa76dc6a9013ed0865cb2eb48d4b1c0f512c6e8d'
SIMULATION_FIR_SHA256 = '6748b9756799c086a6fb06e6d80b4b052f2771f9eb1858c172e14dcf441ba149'
ADDRESS_IDENTITY = 'gemmini-fire-aligned-i8-mvin-cols64-blockstride16:' + TARGET_CORE_SHA256


def selected_addressing_evidence(facts):
    """Bind this source audit to its independently recorded selected RTL.

    These hashes identify audited hardware, not a certificate inferred from
    observed expert transfer widths or an arbitrary caller-authored law.
    """
    from .dense_load_coalescing import _facts_digest
    digest = _facts_digest(facts)
    f = facts['facts']
    corr = f.get('simulation_correspondence', {})
    target = corr.get('configs', {}).get('target', {})
    simulation = corr.get('configs', {}).get('simulation', {})
    bodies = corr.get('firrtl_module_bodies', {})
    if (f.get('target') != 'gemmini'
            or facts.get('inputs', {}).get('core_hw_sha256') != TARGET_CORE_SHA256
            or target.get('core_hw', {}).get('sha256') != TARGET_CORE_SHA256
            or simulation.get('firrtl', {}).get('sha256') != SIMULATION_FIR_SHA256
            or corr.get('status') != 'structurally_equivalent'
            or bodies.get('compared') != 110 or bodies.get('missing') != [] or bodies.get('differ') != []
            or isa.DIM != 16 or isa.SPAD_ROWS != 16384):
        raise ValueError('wide MVIN addressing source audit does not match selected RTL facts')
    return dict(identity=ADDRESS_IDENTITY, facts_document_sha256=digest,
        target_core_hw_sha256=TARGET_CORE_SHA256, simulation_firrtl_sha256=SIMULATION_FIR_SHA256,
        source='selected simulation FIRRTL; all 110 unit module bodies identical to target',
        decoded_column_bits=7, decoded_row_bits=5, admitted_max_columns=64, admitted_max_rows=16,
        local_block_stride=16, element_bytes=1,
        mapping='source_byte+r*source_pitch_bytes+c -> local_row+(c//16)*16+r, lane c%16',
        numerical='same original i8 lane-wise load path; scale=1, pixel_repeats=1, scratchpad destination',
        completion='original explicit drains and bounded panel admission retained')


@dataclass(frozen=True)
class DenseWideLoadCodec:
    problem: object
    storage: object
    adapter: DenseLoopAdapter
    identity: str = ADDRESS_IDENTITY

    def __post_init__(self):
        p, s = self.problem, self.storage
        if (type(self.adapter) is not DenseLoopAdapter
                or self.adapter.configuration_identity != 'gemmini-fire-aligned-dim16-ws'
                or self.adapter.numerical_policy_identity != 'i8-i8-i32-modular'
                or (p.tile_m, p.tile_k, p.tile_n, p.lhs_bytes, p.rhs_bytes) != (16,16,16,1,1)
                or s.lhs.backing != 'spad' or s.rhs.backing != 'spad'
                or s.lhs.unit_bytes != 16 or s.rhs.unit_bytes != 16):
            raise ValueError('wide dense loads require original contiguous i8 DIM16 scratchpad layout')
        self.adapter.__post_init__()

    def project(self, call):
        if call.instr not in ('gemmini_mvin_lhs', 'gemmini_mvin_weight'):
            return None
        operation = self.adapter.project(call)
        if (type(operation) is not Load or operation.role not in ('lhs','weight')
                or self.adapter.encode(operation) != call or operation.step != Const(1)):
            raise ValueError('wide dense load requires complete actual original load codec')
        left = operation.role == 'lhs'
        pitch = self.problem.k if left else self.problem.n
        view = self.storage.lhs if left else self.storage.rhs
        law = TileLoadAddressLaw(16,16,64,1,pitch,16,self.identity)
        return TileLoadMapping(operation.role,view.backing,
            operation.dram_row*pitch+operation.dram_col,operation.row,
            operation.rows,operation.cols,law)

    def encode(self, first, columns):
        if type(columns) is not int or not 0 < columns <= 64:
            raise ValueError('wide MVIN width exceeds independently audited 64-byte bound')
        self.project(first)
        return self.adapter.encode(replace(self.adapter.project(first), cols=Const(columns)))

    def encode_band(self, actual, band):
        """Build a checking band, including its TRUE logical block coordinate."""
        mapping = self.project(actual)
        if mapping is None or band.law != mapping.law:
            raise ValueError('wide MVIN checking band lacks original selected channel state')
        ac, av = _linear(mapping.source_byte)
        bc, bv = _linear(band.source_byte)
        offset = bc-ac
        if av != bv or offset < 0 or offset%16:
            raise ValueError('wide MVIN checking band is not an exact contiguous tile')
        op = self.adapter.project(actual)
        return self.adapter.encode(replace(op,
            block=(op.block[0],op.block[1]+offset//16),
            dram_col=op.dram_col+offset,row=band.local_row,rows=band.rows,cols=band.columns))


@dataclass(frozen=True)
class DenseWideLoadResult:
    """Emission owner for the actual admitted wide kernel, never its expansion."""
    checked: object
    realization: dict
    facts: dict
    facts_digest: str

    @property
    def kernel(self):
        return self.checked.kernel

    @property
    def admission(self):
        return self.checked.admission

    def revalidate(self):
        from .dense_load_coalescing import _facts_digest
        if _facts_digest(self.facts) != self.facts_digest:
            raise ValueError('selected wide MVIN facts changed after admission')
        selected_addressing_evidence(self.facts)
        self.checked.revalidate()

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        result = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return result


def select_dense_wide_loads(original, problem, block, resources, storage, adapter, facts, *,
                            selection='off', max_tiles=4, static_budget=128,
                            timeout_ms=5000, max_static_nodes=4096):
    """Call before store-epoch rewriting, using the actual checked panel inputs.

    The caller emits unchanged CONFIG_LD(k, channel=0) and CONFIG_LD(n,
    channel=1), scale=1, pixel_repeats=1, block_stride=16; the ordinary dense
    terminal callback already packs the resulting Load.cols into real MVIN.
    This function does not change configuration or completion. Unsupported
    checked owner types retain their original route with an explicit refusal.
    """
    if selection not in ('off','on'):
        raise ValueError('DENSE_WIDE_LOADS must be off or on')
    if selection == 'off':
        return original, None
    if type(original) not in (DensePanelPipelineResult, DenseSegmentedPipelineResult):
        return original, dict(selection='on',applied=False,
            refusal='wide MVIN temporal consumer requires original checked panel or segmented result')
    original.revalidate()
    evidence = selected_addressing_evidence(facts)
    codec = DenseWideLoadCodec(problem,storage,adapter)
    proposal, report = propose_tile_load_coalescing(original.kernel,
        project=codec.project, encode=codec.encode,max_tiles=max_tiles,
        max_static_nodes=max_static_nodes,coalesce_tail=True)
    if not report['changes']:
        return original, dict(selection='on',applied=False,refusal='no contiguous full-tile load-only loop',
                             addressing=evidence,proposal=report)
    admit = (admit_segmented_tile_loads if type(original) is DenseSegmentedPipelineResult
             else admit_panel_tile_loads)
    checked = admit(original,proposal,problem,block,resources,storage,adapter,
        project=codec.project,encode_band=codec.encode_band,static_budget=static_budget,
        timeout_ms=timeout_ms,max_static_nodes=max_static_nodes)
    receipt = dict(selection='on',applied=True,addressing=evidence,proposal=report,
        tile_load_projection=checked.realization['tile_load_projection'],
        complete_call_cycles='UNKNOWN',target_execution='not_run')
    result = DenseWideLoadResult(checked,dict(original.realization,
        kernel_sha256=proposal.digest(),wide_loads=receipt),facts,evidence['facts_document_sha256'])
    result.revalidate()
    return result, receipt
