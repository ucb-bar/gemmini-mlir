"""Exact final wide-load composition for the checked segmented owner.

The caller must supply a selected physical address law. Packet field width is
not that law. This module never selects a speculative maximum DMA width.
"""
from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Const
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, expand_tile_load_mapping,
    propose_tile_load_coalescing,
)
from merlin.compile.scheduling.dense_segmented_pipeline import admit_segmented_tile_loads


@dataclass(frozen=True)
class CoalescedPanel:
    checked: object
    proposal: dict

    @property
    def kernel(self):
        return self.checked.kernel

    @property
    def realization(self):
        return {**self.checked.realization, 'coalescing': self.proposal}

    def revalidate(self):
        self.checked.revalidate()

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        report = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return report


def coalesce_panel(result, problem, block, resources, storage, adapter, *,
                   weight_law, static_budget=128, timeout_ms=15000):
    """Coalesce native LHS and RHS columns; preserve original K epochs.

    ``weight_law`` must describe selected physical source pitch, local tile
    stride, complete row/column bounds and exact-copy semantics. Both operands
    are independently projected during the final full check.
    """
    if type(weight_law) is not TileLoadAddressLaw:
        raise ValueError('an explicit selected physical weight load law is required')
    if (weight_law.element_bytes != problem.rhs_bytes
            or weight_law.source_pitch_bytes != problem.n * problem.rhs_bytes
            or weight_law.tile_columns != problem.tile_n
            or weight_law.local_block_stride != problem.tile_k):
        raise ValueError('selected weight law differs from original panel layout')
    lhs_law = replace(weight_law, tile_columns=problem.tile_k,
                      max_columns=weight_law.max_columns,
                      source_pitch_bytes=problem.k * problem.lhs_bytes,
                      element_bytes=problem.lhs_bytes,
                      local_block_stride=problem.tile_m)

    def project(call):
        if call.instr == DRAIN:
            return None
        op = adapter.project(call)
        if type(op) is not Load:
            return None
        left = op.role == 'lhs'
        view = storage.lhs if left else storage.rhs
        law = lhs_law if left else weight_law
        return TileLoadMapping(op.role, view.backing,
            op.dram_row * law.source_pitch_bytes + op.dram_col * law.element_bytes,
            op.row, op.rows, op.cols, law)

    def encode(first, columns):
        # Keep all target operands and metadata; the decoder above checks the
        # actual width along with the actual source and local address.
        return replace(first, args=tuple((key, Const(columns) if key == 'cols' else value)
                                        for key, value in first.args))

    def encode_band(actual, band):
        original = project(actual)
        if type(original.columns) is not Const:
            if original == band:
                return actual
            raise ValueError('dynamic checking tail differs from its actual primitive load')
        for index, candidate in enumerate(expand_tile_load_mapping(original)):
            if candidate == band:
                args = dict(actual.args)
                args.update(tile1=args['tile1'] + index,
                            col=args['col'] + index * original.law.tile_columns,
                            dst=args['dst'] + index * original.law.local_block_stride,
                            rows=band.rows, cols=band.columns)
                return replace(actual, args=tuple((key, args[key]) for key, _ in actual.args))
        raise ValueError('checking band does not belong to the actual native load')

    proposal, report = propose_tile_load_coalescing(result.kernel,
        project=project, encode=encode,
        max_tiles=weight_law.max_columns // weight_law.tile_columns, coalesce_tail=True)
    if not report['changes']:
        return result
    checked = admit_segmented_tile_loads(result, proposal, problem, block, resources,
        storage, adapter, project=project, encode_band=encode_band,
        static_budget=static_budget, timeout_ms=timeout_ms)
    return CoalescedPanel(checked, report)
