"""Exact block DMA for invocation-local convolution RHS preparation.

The admitted raw gemmini.h move-in-B loop maps each adjacent N tile to
local + DIM rows, including a partial final K row. Spatial panels independently
use that same reduction-major, N-inner layout. No compute, input, readout or
completion moves: the transformed product supplies only the original weight
preparation calls before their existing full phase-ready drain.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256

from merlin.compiler.schedule.tile_load_coalescing import (
    TileLoadAddressLaw, TileLoadMapping, propose_tile_load_coalescing,
)
from merlin.compiler.schedule.panel_tile_loads import expand_panel_tile_loads
from merlin.compiler.schedule.llvm_program import emit_index_program
from merlin.compiler.kernel_ir.ir import Const, Kernel, Loop, Var, call, concretize, instances
from .dense_rhs_transfers import REFERENCE_FILES
from . import isa


@dataclass(frozen=True)
class ConvRhsTransfers:
    kernel: object
    realization: dict

    def emit(self, builder, lower, phase_column):
        if self.kernel.digest() != self.realization['actual_kernel_sha256']:
            raise ValueError('convolution RHS product changed after all-band admission')
        # Only a common original N-phase translation is added. The checked
        # reduction/local layout and every actual width are consumed verbatim.
        def effect(kind, fields):
            if kind != 'load_weight':
                raise ValueError('convolution weight preparation has a non-weight effect')
            lower(kind, dict(fields, output_column=builder.op(
                'add', phase_column, fields['output_column'])))
        receipt = emit_index_program(self.kernel, builder, effect)
        if self.kernel.digest() != self.realization['actual_kernel_sha256']:
            raise ValueError('convolution RHS product changed during emission')
        return receipt


@lru_cache(maxsize=128)
def plan_spatial_rhs_transfers(kh, kw, ci, pitch, rhs_base, widths):
    """Check every final band against every original primitive RHS effect.

    At most SPAD_ROWS / DIM primitive panels fit this private phase. Enumerating
    this bounded address proof depends on private capacity, never image size.
    The ordinary convolution owner retains source immutability, all K/N compute
    consumers, complete preparation and phase-last-use completion.
    """
    if (any(type(v) is not int or v <= 0 for v in (kh, kw, ci, pitch))
            or type(rhs_base) is not int or rhs_base < isa.DIM
            or type(widths) is not tuple or not widths
            or any(type(v) is not int or not 0 < v <= isa.DIM for v in widths)
            or any(v != isa.DIM for v in widths[:-1])):
        raise ValueError('convolution block DMA requires exact original regular N bands')
    active = len(widths)
    chunks = (ci + isa.DIM - 1) // isa.DIM
    bands = kh * kw * chunks * active
    if rhs_base + bands * isa.DIM > isa.SPAD_ROWS or sum(widths) > pitch:
        raise ValueError('convolution block DMA exceeds its original private panel')
    full_n = sum(width == isa.DIM for width in widths)
    if full_n < 2:
        return None
    y, x, c, n = map(Var, ('kernel_y', 'kernel_x', 'channel_chunk', 'output_tile'))
    tap = y * kw + x

    def reduction(chunk, depth):
        slot = tap * chunks + chunk
        start = tap * ci + chunk * isa.DIM
        def weight(index, columns):
            return call('load_weight', reduction_row=start,
                output_column=index * isa.DIM,
                local_row=Const(rhs_base) + (slot * active + index) * isa.DIM,
                rows=depth, columns=columns)
        body = [Loop('output_tile', full_n, (weight(n, isa.DIM),))]
        if full_n < active:
            body.append(weight(Const(full_n), widths[-1]))
        return tuple(body)

    whole, tail = divmod(ci, isa.DIM)
    reductions = []
    if whole:
        reductions.append(Loop('channel_chunk', whole, reduction(c, isa.DIM)))
    if tail:
        reductions.extend(reduction(Const(whole), tail))
    original = Kernel('conv_spatial_rhs_preparation', (), (
        Loop('kernel_y', kh, (Loop('kernel_x', kw, tuple(reductions)),)),))
    identity = sha256(repr(sorted(REFERENCE_FILES.items())).encode()).hexdigest()
    law = TileLoadAddressLaw(isa.DIM, isa.DIM, 64, 1, pitch, isa.DIM,
        'selected-gemmini-software-block-mvin:' + identity)

    def project(effect):
        if effect.instr != 'load_weight':
            return None
        f = dict(effect.args)
        return TileLoadMapping('original_convolution_rhs', 'private_spad',
            f['reduction_row'] * pitch + f['output_column'], f['local_row'],
            f['rows'], f['columns'], law)

    def encode(effect, columns):
        return replace(effect, args=tuple((k, Const(columns) if k == 'columns' else v)
                                         for k, v in effect.args))

    def encode_band(effect, band):
        f = dict(effect.args)
        # Independent final-call projection supplies the exact byte delta.
        delta = band.source_byte + project(effect).source_byte * (-1)
        return replace(effect, args=tuple((k, {
            'output_column': f['output_column'] + delta,
            'local_row': band.local_row, 'rows': band.rows,
            'columns': band.columns,
        }.get(k, v)) for k, v in effect.args))

    actual, proposal = propose_tile_load_coalescing(original,
        project=project, encode=encode, max_tiles=4, coalesce_tail=True)
    if not proposal['changes']:
        return None
    expanded, projection = expand_panel_tile_loads(actual,
        project=project, encode_band=encode_band)
    decoded = [(effect.instr, concretize(effect, env))
               for effect, env in instances(expanded)]
    # Reopen the original source axes independently of the proposal codec.
    expected = []
    for ky in range(kh):
        for kx in range(kw):
            for chunk in range(chunks):
                depth = min(isa.DIM, ci - chunk * isa.DIM)
                for ni, width in enumerate(widths):
                    expected.append(('load_weight', dict(
                        reduction_row=(ky * kw + kx) * ci + chunk * isa.DIM,
                        output_column=ni * isa.DIM,
                        local_row=rhs_base + (((ky * kw + kx) * chunks + chunk) * active + ni) * isa.DIM,
                        rows=depth, columns=width)))
    if len(expected) != bands or decoded != expected:
        raise ValueError('actual convolution wide bands differ from original source/layout/order')
    return ConvRhsTransfers(actual, dict(applied=True,
        source_kernel_sha256=original.digest(), actual_kernel_sha256=actual.digest(),
        expanded_kernel_sha256=expanded.digest(), checked_primitive_bands=bands,
        reference_files=REFERENCE_FILES, proposal=proposal, projection=projection,
        source_axes=['kernel_y', 'kernel_x', 'channel_chunk', 'output_tile'],
        completion='existing full phase-ready, compute/readout and phase-last-use drains retained',
        proof='all actual final decoded bands equal the original complete primitive preparation'))
