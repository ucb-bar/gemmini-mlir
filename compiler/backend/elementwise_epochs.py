"""Fact-bound residual DMA phases over disjoint accumulator tiles.

The original exhaustive i8 numerical screen owns arithmetic admission. Shared
loaded-sum verification proves complete source cover and binds exact scale bits;
epoch planning changes only private storage and full completion boundaries.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

from merlin.compile.scheduling.compact_loaded_sum import (
    LoadedSumLayout, LoadedSumContract, LoadedSumEffect,
)
from merlin.compile.scheduling.loaded_sum_epochs import (
    LoadedSumEpochLaw, plan_loaded_sum_epochs, emit_loaded_sum_epochs,
)
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from . import isa
from .column_batches import _admission


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _plan(shape, proof, activation, *, original_shape=None, subview=None):
    if (not shape or any(type(d) is not int or d <= 0 for d in shape)
            or type(activation) is not int or activation not in (0, 1)
            or type(proof) is not dict or proof.get('pairs_checked') != 65536
            or type(proof.get('max_error_lsb')) is not int or proof['max_error_lsb'] < 0):
        raise ValueError('residual epochs require the original exhaustive scalar screen')
    scales = tuple(proof[key] for key in ('lhs_load_f32', 'rhs_load_f32', 'readout_scale_f32'))
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in scales):
        raise ValueError('residual epochs require finite original scale bindings')
    scale_bits = tuple(isa.f32_bits(v) for v in scales)
    original = proof.get('original_contract')
    if original is not None:
        from .elementwise import residual_numeric_screen
        if (type(original) is not dict or set(original) !=
                {'lhs_scale', 'rhs_scale', 'bound_lsb', 'relu'}
                or type(original['relu']) is not bool
                or original['relu'] != bool(activation)):
            raise ValueError('residual original scalar parameters differ from activation')
        checked = residual_numeric_screen(original['lhs_scale'], original['rhs_scale'],
            original['bound_lsb'], original['relu'])
        if any(proof.get(key) != value for key, value in checked.items()):
            raise ValueError('residual supplied screen differs from original source parameters')
    scalar = dict(numeric_screen=dict(proof), original_shape=list(original_shape or shape),
                  contiguous_subview=subview or [0, math.prod(shape)],
                  activation=activation, scale_bits=list(scale_bits))
    identity = _digest(scalar)
    admission = _admission()
    hardware = admission['base_facts_sha256']
    rows, columns = math.prod(shape[:-1]), shape[-1]
    layout = LoadedSumLayout(rows, columns, isa.DIM, isa.DIM, 0,
        isa.DIM * 4, isa.ACC_ROWS * isa.DIM * 4)
    parameters = tuple((('scale_bits', bits), ('activation', activation if i == 2 else 0),
                        ('channel', 1 if i == 1 else 0), ('accumulate', int(i == 1)))
                       for i, bits in enumerate(scale_bits))
    contract = LoadedSumContract(identity, parameters, hardware, True, True)
    mt = (rows + isa.DIM - 1) // isa.DIM
    nt = (columns + isa.DIM - 1) // isa.DIM
    row, column = Var('residual_m') * isa.DIM, Var('residual_n') * isa.DIM
    height = select('residual_m', mt - 1, (rows - 1) % isa.DIM + 1, isa.DIM)
    width = select('residual_n', nt - 1, (columns - 1) % isa.DIM + 1, isa.DIM)
    kinds = ('initialize', 'accumulate', 'store')
    names = {f'residual_{kind}_{identity}': (kind, tensor)
             for kind, tensor in zip(kinds, ('lhs', 'rhs', 'output'), strict=True)}
    calls = tuple(Call(f'residual_{kind}_{identity}',
        (('row', row), ('column', column), ('rows', height), ('columns', width),
         ('local_byte', Const(0)), ('row_bytes', Const(isa.DIM * 4)),
         *((key, Const(value)) for key, value in params)))
        for kind, params in zip(kinds, parameters, strict=True))
    kernel = Kernel('residual_source', tuple(TensorArg(name, (rows, columns), 'i8', access)
        for name, access in (('lhs', 'read'), ('rhs', 'read'), ('output', 'write'))),
        (Loop('residual_m', mt, (Loop('residual_n', nt, calls),)),))

    def project(call):
        kind, tensor = names[call.instr]
        args = dict(call.args)
        def integer(name):
            value = args[name]
            if type(value) is not Const:
                raise ValueError('residual scalar or row-byte binding became dynamic')
            return value.value
        params = tuple((key, integer(key)) for key in ('scale_bits', 'activation', 'channel', 'accumulate'))
        if set(args) != {'row', 'column', 'rows', 'columns', 'local_byte', 'row_bytes',
                         'scale_bits', 'activation', 'channel', 'accumulate'}:
            raise ValueError('residual source effect has unknown operands')
        return LoadedSumEffect(kind, tensor, args['row'], args['column'], args['rows'],
            args['columns'], args['local_byte'], integer('row_bytes'), identity,
            params, 'synchronous')

    law = LoadedSumEpochLaw(hardware, identity,
        admission['ordering_sha256'] + ':fixed-load-store-config-disjoint-private-acc-phases',
        disjoint_loads_pending=True, disjoint_stores_pending=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    plan = plan_loaded_sum_epochs(kernel, layout, contract, project, scalar_proof=scalar,
        target_law=law, max_live_tiles=isa.ACC_ROWS // isa.DIM, flatten_rows=True)
    plan.revalidate(kernel, layout, contract, project, scalar_proof=scalar)
    return plan, kernel, scalar, project, admission


def emit_batched_residual(ctx, lhs, rhs, dst, shape, proof, activation):
    """Emit three full-drained disjoint-tile phases, or refuse before emission.

    The caller selects this route after its existing numeric screen. The
    original logical row pitch and exact original f32 load/readout bits stay
    fixed throughout every phase. All slots drain before any private reuse.
    """
    names = (lhs, rhs, dst)
    try:
        if len(set(names)) != 3:
            return None
        for name in names:
            spec = ctx.spec(name)
            if spec['dtype'] != 'i8' or tuple(spec['shape']) != tuple(shape):
                return None
        aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
        alias_names = set(aliases)
        alias_sources = {value.get('source') for value in aliases.values() if isinstance(value, dict)}
        if set(names) & (alias_names | alias_sources):
            return None
        # Sources must already contain values. A fresh destination can receive
        # a new private allocation after all read-only admission has passed.
        buffers = ctx.buffers
        if lhs not in buffers or rhs not in buffers:
            return None
        pointers = [buffers[name] for name in names if name in buffers]
        if any(pointers[i] is pointers[j] for i in range(len(pointers)) for j in range(i)):
            return None
        count = math.prod(shape)
        if count <= isa.DIM * isa.DIM:
            return None
        full_rows, tail = divmod(count, isa.DIM)
        pieces = ([(0, (full_rows, isa.DIM))] if full_rows else [])
        if tail:
            pieces.append((full_rows * isa.DIM, (1, tail)))
        # Complete, ordered, disjoint contiguous subviews of the original flat
        # tensor. No padding or hidden element appears in either source proof.
        cursor = 0
        prepared = []
        for offset, child_shape in pieces:
            extent = math.prod(child_shape)
            if offset != cursor:
                raise ValueError('residual contiguous source partition has a gap')
            prepared.append((offset, _plan(child_shape, proof, activation,
                original_shape=tuple(shape), subview=[offset, extent])))
            cursor += extent
        if cursor != count:
            raise ValueError('residual source partition is incomplete')
    except (KeyError, ValueError, TypeError, OverflowError):
        return None

    builder = FnBuilderControlAdapter(ctx.fb)
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    ctx.fence()
    reports = []
    for base_offset, (plan, kernel, scalar, project, admission) in prepared:
        columns = plan.layout.columns
        for channel, key in ((0, 'lhs_load_f32'), (1, 'rhs_load_f32')):
            ctx.emit(*isa.config_ld(columns, channel=channel,
                scale_bits=isa.f32_bits(proof[key]), shrink=True))
        ctx.emit(*isa.config_st(columns, scale_bits=isa.f32_bits(proof['readout_scale_f32']),
                               activation=activation))

        def lower(kind, fields):
            if kind == 'complete_phase':
                ctx.fence()
                return
            index = ('initialize', 'accumulate', 'store').index(kind)
            if fields['parameters'] != plan.contract.parameters[index]:
                raise ValueError('actual residual phase parameters differ from checked configuration')
            offset = builder.op('add', base_offset, builder.op('add',
                builder.op('mul', fields['row'], columns), fields['column']))
            local_row = builder.op('udiv', fields['local_byte'], isa.DIM * 4)
            local = isa.acc_addr(local_row, accumulate=kind == 'accumulate', fb=ctx.fb)
            if kind == 'store':
                command = isa.mvout(ctx.ptr(dst, offset), local,
                    rows=fields['rows'], cols=fields['columns'], fb=ctx.fb)
            else:
                command = isa.mvin(ctx.ptr(lhs if kind == 'initialize' else rhs, offset), local,
                    rows=fields['rows'], cols=fields['columns'],
                    channel=0 if kind == 'initialize' else 1, fb=ctx.fb)
            ctx.emit(*command)

        report = emit_loaded_sum_epochs(plan, builder, lower)
        plan.revalidate(kernel, plan.layout, plan.contract, project, scalar_proof=scalar)
        reports.append(dict(report, scalar_proof=scalar, source_kernel_text=kernel.text(),
            source_subview=[base_offset, plan.layout.rows * columns]))
    return dict(route='batched_residual_accumulator_epochs', numeric_screen=proof,
        children=reports, output_elements=count,
        source_tiles=sum(r['source_tiles'] for r in reports),
        groups=sum(r['groups'] for r in reports),
        complete_phase_drains=sum(r['complete_phase_drains'] for r in reports),
        source_partition=dict(original_shape=list(shape), original_elements=count,
            subviews=[r['source_subview'] for r in reports],
            proof='ordered contiguous disjoint subviews cover [0, original_elements) exactly'),
        target_admission=admission,
        configuration='constant original scale bits through every complete phase drain',
        external_storage='three distinct original pointers; relevant tensor aliases refused')
