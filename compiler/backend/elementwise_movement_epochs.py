"""Two-phase accumulator DMA movement/readout with bounded private slots."""
from __future__ import annotations

from hashlib import sha256
import json
import math

from merlin.compile.scheduling.compact_movement import MovementLayout, MovementContract, MovementEffect
from merlin.compile.scheduling.movement_epochs import (
    MovementEpochLaw, plan_movement_epochs, emit_movement_epochs,
)
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from . import isa
from .column_batches import _admission


def _plan(shape, input_dtype, output_dtype, scale, activation, *, original_shape=None, subview=None):
    if (not shape or any(type(d) is not int or d <= 0 for d in shape)
            or input_dtype not in ('i8', 'i32') or output_dtype not in ('i8', 'i32')
            or type(activation) is not int or activation not in (0, 1)
            or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0
            or output_dtype == 'i32' and (activation or scale != 1)):
        raise ValueError('movement epochs require the original exact dtype/readout route')
    scale_bits = isa.f32_bits(scale)
    count = math.prod(shape)
    # The incumbent movement treats contiguous tensors as a flattened vector.
    # Preserve full 16x16 tiles when its complete vector admits a 16-column
    # view. Otherwise retain a complete original rectangular view and tails.
    rows, columns = (count // isa.DIM, isa.DIM) if count % isa.DIM == 0 else (math.prod(shape[:-1]), shape[-1])
    scalar = dict(original_shape=list(original_shape or shape), source_elements=math.prod(original_shape or shape),
        contiguous_subview=subview or [0, count],
        input_dtype=input_dtype, accumulator_dtype='i32', output_dtype=output_dtype,
        load_scale_bits=isa.F32_ONE, readout_scale_bits=scale_bits, activation=activation,
        load='signed i8 identity-scale widening' if input_dtype == 'i8' else 'exact full-width i32 copy',
        readout='exact full-width i32 copy' if output_dtype == 'i32' else
            'optional accumulator ReLU before f32 scale, half-even rounding, signed i8 saturation')
    identity = sha256(json.dumps(scalar, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    admission = _admission()
    hardware = admission['base_facts_sha256']
    layout = MovementLayout(rows, columns, isa.DIM, isa.DIM, 0, isa.DIM * 4,
        isa.ACC_ROWS * isa.DIM * 4, input_dtype=input_dtype, output_dtype=output_dtype)
    parameters = (
        (('scale_bits', isa.F32_ONE), ('activation', 0), ('input_bits', int(input_dtype[1:])),
         ('output_bits', 32), ('shrink', int(input_dtype == 'i8')), ('full', 0)),
        (('scale_bits', scale_bits), ('activation', activation), ('input_bits', 32),
         ('output_bits', int(output_dtype[1:])), ('shrink', 0), ('full', int(output_dtype == 'i32'))))
    contract = MovementContract(identity, parameters, hardware, True, True)
    mt, nt = (rows + isa.DIM - 1) // isa.DIM, (columns + isa.DIM - 1) // isa.DIM
    row, column = Var('movement_m') * isa.DIM, Var('movement_n') * isa.DIM
    height = select('movement_m', mt - 1, (rows - 1) % isa.DIM + 1, isa.DIM)
    width = select('movement_n', nt - 1, (columns - 1) % isa.DIM + 1, isa.DIM)
    calls = tuple(Call(f'movement_{kind}_{identity}',
        (('row', row), ('column', column), ('rows', height), ('columns', width),
         ('local_byte', Const(0)), ('row_bytes', Const(isa.DIM * 4)),
         *((key, Const(value)) for key, value in params)))
        for kind, params in zip(('load', 'store'), parameters, strict=True))
    kernel = Kernel('movement_source', (TensorArg('input', (rows, columns), input_dtype, 'read'),
        TensorArg('output', (rows, columns), output_dtype, 'write')),
        (Loop('movement_m', mt, (Loop('movement_n', nt, calls),)),))
    names = {calls[0].instr: ('load', 'input'), calls[1].instr: ('store', 'output')}

    def project(call):
        kind, tensor = names[call.instr]
        args = dict(call.args)
        def integer(name):
            value = args[name]
            if type(value) is not Const:
                raise ValueError('movement scalar or storage binding became dynamic')
            return value.value
        keys = ('scale_bits', 'activation', 'input_bits', 'output_bits', 'shrink', 'full')
        if set(args) != {'row', 'column', 'rows', 'columns', 'local_byte', 'row_bytes', *keys}:
            raise ValueError('movement source effect has unknown operands')
        return MovementEffect(kind, tensor, args['row'], args['column'], args['rows'],
            args['columns'], args['local_byte'], integer('row_bytes'), identity,
            tuple((key, integer(key)) for key in keys), 'synchronous')

    law = MovementEpochLaw(hardware, identity,
        admission['ordering_sha256'] + ':fixed-config-two-phase-disjoint-acc-movement',
        disjoint_loads_pending=True, disjoint_stores_pending=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    plan = plan_movement_epochs(kernel, layout, contract, project, scalar_proof=scalar,
        target_law=law, max_live_tiles=isa.ACC_ROWS // isa.DIM, flatten_rows=True)
    plan.revalidate(kernel, layout, contract, project, scalar_proof=scalar)
    return plan, kernel, scalar, project, admission


def emit_batched_movement(ctx, src, dst, shape, scale=1.0, activation=0):
    """Return None before emission on refusal; otherwise emit load/store phases."""
    try:
        if src == dst or src not in ctx.buffers:
            return None
        inp, out = ctx.spec(src), ctx.spec(dst)
        if tuple(inp['shape']) != tuple(shape) or tuple(out['shape']) != tuple(shape):
            return None
        aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
        related = set(aliases) | {a.get('source') for a in aliases.values() if isinstance(a, dict)}
        if {src, dst} & related:
            return None
        if dst in ctx.buffers and ctx.buffers[src] is ctx.buffers[dst]:
            return None
        count = math.prod(shape)
        if count <= isa.DIM * isa.DIM:
            return None
        full_rows, tail = divmod(count, isa.DIM)
        pieces = ([(0, (full_rows, isa.DIM))] if full_rows else [])
        if tail:
            pieces.append((full_rows * isa.DIM, (1, tail)))
        cursor, prepared = 0, []
        for offset, child_shape in pieces:
            extent = math.prod(child_shape)
            if offset != cursor:
                raise ValueError('movement contiguous subviews have a gap')
            prepared.append((offset, _plan(child_shape, inp['dtype'], out['dtype'], scale,
                activation, original_shape=tuple(shape), subview=[offset, extent])))
            cursor += extent
        if cursor != count:
            raise ValueError('movement source partition is incomplete')
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    builder = FnBuilderControlAdapter(ctx.fb)
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    ctx.fence()
    reports = []
    for base_offset, (plan, kernel, scalar, project, admission) in prepared:
        columns = plan.layout.columns
        ctx.emit(*isa.config_ld(columns * (1 if inp['dtype'] == 'i8' else 4), channel=0,
            scale_bits=isa.F32_ONE, shrink=inp['dtype'] == 'i8'))
        ctx.emit(*isa.config_st(columns * (1 if out['dtype'] == 'i8' else 4),
            scale_bits=isa.f32_bits(scale), activation=activation))

        def lower(kind, fields):
            if kind == 'complete_phase':
                ctx.fence()
                return
            index = ('load', 'store').index(kind)
            if fields['parameters'] != plan.contract.parameters[index]:
                raise ValueError('actual movement phase differs from checked dtype/configuration')
            offset = builder.op('add', base_offset, builder.op('add',
                builder.op('mul', fields['row'], columns), fields['column']))
            local_row = builder.op('udiv', fields['local_byte'], isa.DIM * 4)
            local = isa.acc_addr(local_row, full=kind == 'store' and out['dtype'] == 'i32', fb=ctx.fb)
            if kind == 'load':
                command = isa.mvin(ctx.ptr(src, offset), local, fields['rows'], fields['columns'], fb=ctx.fb)
            else:
                command = isa.mvout(ctx.ptr(dst, offset), local, fields['rows'], fields['columns'], fb=ctx.fb)
            ctx.emit(*command)

        report = emit_movement_epochs(plan, builder, lower)
        plan.revalidate(kernel, plan.layout, plan.contract, project, scalar_proof=scalar)
        reports.append(dict(report, scalar_proof=scalar, source_kernel_text=kernel.text(),
            source_subview=[base_offset, plan.layout.rows * columns]))
    return dict(route='batched_movement_accumulator_epochs', children=reports,
        output_elements=count, source_tiles=sum(r['source_tiles'] for r in reports),
        groups=sum(r['groups'] for r in reports),
        complete_phase_drains=sum(r['complete_phase_drains'] for r in reports),
        source_partition=dict(original_shape=list(shape), original_elements=count,
            subviews=[r['source_subview'] for r in reports],
            proof='ordered contiguous disjoint subviews cover [0, original_elements) exactly'),
        target_admission=admission,
        configuration='original dtype and scale bindings constant until both real phase drains')
