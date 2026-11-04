"""Bias broadcast epochs and the shared standalone accumulator readout route."""
from hashlib import sha256
import json

from merlin.compile.scheduling.bias_sum_view import BiasSumEffect, normalize_bias_sum
from merlin.compile.scheduling.compact_loaded_sum import LoadedSumLayout, LoadedSumContract, LoadedSumEffect
from merlin.compile.scheduling.loaded_sum_epochs import LoadedSumEpochLaw, plan_loaded_sum_epochs, emit_loaded_sum_epochs
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from . import isa
from .column_batches import _admission


def emit_batched_readout(ctx, src, dst, shape, scale=1.0, activation=0):
    """Use the existing exact two-phase movement/readout consumer, or refuse."""
    try:
        if ctx.spec(src)['dtype'] != 'i32':
            return None
    except (KeyError, ValueError, TypeError):
        return None
    from .elementwise_movement_epochs import emit_batched_movement
    return emit_batched_movement(ctx, src, dst, shape, scale, activation)


def _bias_plan(shape):
    if (len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)):
        raise ValueError('bias epochs require a positive rank-two rectangle')
    rows, columns = shape
    scalar = dict(source='output[r,c] = signed_i32_wrap(lhs[r,c] + bias[c])',
        source_shape=list(shape), bias_shape=[columns], input_dtype='i32', output_dtype='i32',
        accumulator_dtype='i32', load_scale_bits=isa.F32_ONE,
        readout_scale_bits=isa.F32_ONE, activation=0, readout_full=True,
        bias_source_pitch_bytes=0, matrix_source_pitch_bytes=columns * 4,
        arithmetic='full-width DMA identity loads, modulo 2^32 accumulation, raw full-width store')
    identity = sha256(json.dumps(scalar, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    admission = _admission()
    layout = LoadedSumLayout(rows, columns, isa.DIM, isa.DIM, 0,
        isa.DIM * 4, isa.ACC_ROWS * isa.DIM * 4, input_dtype='i32', output_dtype='i32')
    parameters = tuple((('scale_bits', isa.F32_ONE), ('activation', 0),
        ('channel', int(i == 1)), ('accumulate', int(i == 1)), ('shrink', 0),
        ('full', int(i == 2)), ('source_pitch_bytes', 0 if i == 1 else columns * 4))
        for i in range(3))
    contract = LoadedSumContract(identity, parameters, admission['base_facts_sha256'], True, True)
    mt, nt = (rows + isa.DIM - 1) // isa.DIM, (columns + isa.DIM - 1) // isa.DIM
    row, column = Var('bias_m') * isa.DIM, Var('bias_n') * isa.DIM
    height = select('bias_m', mt - 1, (rows - 1) % isa.DIM + 1, isa.DIM)
    width = select('bias_n', nt - 1, (columns - 1) % isa.DIM + 1, isa.DIM)
    kinds = ('initialize', 'accumulate', 'store')
    names = {f'bias_{kind}_{identity}': (kind, tensor) for kind, tensor in
        zip(kinds, ('lhs', 'rhs', 'output'), strict=True)}
    calls = tuple(Call(f'bias_{kind}_{identity}', (
        ('row', row), ('column', column), ('rows', height), ('columns', width),
        ('source_row', Const(0) if i == 1 else row), ('source_column', column),
        ('local_byte', Const(0)), ('row_bytes', Const(isa.DIM * 4)),
        *((key, Const(value)) for key, value in parameters[i])))
        for i, kind in enumerate(kinds))
    source = Kernel('bias_sum_source', (TensorArg('lhs', shape, 'i32', 'read'),
        TensorArg('bias', (columns,), 'i32', 'read'), TensorArg('output', shape, 'i32', 'write')),
        (Loop('bias_m', mt, (Loop('bias_n', nt, calls),)),))

    def project(call):
        kind, tensor = names[call.instr]
        args = dict(call.args)
        def integer(name):
            value = args[name]
            if type(value) is not Const:
                raise ValueError('bias scalar or storage operand became dynamic')
            return value.value
        keys = tuple(key for key, _ in parameters[0])
        if set(args) != {'row', 'column', 'rows', 'columns', 'source_row', 'source_column',
            'local_byte', 'row_bytes', *keys}:
            raise ValueError('bias actual source has unknown operands')
        logical = LoadedSumEffect(kind, tensor, args['row'], args['column'], args['rows'],
            args['columns'], args['local_byte'], integer('row_bytes'), identity,
            tuple((key, integer(key)) for key in keys), 'synchronous')
        return BiasSumEffect(logical, args['source_row'], args['source_column'], integer('source_pitch_bytes'))

    normalized, logical_project, view = normalize_bias_sum(source, layout, contract, project,
        scalar_proof=scalar)
    law = LoadedSumEpochLaw(admission['base_facts_sha256'], identity,
        admission['ordering_sha256'] + ':fixed-config-disjoint-column-bias-epochs',
        disjoint_loads_pending=True, disjoint_stores_pending=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    plan = plan_loaded_sum_epochs(normalized, layout, contract, logical_project,
        scalar_proof=scalar, target_law=law, max_live_tiles=isa.ACC_ROWS // isa.DIM, flatten_rows=True)
    plan.revalidate(normalized, layout, contract, logical_project, scalar_proof=scalar)
    return plan, source, scalar, project, view, admission


def emit_batched_bias(ctx, src, bias, dst, shape, scale=1.0, activation=0, output_dtype='i32'):
    """Batch the exact incumbent i32 column bias addition; no emit on refusal."""
    names = (src, bias, dst)
    try:
        shape = tuple(shape)
        if len(set(names)) != 3 or scale != 1 or activation != 0 or output_dtype != 'i32':
            return None
        if (ctx.spec(src)['dtype'] != 'i32' or ctx.spec(bias)['dtype'] != 'i32'
            or ctx.spec(dst)['dtype'] != 'i32' or tuple(ctx.spec(src)['shape']) != shape
            or tuple(ctx.spec(dst)['shape']) != shape or tuple(ctx.spec(bias)['shape']) != (shape[-1],)):
            return None
        if src not in ctx.buffers or bias not in ctx.buffers:
            return None
        aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
        related = set(aliases) | {a.get('source') for a in aliases.values() if isinstance(a, dict)}
        if set(names) & related:
            return None
        pointers = [ctx.buffers[name] for name in names if name in ctx.buffers]
        if any(pointers[i] is pointers[j] for i in range(len(pointers)) for j in range(i)):
            return None
        plan, source, scalar, project, view, admission = _bias_plan(shape)
        if plan.row_tiles * plan.column_tiles <= 1:
            return None
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return None

    # Allocation follows every read-only source/numerical/storage admission.
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    builder = FnBuilderControlAdapter(ctx.fb)
    columns = plan.layout.columns
    ctx.fence()
    ctx.emit(*isa.config_ld(columns * 4, channel=0, shrink=False))
    ctx.emit(*isa.config_ld(0, channel=1, shrink=False))
    ctx.emit(*isa.config_st(columns * 4))

    def lower(kind, fields):
        if kind == 'complete_phase':
            ctx.fence()
            return
        index = ('initialize', 'accumulate', 'store').index(kind)
        if fields['parameters'] != plan.contract.parameters[index]:
            raise ValueError('actual bias phase configuration differs from verified source')
        local_row = builder.op('udiv', fields['local_byte'], isa.DIM * 4)
        local = isa.acc_addr(local_row, accumulate=kind == 'accumulate', full=kind == 'store', fb=ctx.fb)
        if kind == 'accumulate':
            command = isa.mvin(ctx.ptr(bias, fields['column']), local,
                fields['rows'], fields['columns'], channel=1, fb=ctx.fb)
        else:
            offset = builder.op('add', builder.op('mul', fields['row'], columns), fields['column'])
            pointer = ctx.ptr(src if kind == 'initialize' else dst, offset)
            command = (isa.mvin if kind == 'initialize' else isa.mvout)(pointer, local,
                fields['rows'], fields['columns'], fb=ctx.fb)
        ctx.emit(*command)

    report = emit_loaded_sum_epochs(plan, builder, lower)
    normalized, logical_project, current_view = normalize_bias_sum(source, plan.layout,
        plan.contract, project, scalar_proof=scalar)
    plan.revalidate(normalized, plan.layout, plan.contract, logical_project, scalar_proof=scalar)
    if current_view != view:
        raise ValueError('bias original source view changed during emission')
    return dict(report, route='batched_column_bias_epochs', scalar_proof=scalar,
        source_kernel_text=source.text(), source_view=view, target_admission=admission,
        configuration='original full-i32 loads, bias pitch zero, raw full-i32 readout; three real phase drains')


def _validate_column_attributes(attrs):
    """Refuse source arithmetic that the standalone column consumer cannot apply.

    The command schema leaves attributes open for provenance and scheduling.
    That is not permission to discard a second numerical operation. In
    particular, the shared declared_acc_scale_by_column contract makes scalar
    and column scales mutually exclusive. Other arithmetic families need their
    own lowering; even an identity-looking declaration is outside this route.
    """
    from merlin.runtime.commandbuffer import declared_acc_scale_by_column

    unsupported = {
        'scale', 'scales', 'activation', 'clamp', 'clamp_min', 'clamp_max',
        'rounding', 'rounding_mode', 'saturation', 'saturate', 'combine',
        'input_scale', 'input_scales', 'output_scale', 'output_scales',
        'lhs_scale', 'rhs_scale', 'bound_lsb', 'acc_scale_bits',
        'shift', 'multiplier', 'offset', 'offsets',
    }
    prefixes = ('bias', 'accumulator_init', 'pool', 'maxpool', 'requant',
                'mvin_', 'load_scale', 'input_scale', 'output_scale')
    for key in attrs:
        # Namespaced provenance is descriptive, including numeric source IDs.
        if isinstance(key, str) and key.startswith(('prov.', 'merlin.')):
            continue
        if (not isinstance(key, str) or key in unsupported
            or key.startswith(prefixes) or 'zero_point' in key):
            raise ValueError(f'column readout cannot apply source attribute {key!r}')
    scales = attrs.get('acc_scale_by_column')
    # Keep the existing tuple convenience while using the public numerical
    # declaration checker, including scalar/column exclusivity and f32 limits.
    declared_acc_scale_by_column(dict(attrs, acc_scale_by_column=
        list(scales) if isinstance(scales, tuple) else scales),
        len(scales) if isinstance(scales, (tuple, list)) else 0,
        op='standalone column readout')


class _ContiguousColumnRows:
    """A logical matrix view of checked contiguous storage, without allocation.

    For original shape [..., N], row r and column c address exactly r*N+c.
    Only the two tensor specifications are viewed as matrices; every pointer,
    instruction and completion operation belongs to the original context.
    """
    def __init__(self, ctx, names, matrix_shape):
        self.ctx, self.names, self.matrix_shape = ctx, names, matrix_shape

    def spec(self, name):
        spec = self.ctx.spec(name)
        return dict(spec, shape=list(self.matrix_shape)) if name in self.names else spec

    def __getattr__(self, name):
        return getattr(self.ctx, name)


def lower_column_readout(ctx, src, dst, shape, attributes):
    """Realize standalone i32->i8 column scales, including a synchronous fallback.

    All source, scale, alias and route checks precede any context mutation.
    Arithmetic is exactly the shared terminal store_tile implementation; no
    scales are folded, regrouped numerically, or loaded as tensor values.
    """
    import math
    import struct
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    from .column_batches import emit_batched_column_readout, _palette
    from .dense_readout import store_tile

    shape = tuple(shape)
    attrs = dict(attributes)
    _validate_column_attributes(attrs)
    ep = attrs.get('epilogue', [])
    if (not shape or any(type(d) is not int or d <= 0 for d in shape)
        or not isinstance(ep, (tuple, list)) or any(stage not in ('acc_scale', 'relu') for stage in ep)
        or len(ep) != len(set(ep))):
        raise ValueError('column readout requires positive static extents and only scale/ReLU epilogue')
    m, n = math.prod(shape[:-1]), shape[-1]
    if (src == dst or src not in ctx.buffers or ctx.spec(src)['dtype'] != 'i32'
        or ctx.spec(dst)['dtype'] != 'i8' or tuple(ctx.spec(src)['shape']) != shape
        or tuple(ctx.spec(dst)['shape']) != shape or attrs.get('output_dtype', 'i8') != 'i8'):
        raise ValueError('column readout requires disjoint complete i32 source and i8 destination')
    if attrs.get('semantic', 'mvin_mvout') not in ('mvin_mvout', 'copy'):
        raise ValueError('column readout has no selected source movement semantics')
    if len(shape) != 2:
        if any(spec.get('layout') not in (None, 'row_major', 'contiguous')
               for spec in (ctx.spec(src), ctx.spec(dst), attrs)):
            raise ValueError('ranked column readout requires contiguous row-major storage')
        if any(getattr(ctx, 'row_stride', lambda name, default: default)(name, n) != n
               for name in (src, dst)):
            raise ValueError('ranked column readout cannot infer a noncontiguous row pitch')
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    related = set(aliases) | {a.get('source') for a in aliases.values() if isinstance(a, dict)}
    if dst in related or dst in ctx.buffers and ctx.buffers[src] is ctx.buffers[dst]:
        raise ValueError('column readout destination aliases are unsupported')
    source_pitch = n
    source_view = None
    if src in related:
        if len(shape) != 2:
            raise ValueError('ranked column readout aliases require an independent contiguous view proof')
        alias = aliases.get(src)
        # EmissionContext has independently checked this source-bound view's
        # offset, typed physical span and read-only row pitch while registering
        # its pointer. A destination/root alias or an unproved layout refuses.
        if (not isinstance(alias, dict) or alias.get('access') != 'read'
            or alias.get('source') in aliases or alias.get('source') == dst
            or alias.get('dtype') != 'i32' or tuple(alias.get('shape', ())) != shape):
            raise ValueError('column readout needs an independently registered read-only source view')
        source_pitch = alias.get('row_stride', n)
        extent = (m - 1) * source_pitch + n if type(source_pitch) is int else None
        if (type(source_pitch) is not int or source_pitch < n
            or ctx.row_stride(src, n) != source_pitch
            or source_pitch != n and alias.get('element_span') != extent):
            raise ValueError('column readout source pitch differs from its checked physical span')
        source_view = dict(alias)
    scales = attrs.get('acc_scale_by_column')
    if (not isinstance(scales, (list, tuple)) or len(scales) != n
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in scales)):
        raise ValueError('column readout needs one finite nonnegative source scale per column')
    if 'relu' in attrs and (type(attrs['relu']) not in (bool, int) or attrs['relu'] not in (0, 1)):
        raise ValueError('column readout activation must be exactly zero or one')
    # Preflight f32 conversion/complete command encodings without emitting. A
    # finite Python float whose selected f32 is infinite cannot enter this path.
    bits = []
    try:
        for value in scales:
            encoded = isa.f32_bits(value)
            decoded = struct.unpack('<f', struct.pack('<I', encoded))[0]
            if not math.isfinite(decoded):
                raise ValueError('column source scale is not finite in f32')
            isa.config_st(n, scale_bits=encoded, activation=int(bool(attrs.get('relu', False) or 'relu' in ep)))
            bits.append(encoded)
    except OverflowError as exc:
        raise ValueError('column source scale cannot be represented in f32') from exc
    selection = attrs.get('column_readout_batching', knob('COLUMN_READOUT_BATCHING', default='off',
        values=('off', 'on', 'palette'), root=Path(__file__).resolve().parents[1]))
    if selection not in ('off', 'on', 'palette'):
        raise ValueError('column readout batching must be off, on or palette')
    attrs['acc_scale_by_column'] = list(scales)
    refusal = None
    if selection != 'off' and source_pitch != n:
        # The existing batch consumer owns a contiguous source contract. Keep
        # its exact synchronous one-column consumer for a pitched view instead.
        refusal, selection = 'batched column consumer requires contiguous source rows', 'off'
    if selection != 'off':
        try:
            _admission()
            if selection == 'palette':
                _palette(n, attrs)
        except ValueError as exc:
            # Static refusal retains the same selected per-column terminal
            # lowering. No device command has been issued at this point.
            refusal, selection = str(exc), 'off'
    attrs['column_readout_batching'] = selection
    if dst not in ctx.buffers:
        ctx.buffer(dst)
    consumer = ctx if len(shape) == 2 else _ContiguousColumnRows(ctx, (src, dst), (m, n))
    view_fields = {} if len(shape) == 2 else {'contiguous_last_axis_view': {
        'source_shape': list(shape), 'matrix_shape': [m, n],
        'element_address': 'row * last_axis_extent + column',
        'storage': 'original contiguous source and destination; no materialization'}}
    batching = emit_batched_column_readout(consumer, src, dst, m, n, attrs)
    if batching is not None:
        return dict(route='standalone_column_readout', batching=batching,
            shape=list(shape), source_scale_bits=bits, output_dtype='i8',
            source_row_stride_elements=source_pitch, source_view=source_view, **view_fields)

    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(source_pitch * 4, channel=2, shrink=False))
    def column(col):
        def tile(row, rows):
            offset = fb.add_i(fb.mul_i(row, fb.const(source_pitch)), col)
            ctx.emit(*isa.mvin(ctx.ptr(src, offset), isa.acc_addr(0), rows, 1, channel=2, fb=fb))
            ctx.fence()
            store_tile(ctx, dst, 0, rows, 1, n, row, col, attrs)
        complete = m // isa.DIM * isa.DIM
        if complete:
            fb.for_loop(0, complete, isa.DIM, lambda row: tile(row, isa.DIM), retain_loop=True)
        if complete < m:
            tile(fb.const(complete), m - complete)
    fb.for_loop(0, n, 1, column, retain_loop=True)
    ctx.fence()
    return dict(route='standalone_column_readout', batching=None, shape=list(shape),
        source_scale_bits=bits, output_dtype='i8', batching_refusal=refusal,
        source_row_stride_elements=source_pitch, source_view=source_view,
        completion='original per-column full load/store drains', **view_fields)
