"""Standalone device movement and accumulator operand sums.

Tensor arithmetic occurs only in Gemmini loads and accumulator readout. Host
instructions form addresses and tile-loop controls. Residual admission exhaustively
checks the declared i8 domain against the source's single-rounding f32 arithmetic.
"""
from functools import lru_cache
from math import isfinite, prod
import struct
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import StringAttr

from isa_encoder import (ACC_ADDR, ACC_ACCUMULATE, ACC_FULL, config_load,
                         config_store, pack_matrix)


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _sat8(x):
    return max(-128, min(127, int(x)))


@lru_cache(maxsize=256)
def residual_load_scales(lhs_scale, rhs_scale, bound_lsb, relu=False):
    """Select an exhaustively valid load/readout decomposition, or fail closed.

    The load scaler rounds and saturates each operand into i8. Normalization
    permits scales above unity without silently clipping the operands too early.
    Checking all 65536 pairs is shape independent and cached across commands.
    """
    if isinstance(bound_lsb, bool) or not isinstance(bound_lsb, int) or bound_lsb < 0:
        raise ValueError('RESIDUAL_ADD requires a declared nonnegative bound_lsb')
    lhs_scale, rhs_scale = _f32(lhs_scale), _f32(rhs_scale)
    if not all(isfinite(s) for s in (lhs_scale, rhs_scale)):
        raise ValueError('RESIDUAL_ADD nonfinite scales have no admitted device decomposition')
    # The direct form is the public operand-sum witness; the normalized form is
    # useful when a larger scale would saturate an individual scaled operand.
    candidates = [1.0, _f32(max(1., abs(lhs_scale), abs(rhs_scale)))]
    for readout in dict.fromkeys(candidates):
        ls, rs = _f32(lhs_scale / readout), _f32(rhs_scale / readout)
        a_source = [_f32(a * lhs_scale) for a in range(-128, 128)]
        b_source = [_f32(b * rhs_scale) for b in range(-128, 128)]
        a_load = [_sat8(round(_f32(a * ls))) for a in range(-128, 128)]
        b_load = [_sat8(round(_f32(b * rs))) for b in range(-128, 128)]
        valid = True
        for ai in range(256):
            for bi in range(256):
                ref = _sat8(round(_f32(a_source[ai] + b_source[bi])))
                acc = a_load[ai] + b_load[bi]
                if relu:
                    acc = max(0, acc)
                    ref = max(0, ref)
                got = _sat8(round(_f32(acc * readout)))
                if abs(got - ref) > bound_lsb:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return ls, rs, readout
    raise ValueError('RESIDUAL_ADD compiler proof gap: accumulator operand-sum '
                     'decompositions exceed the declared bound_lsb')


def _matrix(tensor):
    shape = tensor['shape']
    if not shape or any(type(s) is not int or s <= 0 for s in shape):
        raise ValueError('device elementwise lowering requires positive static extents')
    return prod(shape[:-1]), shape[-1]


def _tiles(e, m, n, body):
    """Four constant-size tile domains; emitted IR size is independent of extents."""
    def axis(size, callback):
        full = size // 16 * 16
        if full:
            e.for_loop(0, full, 16, lambda i: callback(i, 16))
        if size != full:
            callback(e.b.const(full), size - full)
    axis(m, lambda r, nr: axis(n, lambda c, nc: body(r, c, nr, nc)))



def _operand_sum_panel(e, m, n, a, b, out, pointers, tensors, opcode, dtype, full):
    """Complete 64 independent accumulator tiles per initialization fence.

    Selected memory_address_spaces facts describe two banks of 512 rows, each
    with 16 i32 lanes: 1024 logical rows, hence 64 full 16-row tiles. All A
    tiles initialize disjoint rows before the first fence. Accumulating B loads
    then read those completed rows; every output store finishes before a panel
    can reuse them or a consumer can load its DRAM result. Inputs may coincide:
    neither source is mutated, and corresponding loads remain separate.
    """
    nt = (n + 15) // 16
    tile_count = ((m + 15) // 16) * nt
    panel_tiles = 64
    build = e.b
    def packed(addr, rows, cols):
        return build.or_i(e.value(addr), build.or_i(build.shl_i(e.value(cols), 32),
                                                   build.shl_i(e.value(rows), 48)))
    def panel(first, count):
        def tile(i):
            ordinal = build.add_i(e.value(first), i)
            r = build.mul_i(build.udiv_i(ordinal, e.value(nt)), e.value(16))
            c = build.mul_i(build.urem_i(ordinal, e.value(nt)), e.value(16))
            nr = build.smin(build.sub_i(e.value(m), r), e.value(16))
            nc = build.smin(build.sub_i(e.value(n), c), e.value(16))
            index = build.add_i(build.mul_i(r, e.value(n)), c)
            local = build.add_i(e.value(ACC_ADDR), build.mul_i(i, e.value(16)))
            return c, nr, nc, index, local
        def initialize(i):
            _, nr, nc, index, local = tile(i)
            ptr = e.ptr(pointers[a], index, tensors[a]['dtype'])
            e.insn('mvin', e.address(ptr), packed(local, nr, nc))
        e.for_loop(0, count, 1, initialize)
        e.fence()
        def add_and_store(i):
            c, nr, nc, index, local = tile(i)
            bi = c if opcode == 'BIAS_ADD' else index
            ptr = e.ptr(pointers[b], bi, tensors[b]['dtype'])
            e.insn('mvin2', e.address(ptr), packed(build.or_i(local, e.value(ACC_ACCUMULATE)), nr, nc))
            dst = e.ptr(pointers[out], index, dtype)
            e.insn('mvout', e.address(dst), packed(build.or_i(local, e.value(full)), nr, nc))
        e.for_loop(0, count, 1, add_and_store)
        e.fence()
    full_panels = tile_count // panel_tiles * panel_tiles
    if full_panels:
        e.for_loop(0, full_panels, panel_tiles, lambda first: panel(first, panel_tiles))
    if full_panels != tile_count:
        panel(e.value(full_panels), tile_count - full_panels)


def _select_elementwise_order(m, n, two_sources, requested, policy, bias_output_alias=False):
    """Choose only an existing schedule, from its synchronization mechanism.

    A tile schedule needs T initialization fences and a final output fence. A
    panel needs two fences per group of at most 64 tiles. T >= 2 saves at least
    one fence, without increasing the transferred bytes or accumulator work.
    This is a structural selection hypothesis, not a measured timing claim.
    """
    tiles = ((m + 15) // 16) * ((n + 15) // 16)
    if not two_sources:
        return 'tile', 'single_source_keeps_tile'
    if requested == 'acc_panel':
        return 'acc_panel', 'explicit_acc_panel'
    if policy != 'regional':
        return 'tile', 'incumbent_tile'
    if bias_output_alias:
        return 'tile', 'broadcast_output_alias_keeps_tile'
    if tiles < 2:
        return 'tile', 'single_tile_no_fence_batching'
    return 'acc_panel', 'multiple_sum_tiles_batch_initialization_fences'


def emit_elementwise(e, cb, cmd, pointers=None):
    """Emit recognized standalone semantic commands; return False for other ops.

    ``pointers`` maps logical tensor names to LLVM pointer SSA values and defaults
    to the emitter's argument map. Unsupported recognized domains raise ValueError
    so callers cannot accidentally substitute CPU tensor arithmetic.
    """
    opcode = cmd['opcode']
    if opcode not in ('MOVEMENT', 'BIAS_ADD', 'RESIDUAL_ADD', 'INTEGER_ADD', 'INTEGER_RELU', 'SATURATING_NARROW'):
        return False
    ts = cb['tensors']
    p = e.args if pointers is None else pointers
    ops, attrs = cmd['operands'], cmd.get('attributes', {})
    out = ops['dst']
    dst = ts[out]
    m, n = _matrix(dst)
    dtype = dst['dtype']
    if attrs.get('output_dtype', dtype) != dtype:
        raise ValueError(f'{opcode}: declared output dtype disagrees with destination')
    if opcode == 'RESIDUAL_ADD':
        a, b = ops['lhs'], ops['rhs']
        if any(ts[x]['shape'] != dst['shape'] or ts[x]['dtype'] != 'i8' for x in (a, b)) or dtype != 'i8':
            raise ValueError('RESIDUAL_ADD standalone operand-sum domain is matching i8 tensors -> i8')
        stages = attrs.get('epilogue', [])
        if stages not in ([], ['relu']):
            raise ValueError('RESIDUAL_ADD admits only a final optional relu')
        ls, rs, readout = residual_load_scales(attrs['lhs_scale'], attrs['rhs_scale'], attrs['bound_lsb'], bool(stages))
        setup = e.insn('config', *config_load(n, scale=ls, shrink=True))
        e.insn('config', *config_load(n, state_id=1, scale=rs, shrink=True))
        e.insn('config', *config_store(n, scale=readout, activation=int(bool(stages))))
        src_dtype = 'i8'
        full = 0
    elif opcode == 'SATURATING_NARROW':
        a, b = ops['src'], None
        if ts[a]['shape'] != dst['shape'] or ts[a]['dtype'] != 'i32' or dtype != 'i8':
            raise ValueError('SATURATING_NARROW requires matching i32 source and i8 result')
        stages = attrs.get('epilogue', [])
        if stages not in ([], ['relu']):
            raise ValueError('SATURATING_NARROW admits only an optional signed relu')
        # Source clamp thresholds are exactly representable in f32. Conversion
        # and unit multiplication cannot move an out-of-range i32 value across
        # either threshold, so this readout equals integer saturation exactly.
        src_dtype, full = 'i32', 0
        setup = e.insn('config', *config_load(n * 4, shrink=False))
        e.insn('config', *config_store(n, scale=1.0, activation=int(bool(stages))))
    elif opcode == 'INTEGER_RELU':
        a, b = ops['src'], None
        if ts[a]['shape'] != dst['shape'] or ts[a]['dtype'] != 'i8' or dtype != 'i8':
            raise ValueError('standalone INTEGER_RELU requires i8 input and result; i32 full readout bypasses activation')
        src_dtype, full = 'i8', 0
        setup = e.insn('config', *config_load(n, shrink=True))
        e.insn('config', *config_store(n, activation=1))
    elif opcode in ('BIAS_ADD', 'INTEGER_ADD'):
        a = ops['src'] if opcode == 'BIAS_ADD' else ops['lhs']
        b = ops['bias'] if opcode == 'BIAS_ADD' else ops['rhs']
        bias_shape = [n] if opcode == 'BIAS_ADD' else dst['shape']
        if (dtype != 'i32' or ts[a]['dtype'] not in ('i8', 'i32') or ts[b]['dtype'] not in ('i8', 'i32')
                or ts[a]['shape'] != dst['shape'] or ts[b]['shape'] != bias_shape):
            raise ValueError(f'{opcode}: standalone operand-sum requires matching signed i8/i32 sources and i32 output')
        if attrs.get('epilogue', []) not in ([], ['bias_add']):
            raise ValueError(f'{opcode}: unsupported accumulator arithmetic epilogue')
        ab = 4 if ts[a]['dtype'] == 'i32' else 1
        bb = 4 if ts[b]['dtype'] == 'i32' else 1
        setup = e.insn('config', *config_load(n * ab, shrink=ab == 1))
        e.insn('config', *config_load(0 if opcode == 'BIAS_ADD' else n * bb, state_id=1, shrink=bb == 1))
        e.insn('config', *config_store(n * 4))
        src_dtype, full = ts[a]['dtype'], ACC_FULL
    else:
        a, b = ops['src'], None
        src_dtype = ts[a]['dtype']
        if ts[a]['shape'] != dst['shape'] or src_dtype not in ('i8', 'i32') or dtype not in ('i8', 'i32'):
            raise ValueError('MOVEMENT admits identity i8/i32 tensors with matching shapes')
        if src_dtype == 'i32' and dtype == 'i8':
            raise ValueError('MOVEMENT forbids narrowing i32 to i8')
        if attrs.get('semantic', 'mvin_mvout') != 'mvin_mvout':
            raise ValueError('MOVEMENT unknown device movement semantic')
        setup = e.insn('config', *config_load(n * (4 if src_dtype == 'i32' else 1), shrink=src_dtype == 'i8'))
        e.insn('config', *config_store(n * (4 if dtype == 'i32' else 1)))
        full = ACC_FULL if dtype == 'i32' else 0
    order = knob('elementwise_order', default='tile', values=('tile', 'acc_panel'),
                 root=Path(__file__).parent)
    policy = knob('elementwise_policy', default='incumbent',
                  values=('incumbent', 'regional'), root=Path(__file__).parent)
    order, reason = _select_elementwise_order(
        m, n, b is not None, order, policy,
        bias_output_alias=opcode == 'BIAS_ADD' and p[out] is p[b])
    setup.attributes['merlin.elementwise_policy'] = StringAttr(policy)
    setup.attributes['merlin.elementwise_path'] = StringAttr(order)
    setup.attributes['merlin.elementwise_reason'] = StringAttr(reason)
    setup.attributes['merlin.elementwise_geometry'] = StringAttr(
        f'M={m};N={n};tiles={((m + 15) // 16) * ((n + 15) // 16)};'
        'panel_tiles=64;acc_rows=1024')
    if order == 'acc_panel':
        _operand_sum_panel(e, m, n, a, b, out, p, ts, opcode, dtype, full)
        return True
    def tile(r, c, nr, nc):
        index = e.b.add_i(e.b.mul_i(r, e.b.const(n)), c)
        e.insn('mvin', e.address(e.ptr(p[a], index, src_dtype)), pack_matrix(ACC_ADDR, nr, nc))
        if b is not None:
            # Different load queues may complete out of issue order. The first
            # load must finish initializing these rows before the second load
            # reads their previous contents for accumulator addition.
            e.fence()
            bi = c if opcode == 'BIAS_ADD' else index
            e.insn('mvin2', e.address(e.ptr(p[b], bi, ts[b]['dtype'])), pack_matrix(ACC_ADDR | ACC_ACCUMULATE, nr, nc))
        e.insn('mvout', e.address(e.ptr(p[out], index, dtype)), pack_matrix(ACC_ADDR | full, nr, nc))
    _tiles(e, m, n, tile)
    # The next operation may consume this result through DRAM. Local-memory
    # scoreboarding cannot order mvout(accumulator) against mvin(scratchpad).
    e.fence()
    return True


def emit_graph_elementwise(e, op, host):
    """Recognize typed parallel sums, movement, ReLU, and saturating narrowing.

    This adapter proves the scalar SSA expression and every indexing map. An i8
    `addi` is wrapping arithmetic and is deliberately not read as saturation.
    Integer extensions preserve signedness; unsigned extensions are not elided.
    """
    from merlin.targetgen.contract import linalg_iface as li
    from xdsl.ir.affine import AffineDimExpr
    if li._op_name(op) != 'linalg.generic' or len(op.results) != 1:
        return False
    ins, outs = li._ins_outs(op)
    if not ins or len(outs) != 1:
        return False
    if any(getattr(it.data, 'value', None) != 'parallel' for it in op.iterator_types):
        return False
    if any(str(v.type.element_type) not in ('i8', 'i32') for v in ins):
        return False
    shape = tuple(op.results[0].type.get_shape())
    dtype = str(op.results[0].type.element_type)
    maps = [x.data for x in op.indexing_maps]
    rank = len(shape)
    def identity(amap):
        return (amap.num_dims == rank and len(amap.results) == rank
                and all(isinstance(x, AffineDimExpr) and x.position == i
                        for i, x in enumerate(amap.results)))
    if not identity(maps[-1]):
        return False
    input_values = [host.values[v] for v in ins]
    full = [identity(amap) and tuple(v.shape) == shape for amap, v in zip(maps, input_values)]
    vector = [amap.num_dims == rank and len(amap.results) == 1
              and isinstance(amap.results[0], AffineDimExpr)
              and amap.results[0].position == rank - 1 and tuple(v.shape) == (shape[-1],)
              for amap, v in zip(maps, input_values)]
    block = op.regions[0].blocks[0]
    env = {a: ('arg', i, str(a.type)) for i, a in enumerate(block.args)}
    yielded = None
    for scalar in block.ops:
        kind = li._op_name(scalar)
        if kind == 'linalg.yield':
            if len(scalar.operands) != 1:
                return False
            yielded = env.get(scalar.operands[0])
            continue
        if len(scalar.results) != 1:
            return False
        if kind == 'arith.constant':
            value = li._scalar_const(scalar.results[0])
            if not isinstance(value, int):
                return False
            expr = ('const', value)
        else:
            args = []
            for v in scalar.operands:
                expr = env.get(v)
                if expr is None:
                    # Captured scalar constants often live outside the generic
                    # body (PB clamp bounds do). Read their defining SSA value,
                    # never a runtime host value or a provenance/name hint.
                    value = li._scalar_const(v)
                    if isinstance(value, int) and str(v.type) in ('i8', 'i32'):
                        expr = ('const', value)
                args.append(expr)
            if any(v is None for v in args):
                return False
            if kind == 'arith.extsi' and str(scalar.results[0].type) == 'i32':
                expr = args[0]
            elif kind == 'arith.trunci':
                if str(scalar.operands[0].type) != 'i32' or str(scalar.results[0].type) != 'i8':
                    return False
                expr = (kind, *args)
            elif kind in ('arith.addi', 'arith.maxsi', 'arith.minsi'):
                expr = (kind, *args)
            else:
                return False
        env[scalar.results[0]] = expr
    if yielded is None:
        return False
    def argument(expr):
        return expr is not None and expr[0] == 'arg' and expr[1] < len(ins)
    opcode = None
    operands = {}
    attributes = {'output_dtype': dtype}
    if argument(yielded) and full[yielded[1]]:
        i = yielded[1]
        if dtype == input_values[i].dtype or (dtype == 'i32' and input_values[i].dtype == 'i8'):
            opcode, operands = 'MOVEMENT', {'src': str(i)}
    elif yielded[0] == 'arith.trunci' and dtype == 'i8':
        def clamp_interval(expr):
            # A sequence of signed min/max with constants is a monotone clamp.
            # Propagating its endpoint interval proves commuted operands and
            # either lower/upper clamp order, including redundant weaker bounds.
            if argument(expr) and expr[2] == 'i32':
                return expr[1], -(1 << 31), (1 << 31) - 1
            if expr[0] not in ('arith.minsi', 'arith.maxsi'):
                return None
            left, right = expr[1:]
            if left[0] == 'const':
                constant, child = left[1], right
            elif right[0] == 'const':
                constant, child = right[1], left
            else:
                return None
            state = clamp_interval(child)
            if state is None:
                return None
            index, lo, hi = state
            f = min if expr[0] == 'arith.minsi' else max
            return index, f(lo, constant), f(hi, constant)
        state = clamp_interval(yielded[1])
        if state is not None:
            i, lo, hi = state
            if (lo, hi) in ((-128, 127), (0, 127)) and full[i] and input_values[i].dtype == 'i32':
                opcode, operands = 'SATURATING_NARROW', {'src': str(i)}
                attributes['epilogue'] = ['relu'] if lo == 0 else []
    elif yielded[0] == 'arith.addi' and dtype == 'i32':
        a, b = yielded[1:]
        if argument(a) and argument(b):
            ai, bi = a[1], b[1]
            if full[ai] and full[bi]:
                opcode, operands = 'INTEGER_ADD', {'lhs': str(ai), 'rhs': str(bi)}
            elif full[ai] and vector[bi]:
                opcode, operands = 'BIAS_ADD', {'src': str(ai), 'bias': str(bi)}
            elif vector[ai] and full[bi]:
                opcode, operands = 'BIAS_ADD', {'src': str(bi), 'bias': str(ai)}
    elif yielded[0] == 'arith.maxsi' and dtype == 'i8':
        a, b = yielded[1:]
        arg = a if argument(a) and b == ('const', 0) else b if argument(b) and a == ('const', 0) else None
        if arg is not None and full[arg[1]] and input_values[arg[1]].dtype == 'i8':
            opcode, operands = 'INTEGER_RELU', {'src': str(arg[1])}
    if opcode is None:
        return False
    dst = host.result(op.results[0])
    tensors = {str(i): {'shape': list(v.shape), 'dtype': v.dtype} for i, v in enumerate(input_values)}
    tensors['dst'] = {'shape': list(dst.shape), 'dtype': dst.dtype}
    pointers = {str(i): v.ptr for i, v in enumerate(input_values)}
    pointers['dst'] = dst.ptr
    operands['dst'] = 'dst'
    emit_elementwise(e, {'tensors': tensors}, {'opcode': opcode, 'operands': operands,
                     'attributes': attributes}, pointers)
    return True
