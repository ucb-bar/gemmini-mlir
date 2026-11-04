"""Exact one-tile contractions sharing one RHS tile and separate source pointers."""
from xdsl.dialects.builtin import i8, i32
from . import isa


def lower_shared_rhs(ctx, command):
    """Emit each original output without packing or CPU tensor arithmetic.

    The graph rewrite proves a mutually independent source-task set, signed i8
    operands, zero i32 destinations and identical immutable RHS storage/M/K/N.
    At most eight distinct accumulator tiles occupy rows0:128, and the basic
    tile staging windows occupy scratchpad rows0:32. Completion is explicit.
    """
    members = command['attributes']['members']
    if not 2 <= len(members) <= 8:
        raise ValueError('shared RHS requires two through eight members')
    first = members[0]
    m, k, n = first['m'], first['k'], first['n']
    if (any(type(d) is not int or d <= 0 for d in (m, k, n))
            or m > isa.DIM
            or any((x['m'], x['rhs'], x['k'], x['n']) !=
                   (m, first['rhs'], k, n) for x in members)):
        raise ValueError('shared RHS members require common M1..16 and identical RHS/K/N')
    writes = {x['dst'] for x in members}
    reads = {x[name] for x in members for name in ('lhs', 'rhs')}
    if len(writes) != len(members) or writes & reads:
        raise ValueError('shared RHS members have interfering source storage')
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    choice = command['attributes'].get('_shared_rhs_columns')
    if choice is None:
        choice = knob('matmul_column_tiles', default=1,
                      values=(1, 2, 4, 'auto', 'permute'),
                      root=Path(__file__).resolve().parents[1])
    elif type(choice) is not int or choice not in (1, 4):
        raise ValueError('automatic shared RHS requires internal columns1 or4')
    mode = knob('sibling_fusion', default=False,
                values=(False, True, 'auto', 'rhs', 'rhs_auto', 'rhs_rows'),
                root=Path(__file__).resolve().parents[1])
    automatic_rows = command['attributes'].get('_shared_rhs_combine_rows', False)
    if automatic_rows and (automatic_rows is not True or len(members) != 8 or m != 1):
        raise ValueError('automatic physical row combination requires eight single-row members')
    if automatic_rows or mode == 'rhs_rows' and len(members) * m <= isa.DIM:
        from .shared_rhs_rows import lower_shared_rhs_rows
        return lower_shared_rhs_rows(ctx, command,
                                     column_tiles=choice if choice in (1, 2, 4) else 4)
    if choice in (2, 4):
        return _lower_columns(ctx, members, choice)
    f = ctx.f
    saved = (getattr(f, 'source_op_index', None), getattr(f, 'region_id', None))
    b = ctx.ptr(first['rhs'])
    pointers = [(ctx.ptr(x['lhs']), ctx.ptr(x['dst'])) for x in members]
    def tag(member):
        attrs = member['attributes']
        f.source_op_index = attrs.get('source_op_index')
        f.region_id = attrs.get('source_region_id')
    tag(first)
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, k, state=0)
    isa.config_ld(f, n, state=1)
    isa.config_st(f, n * 4, scale=1.0, activation=0)
    def columns(nj):
        tn = f.smin(f.const(16), f.sub_i(f.const(n), nj))
        def reduction(kk):
            tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
            tag(first)
            b_index = f.add_i(f.mul_i(kk, f.const(n)), nj)
            isa.mvin(f, f.gep(b, b_index, i8), 16, tn, tk, state=1)
            flags = f.or_i(f.const(0x80000000),
                           f.shl_i(f.smin(kk, f.const(1)), 30))
            for i, member in enumerate(members):
                tag(member)
                a, _ = pointers[i]
                isa.mvin(f, f.gep(a, kk, i8), 0, tk, m, state=0)
                isa.preload(f, 16, f.or_i(flags, f.const(i * 16)), tn, tk, tn, m)
                isa.compute(f, 0, isa.GARBAGE_ADDR, tk, m, tn, m, flip=True)
        f.for_loop(0, k, 16, reduction, retain_loop=True)
        for i, member in enumerate(members):
            tag(member)
            isa.mvout(f, f.gep(pointers[i][1], nj, i32),
                      0xA0000000 | i * 16, tn, m)
    f.for_loop(0, n, 16, columns, retain_loop=True)
    isa.fence(f)
    f.source_op_index, f.region_id = saved


def _lower_columns(ctx, members, column_tiles):
    """Explicit adjacent-N bundle; A is shared across columns, B across heads.

    The caller has completed the same source-semantic checks as col1. Up to
    eight heads times four columns times16 rows fits the512-row window. All B
    staging tiles fit scratchpad16:80, while A uses0:16. Tail groups keep their
    original source strides and store only live columns.
    """
    first = members[0]
    m, k, n = first['m'], first['k'], first['n']
    f = ctx.f
    saved = (getattr(f, 'source_op_index', None), getattr(f, 'region_id', None))
    b = ctx.ptr(first['rhs'])
    pointers = [(ctx.ptr(x['lhs']), ctx.ptr(x['dst'])) for x in members]
    def tag(member):
        attrs = member['attributes']
        f.source_op_index = attrs.get('source_op_index')
        f.region_id = attrs.get('source_region_id')
    tag(first)
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, k, state=0)
    isa.config_ld(f, n, state=1)
    isa.config_st(f, n * 4, scale=1.0, activation=0)
    def group(nj, widths):
        def reduction(kk):
            tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
            tag(first)
            for col, width in enumerate(widths):
                index = f.add_i(f.add_i(f.mul_i(kk, f.const(n)), nj),
                                f.const(col * 16))
                isa.mvin(f, f.gep(b, index, i8), 16 + col * 16,
                         width, tk, state=1)
            flags = f.or_i(f.const(0x80000000),
                           f.shl_i(f.smin(kk, f.const(1)), 30))
            for head, member in enumerate(members):
                tag(member)
                isa.mvin(f, f.gep(pointers[head][0], kk, i8), 0, tk, m, state=0)
                for col, width in enumerate(widths):
                    local = (head * column_tiles + col) * 16
                    isa.preload(f, 16 + col * 16, f.or_i(flags, f.const(local)),
                                width, tk, width, m)
                    isa.compute(f, 0, isa.GARBAGE_ADDR, tk, m, width, m, flip=True)
        f.for_loop(0, k, 16, reduction, retain_loop=True)
        for head, member in enumerate(members):
            tag(member)
            for col, width in enumerate(widths):
                index = f.add_i(nj, f.const(col * 16))
                local = (head * column_tiles + col) * 16
                isa.mvout(f, f.gep(pointers[head][1], index, i32),
                          0xA0000000 | local, width, m)
    step = column_tiles * 16
    full = n // step * step
    f.for_loop(0, full, step, lambda nj: group(nj, [16] * column_tiles),
               retain_loop=True)
    if full < n:
        widths = [min(16, n - start) for start in range(full, n, 16)]
        group(f.const(full), widths)
    isa.fence(f)
    f.source_op_index, f.region_id = saved
