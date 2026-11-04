"""Physically concatenate independent short LHS tiles in scratchpad rows.

Inputs and outputs retain their original DRAM buffers. Only DMA staging rows
are concatenated; no CPU tensor packing or numerical arithmetic is emitted.
"""
from xdsl.dialects.builtin import i8, i32
from . import isa


def lower_shared_rhs_rows(ctx, command, column_tiles=4):
    """Compute independent common-RHS members with one array row group.

    Caller proves signed i8 A/B, raw i32 output, zero initial addends and
    mutually independent members. Require common M/K/N/RHS and heads*M<=16.
    A occupies rows0:16, B rows16:80, accumulators rows0:64. Ordinary retained
    loops bound code size independently of K and N. Each head's load/readout
    carries its source tags; commands shared by all heads use the first tag.
    """
    members = command['attributes']['members']
    if not 2 <= len(members) <= 8:
        raise ValueError('shared RHS row packing requires two through eight members')
    if type(column_tiles) is not int or column_tiles not in (1, 2, 4):
        raise ValueError('shared RHS row packing requires columns1,2,or4')
    first = members[0]
    m, k, n = first['m'], first['k'], first['n']
    rows = len(members) * m
    if (any(type(d) is not int or d <= 0 for d in (m, k, n))
            or rows > isa.DIM
            or any((x['m'], x['rhs'], x['k'], x['n']) !=
                   (m, first['rhs'], k, n) for x in members)):
        raise ValueError('shared RHS row packing requires common dimensions and heads*M<=16')
    writes = {x['dst'] for x in members}
    reads = {x[name] for x in members for name in ('lhs', 'rhs')}
    if len(writes) != len(members) or writes & reads:
        raise ValueError('shared RHS row packing members have interfering storage')
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
            for head, member in enumerate(members):
                tag(member)
                isa.mvin(f, f.gep(pointers[head][0], kk, i8),
                         head * m, tk, m, state=0)
            tag(first)
            for col, width in enumerate(widths):
                index = f.add_i(f.add_i(f.mul_i(kk, f.const(n)), nj), f.const(col * 16))
                isa.mvin(f, f.gep(b, index, i8), 16 + col * 16, width, tk, state=1)
            flags = f.or_i(f.const(0x80000000), f.shl_i(f.smin(kk, f.const(1)), 30))
            for col, width in enumerate(widths):
                isa.preload(f, 16 + col * 16, f.or_i(flags, f.const(col * 16)),
                            width, tk, width, rows)
                isa.compute(f, 0, isa.GARBAGE_ADDR, tk, rows, width, rows, flip=True)
        f.for_loop(0, k, 16, reduction, retain_loop=True)
        for head, member in enumerate(members):
            tag(member)
            for col, width in enumerate(widths):
                index = f.add_i(nj, f.const(col * 16))
                local = col * 16 + head * m
                isa.mvout(f, f.gep(pointers[head][1], index, i32),
                          0xA0000000 | local, width, m)
    step = column_tiles * 16
    full = n // step * step
    f.for_loop(0, full, step, lambda nj: group(nj, [16] * column_tiles), retain_loop=True)
    if full < n:
        widths = [min(16, n - start) for start in range(full, n, 16)]
        group(f.const(full), widths)
    isa.fence(f)
    f.source_op_index, f.region_id = saved
