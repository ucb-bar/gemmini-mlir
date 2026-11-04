"""Lower fused sibling contractions with shared A DMA and separate outputs."""
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from xdsl.dialects.builtin import i8, i32
from . import isa
from .row_policy import select_row_tiles
from .sibling_policy import fusion_decision

# Selected geometry: 512 accumulator rows per bank; 4096 scratchpad rows
# per bank. Use only one bank-sized window of each address space.
_ACCUMULATOR_BANK_ROWS = 512
_SCRATCHPAD_BANK_ROWS = 4096


def lower_siblings(ctx, command):
    members = command['attributes']['members']
    root = Path(__file__).resolve().parents[1]
    mode = knob('sibling_fusion', default=False, values=(False, True, 'auto', 'rhs', 'rhs_auto', 'rhs_rows'), root=root)
    # Auto decisions belong before the source-task merge. Check again here so
    # an accidentally merged rejected group cannot silently take the fused
    # route. Boolean modes preserve their existing lowering exactly.
    if mode == 'auto':
        decision = fusion_decision(members)
        if not decision['profitable']:
            raise ValueError('auto sibling fusion requires an accepted pre-merge policy decision: '
                             + decision['reason'])
    if not 2 <= len(members) <= 4:
        raise ValueError('sibling fusion requires two through four members')
    first = members[0]
    m, k = int(first['m']), int(first['k'])
    if any((x['lhs'], x['m'], x['k']) != (first['lhs'], m, k) for x in members):
        raise ValueError('sibling fusion members must share exact A storage and M/K')
    f = ctx.f
    a = ctx.ptr(first['lhs'])
    pointers = [(ctx.ptr(x['rhs']), ctx.ptr(x['dst'])) for x in members]
    row_tiles = knob('matmul_row_tiles', default=1, values=(1, 2, 4, 8, 16, 32),
                     root=root)
    row_tiles = select_row_tiles(m, k, row_tiles, root=root)
    # All members retain separate accumulator tiles concurrently. A uses one
    # scratchpad tile per row tile, followed by one shared B staging tile.
    row_tiles = min(row_tiles,
                    _ACCUMULATOR_BANK_ROWS // (len(members) * isa.DIM),
                    _SCRATCHPAD_BANK_ROWS // isa.DIM - 1)
    if row_tiles < 1:
        raise ValueError('sibling group exceeds selected local storage capacity')
    isa.config_ex(f, dataflow=1, activation=0, scale=1.0)
    isa.config_ld(f, k, state=0)
    # Raw i32 siblings have no bias load, leaving DMA states 1 and 2
    # available for RHS strides. Keep their configurations outside K when
    # the sibling set needs at most two distinct strides.
    rhs_strides = list(dict.fromkeys(int(x['n']) for x in members))
    rhs_states = {n: i + 1 for i, n in enumerate(rhs_strides)} if len(rhs_strides) <= 2 else {}
    for n, state in rhs_states.items():
        isa.config_ld(f, n, state=state)
    endpoints = sorted({0, *((int(x['n']) + 15) // 16 for x in members)})

    def tag(member):
        attrs = member['attributes']
        f.source_op_index = attrs.get('source_op_index')
        f.region_id = attrs.get('source_region_id')

    def group(mi, count):
        b_local = count * 16
        def each_row(body):
            if count == 1:
                # A single row tile needs no dynamic induction variable or
                # loop branch. This is common for decode and small tails.
                tm = f.smin(f.const(16), f.sub_i(f.const(m), mi))
                body(f.const(0), mi, tm)
                return
            def row(ri):
                row_start = f.add_i(mi, f.mul_i(ri, f.const(16)))
                tm = f.smin(f.const(16), f.sub_i(f.const(m), row_start))
                body(ri, row_start, tm)
            f.for_loop(0, count, 1, row, retain_loop=True)
        for begin, end in zip(endpoints, endpoints[1:]):
            active = [(i, x) for i, x in enumerate(members) if (int(x['n']) + 15) // 16 > begin]
            def columns(nj):
                def reduction(kk):
                    tk = f.smin(f.const(16), f.sub_i(f.const(k), kk))
                    tag(first)
                    def load_a(ri, row_start, tm):
                        a_index = f.add_i(f.mul_i(row_start, f.const(k)), kk)
                        isa.mvin(f, f.gep(a, a_index, i8), f.mul_i(ri, f.const(16)), tk, tm, state=0)
                    each_row(load_a)
                    flags = f.or_i(f.const(0x80000000), f.shl_i(f.smin(kk, f.const(1)), 30))
                    for i, member in active:
                        tag(member)
                        n = int(member['n'])
                        tn = f.smin(f.const(16), f.sub_i(f.const(n), nj))
                        b, _ = pointers[i]
                        state = rhs_states.get(n, 1)
                        if not rhs_states:
                            isa.config_ld(f, n, state=state)
                        index = f.add_i(f.mul_i(kk, f.const(n)), nj)
                        isa.mvin(f, f.gep(b, index, i8), b_local, tn, tk, state=state)
                        def compute_row(ri, row_start, tm):
                            local = f.add_i(f.const(i * count * 16), f.mul_i(ri, f.const(16)))
                            isa.preload(f, b_local, f.or_i(flags, local), tn, tk, tn, tm)
                            isa.compute(f, f.mul_i(ri, f.const(16)), isa.GARBAGE_ADDR, tk, tm, tn, tm, flip=True)
                        each_row(compute_row)
                f.for_loop(0, k, 16, reduction, retain_loop=True)
                for i, member in active:
                    tag(member)
                    n = int(member['n'])
                    tn = f.smin(f.const(16), f.sub_i(f.const(n), nj))
                    _, c = pointers[i]
                    isa.config_st(f, n * 4, scale=1.0, activation=0)
                    def store_row(ri, row_start, tm):
                        local = f.add_i(f.const(i * count * 16), f.mul_i(ri, f.const(16)))
                        index = f.add_i(f.mul_i(row_start, f.const(n)), nj)
                        isa.mvout(f, f.gep(c, index, i32), f.or_i(f.const(0xA0000000), local), tn, tm)
                    each_row(store_row)
            f.for_loop(begin * 16, end * 16, 16, columns, retain_loop=True)
    group_rows = row_tiles * 16
    full_rows = m // group_rows * group_rows
    f.for_loop(0, full_rows, group_rows, lambda mi: group(mi, row_tiles), retain_loop=True)
    if full_rows < m:
        group(f.const(full_rows), (m - full_rows + 15) // 16)
    isa.fence(f)
