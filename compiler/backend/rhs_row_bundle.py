"""Pack independent M=1 products into checked dense primitive rows.

The virtual matrix row i is exactly member i's original immutable LHS row;
its output row is exactly that member's original fresh output. Only DMA calls
are split. RHS, preload, MAC, reduction order and accumulator seeding come from
one checked compact dense schedule. No host tensor arithmetic or copies occur.
"""
from __future__ import annotations

from merlin.common.symbolic_lock import serialized_symbolic_check

from dataclasses import dataclass, replace

from merlin.compile.scheduling.batched_dense import (
    BatchMatrixView, BatchedDenseProblem, counted_batched_dense_products,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Loop, TensorArg, as_expr, check_structure

from . import isa
from .dense import _schedule
from .dense_readout import _pack


@dataclass(frozen=True)
class _Bundle:
    members: tuple
    k: int
    n: int
    schedule: object
    kernel: object
    receipt: dict

    def revalidate(self):
        self.schedule.revalidate()
        expected, receipt = _mapped_kernel(self.schedule, self.members, self.k, self.n)
        if expected != self.kernel or receipt != self.receipt:
            raise ValueError('row bundle source, pointer mapping or emitted schedule changed')


@serialized_symbolic_check
def _mapped_kernel(schedule, members, k, n):
    """Checked row bijection with a complete inverse to the original kernel."""
    import z3
    from merlin.sched.ir import Const, Var, Sum, Scaled, Select
    count = len(members)
    schedule.revalidate()
    expected = [('lhs',(count,k),'i8','read'), ('rhs',(k,n),'i8','read'),
                ('output',(count,n),'i32','write')]
    if (not 1 <= count <= isa.DIM or
            [(a.name,a.shape,a.dtype,a.access) for a in schedule.kernel.args] != expected):
        raise ValueError('row bundle needs its exact zero-seeded virtual source ABI')
    queries = 0

    def expression(value, env):
        if isinstance(value, Const): return z3.IntVal(value.value)
        if isinstance(value, Var): return env[value.name]
        if isinstance(value, Sum): return sum(expression(v, env) for v in value.terms)
        if isinstance(value, Scaled): return expression(value.expr, env)*value.factor
        if isinstance(value, Select):
            return z3.If(env[value.var] == value.equals,
                         expression(value.then, env), expression(value.other, env))
        raise ValueError('row bundle encountered an unknown address expression')

    def prove(scopes, args, address):
        nonlocal queries
        env = {key:z3.Int(key) for key, _ in scopes}
        solver = z3.Solver(); solver.set(timeout=2000)
        solver.add(*(z3.And(env[key] >= 0, env[key] < extent) for key, extent in scopes))
        a = {key:expression(value, env) for key, value in args.items()}
        # Splitting transfers is valid for exactly this whole one-tile M range.
        solver.add(z3.Not(z3.And(a['dram_row'] == 0, a['rows'] == count,
            a[address] >= 0, a[address]+count <= isa.DIM)))
        queries += 1
        if solver.check() != z3.unsat:
            raise ValueError('row bundle cannot establish exact bounded original row coverage')

    def walk(items, scopes=()):
        mapped, inverse = [], []
        for item in items:
            if type(item) is Loop:
                if item.iter_args or item.results or item.yields:
                    raise ValueError('row bundle cannot transport carried completion events')
                body, restored = walk(item.body, (*scopes, (item.var, item.extent)))
                mapped.append(replace(item, body=body))
                inverse.append(replace(item, body=restored))
                continue
            if type(item) is not Call or any((item.unit, item.produces, item.awaits,
                                              item.stages, item.sets, item.assumes)):
                raise ValueError('row bundle needs the checked synchronous compact instruction form')
            a = dict(item.args)
            if item.instr in ('gemmini_mvin_lhs', 'gemmini_mvout'):
                load = item.instr == 'gemmini_mvin_lhs'
                address = 'addr' if load else 'acc'
                prove(scopes, a, address)
                split = []
                for index in range(count):
                    args = dict(a, rows=as_expr(1), dram_row=as_expr(0))
                    args[address] = a[address]+index
                    split.append(replace(item,
                        instr=('row_bundle_lhs_' if load else 'row_bundle_out_')+str(index),
                        args=tuple(args.items())))
                # Independently invert the complete ordered row scatter/gather.
                for index, part in enumerate(split):
                    args = dict(part.args)
                    expected = dict(a, rows=as_expr(1), dram_row=as_expr(0))
                    expected[address] = a[address]+index
                    if (part.args != tuple(expected.items()) or part.instr !=
                            ('row_bundle_lhs_' if load else 'row_bundle_out_')+str(index)):
                        raise ValueError('row bundle row mapping is not an exact bijection')
                mapped.extend(split)
                inverse.append(item)
            elif item.instr in ('gemmini_mvin_weight', 'gemmini_preload', 'gemmini_compute_preloaded'):
                mapped.append(item); inverse.append(item)
            else:
                raise ValueError('row bundle only accepts zero-seeded raw products')
        return tuple(mapped), tuple(inverse)

    body, inverse = walk(schedule.kernel.body)
    if inverse != schedule.kernel.body:
        raise ValueError('row bundle inverse changed an original product operation')
    arguments = tuple(TensorArg('lhs_'+str(i), (1,k), 'i8', 'read')
                      for i in range(count))
    arguments += (TensorArg('rhs', (k,n), 'i8', 'read'),)
    arguments += tuple(TensorArg('output_'+str(i), (1,n), 'i32', 'write')
                       for i in range(count))
    kernel = replace(schedule.kernel, name='rhs_row_bundle', args=arguments, body=body)
    if check_structure(kernel):
        raise ValueError('row bundle introduced invalid counted index scope')
    source = BatchedDenseProblem((count,), BatchMatrixView((count,1,k), (0,)),
                                BatchMatrixView((k,n), ()))
    original = counted_batched_dense_products(source, name='original_row_products',
                                              instruction='original_raw_product')
    bindings = tuple(dict(row=i, lhs=row['lhs'], rhs=row['rhs'], output=row['dst'],
        source_op_index=row.get('attributes',{}).get('merlin.source_op_index'))
        for i,row in enumerate(members))
    return kernel, dict(schema='authored_rhs_row_bundle_projection.v1',
        original_products=source.record(), original_product_index_sha256=original.digest(),
        pointer_bindings=bindings, virtual_dense_kernel_sha256=schedule.kernel.digest(),
        mapped_kernel_sha256=kernel.digest(), symbolic_row_queries=queries,
        complete_inverse_projection=True,
        row_law='virtual LHS[i,k]=original lhs_i[0,k]; original output_i[0,n]=virtual output[i,n]',
        numeric_law='each row retains its signed i8 product and original K order, modulo i32',
        effects='distinct LHS SPAD rows and output ACC rows; shared immutable RHS; fresh disjoint outputs')


def _prepare(ctx, members):
    members = tuple(members)
    if not 2 <= len(members) <= 256:
        return None
    # The caller supplies one uninterrupted independent run, e.g. the existing
    # rhs_product_groups discovery. Never infer RHS equality from contents.
    if ctx.cb.get('params',{}).get('tensor_aliases'):
        return None
    rhs = members[0].get('rhs')
    outputs = [row.get('dst') for row in members]
    inputs = {row.get(key) for row in members for key in ('lhs','rhs')}
    if (rhs is None or None in inputs or None in outputs or
            len(set(outputs)) != len(outputs) or set(outputs)&inputs or
            any(row.get('rhs') != rhs for row in members)):
        return None
    k = n = None
    for row in members:
        attrs = row.get('attributes',{})
        known = {'epilogue','output_dtype','acc_scale','relu','bias','accumulator_init',
                 'acc_scale_by_column','requant_shift','bound_lsb',
                 # Frontend source provenance is descriptive only. These
                 # exact keys never affect source arithmetic or this plan.
                 'prov.region_id','prov.op','prov.family'}
        if any(key not in known and not key.startswith('merlin.') for key in attrs):
            return None
        scale = attrs.get('acc_scale',1.0)
        if (type(scale) not in (int,float) or scale != 1 or
            attrs.get('epilogue',()) not in ((),[]) or attrs.get('output_dtype','i32') != 'i32' or
            attrs.get('relu',False) is not False or attrs.get('bias') is not None or
            attrs.get('accumulator_init') is not None or attrs.get('acc_scale_by_column') is not None or
            attrs.get('requant_shift') is not None or attrs.get('bound_lsb',0) != 0):
            return None
        a,b,c = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
        if (a['dtype'] != 'i8' or b['dtype'] != 'i8' or c['dtype'] != 'i32' or
                len(a['shape']) != 2 or len(b['shape']) != 2 or
                tuple(a['shape']) != (1,b['shape'][0]) or
                tuple(c['shape']) != (1,b['shape'][1])):
            return None
        shape = tuple(b['shape'])
        if any(type(v) is not int or v <= 0 for v in shape): return None
        if k is not None and shape != (k,n): return None
        k,n = shape
        if any(key in row and row[key] != value for key,value in (('m',1),('k',k),('n',n))):
            return None
    prepared = []
    for offset in range(0,len(members),isa.DIM):
        chunk = members[offset:offset+isa.DIM]
        schedule = _schedule(len(chunk),k,n,False,False)
        kernel,receipt = _mapped_kernel(schedule,chunk,k,n)
        prepared.append(_Bundle(chunk,k,n,schedule,kernel,receipt))
    return tuple(prepared)


def lower_rhs_row_bundle(ctx, members):
    """Return receipt, or None before emitting for an unsupported whole group.

    The caller owns the uninterrupted independent interval and skips exactly
    these original commits only on success. Explicit alias modules, non-M1,
    nonidentity readouts and unknown executable attributes retain the incumbent.
    """
    try:
        prepared = _prepare(ctx,members)
        if prepared is None: return None
        for bundle in prepared: bundle.revalidate()
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    fb = ctx.fb
    ctx.fence()
    for bundle in prepared:
        ctx.current_source_index = bundle.members[0].get('attributes',{}).get('merlin.source_op_index')
        ctx.emit(*isa.config_ex(dataflow=1))
        ctx.emit(*isa.config_ld(bundle.k,channel=0))
        ctx.emit(*isa.config_ld(bundle.n,channel=1))
        ctx.emit(*isa.config_st(bundle.n*4))
        descriptor = [None]
        def emit(instruction, arguments):
            a = {key:bridge.value(value) for key,value in arguments.items()}
            if instruction.startswith('row_bundle_lhs_'):
                index = int(instruction.removeprefix('row_bundle_lhs_'))
                row = bundle.members[index]
                ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index')
                ctx.emit(*isa.mvin(ctx.ptr(row['lhs'],a['dram_col']),
                    a['addr'],a['rows'],a['cols'],channel=0,fb=fb))
            elif instruction == 'gemmini_mvin_weight':
                offset = fb.add_i(fb.mul_i(a['dram_row'],fb.const(bundle.n)),a['dram_col'])
                ctx.emit(*isa.mvin(ctx.ptr(bundle.members[0]['rhs'],offset),
                    a['addr'],a['rows'],a['cols'],channel=1,fb=fb))
            elif instruction == 'gemmini_preload':
                # Every split input DMA and shared weight tile is complete;
                # the compute that previously owned these rows was drained.
                ctx.fence()
                descriptor[0] = a['cols']
                acc = fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
                ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),
                         _pack(fb,acc,a['rows'],a['cols']))
            elif instruction == 'gemmini_compute_preloaded':
                if descriptor[0] is None: raise ValueError('row bundle compute lacks weight descriptor')
                ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),
                         _pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
                ctx.fence()
            elif instruction.startswith('row_bundle_out_'):
                index = int(instruction.removeprefix('row_bundle_out_'))
                row = bundle.members[index]
                ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index')
                ctx.emit(*isa.mvout(ctx.ptr(row['dst'],a['dram_col']),
                    isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
                ctx.fence()
            else: raise ValueError('unknown row bundle terminal instruction '+instruction)
        bridge = FnBuilderIndexAdapter(fb)
        emit_index_program(bundle.kernel,bridge,emit)
        ctx.fence()
    receipts = [bundle.receipt for bundle in prepared]
    receipt = dict(mechanism='shared_rhs_row_bundle', products=sum(len(b.members) for b in prepared),
        bundles=receipts, emitted_kernel_sha256=[b.kernel.digest() for b in prepared],
        completion='drain every input epoch before preload, every compute before reuse, every raw row readout',
        qualification='checked virtual dense source with exact original pointer row mapping; independent execution required')
    ctx.metadata.setdefault('rhs_row_bundles',[]).append(receipt)
    return receipt
