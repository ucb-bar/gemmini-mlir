"""Optional full-K retained row bundle with the existing checked epoch plan.

Only original-pointer input and output DMA is split into primitive rows.
Additional DMA/readout drains keep the original issued-operation bounds;
RHS/preload/compute operations and their checked completion epochs are intact.
The optional compact bundle and all compiler routing remain unchanged.
"""
from __future__ import annotations

from merlin.common.symbolic_lock import serialized_symbolic_check
from dataclasses import dataclass, replace
from pathlib import Path
from hashlib import sha256
import json

from merlin.compile.scheduling.batched_dense import BatchMatrixView, BatchedDenseProblem, counted_batched_dense_products
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Loop, TensorArg, as_expr, check_structure
from . import isa
from .dense_pipeline import _retained_schedule
from .dense_readout import _pack
from .rhs_row_bundle import _prepare as _prepare_source


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
        kernel, receipt = _mapped_kernel(self.schedule,self.members,self.k,self.n)
        if kernel != self.kernel or receipt != self.receipt:
            raise ValueError('retained row bundle source, pointer mapping or epochs changed')


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
            a[address] >= 0, a[address] % isa.DIM == 0,
            a[address]+count <= (isa.SPAD_ROWS//2 if address == 'addr' else isa.ACC_ROWS))))
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
                # Added completion edges preserve the original issue bounds:
                # one split load/readout per fresh drained epoch. Compute
                # epochs remain exactly the existing checked ordered epochs.
                if load: mapped.append(Call('merlin_dense_panel_drain', ()))
                for part in split:
                    mapped.extend((part, Call('merlin_dense_panel_drain', ())))
                inverse.append(item)
            elif item.instr == 'merlin_dense_panel_drain' and not item.args:
                mapped.append(item); inverse.append(item)
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
    kernel = replace(schedule.kernel, name='rhs_row_bundle_retained', args=arguments, body=body)
    if check_structure(kernel):
        raise ValueError('row bundle introduced invalid counted index scope')
    source = BatchedDenseProblem((count,), BatchMatrixView((count,1,k), (0,)),
                                BatchMatrixView((k,n), ()))
    original = counted_batched_dense_products(source, name='original_row_products',
                                              instruction='original_raw_product')
    bindings = tuple(dict(row=i, lhs=row['lhs'], rhs=row['rhs'], output=row['dst'],
        source_op_index=row.get('attributes',{}).get('merlin.source_op_index'))
        for i,row in enumerate(members))
    return kernel, dict(schema='authored_rhs_row_bundle_retained_projection.v1',
        original_products=source.record(), original_product_index_sha256=original.digest(),
        pointer_bindings=bindings, virtual_dense_kernel_sha256=schedule.kernel.digest(),
        mapped_kernel_sha256=kernel.digest(), symbolic_row_queries=queries,
        complete_inverse_projection=True,
        completion_mapping='original checked compute epochs unchanged; each split DMA/readout separately drained',
        row_law='virtual LHS[i,k]=original lhs_i[0,k]; original output_i[0,n]=virtual output[i,n]',
        numeric_law='each row retains its signed i8 product and original K order, modulo i32',
        effects='distinct LHS SPAD rows and output ACC rows; shared immutable RHS; fresh disjoint outputs')


def _prepare(ctx, members):
    members = tuple(members)
    if not 2 <= len(members) <= isa.DIM: return None
    # Reuse exactly the optional bundle's raw source/alias/readout admission.
    source = _prepare_source(ctx,members)
    if source is None or len(source) != 1: return None
    original = source[0]
    schedule, refusal = _retained_schedule(len(members),original.k,original.n,
                                           False,False,False,'wide','half')
    if schedule is None: return None
    order_bytes = Path(__file__).with_name('dense_ordering.json').read_bytes()
    order = json.loads(order_bytes)
    if (order.get('completion_order',{}).get('execute',{}).get('order') != 'in_order' or
        schedule.realization['admission']['qualification_identity'] != order['identity'] or
        schedule.realization['schedule_ir']['pipeline_kind'] != 'retained_full_k'):
        return None
    kernel, receipt = _mapped_kernel(schedule,members,original.k,original.n)
    bundle = _Bundle(members,original.k,original.n,schedule,kernel,receipt)
    return bundle, dict(ordering_identity=order['identity'],
        ordering_file_sha256=sha256(order_bytes).hexdigest(),
        original_member_bindings_sha256=sha256(json.dumps(members,
            sort_keys=True,allow_nan=False).encode()).hexdigest(),
        selected_rows=dict(spad=isa.SPAD_ROWS,lhs=isa.SPAD_ROWS//2,
                           rhs=isa.SPAD_ROWS//2,acc=isa.ACC_ROWS),
        source_admission=dict(schedule.realization['admission']),
        plan=dict(schedule.realization['schedule_ir']),
        geometry=dict(schedule.realization['panel_geometry']))


def lower_rhs_row_bundle_retained(ctx, members):
    """Return a source-bound receipt, or None before mutation for refusal.

    Owns only a complete independent 2..16 member raw M1 product interval.
    Capacity and source failures preserve the caller's existing route.
    """
    try:
        prepared = _prepare(ctx,members)
        if prepared is None: return None
        bundle, law = prepared
        bundle.revalidate()
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    fb = ctx.fb
    ctx.fence()
    ctx.current_source_index = bundle.members[0].get('attributes',{}).get('merlin.source_op_index')
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(bundle.k,channel=0))
    ctx.emit(*isa.config_ld(bundle.n,channel=1))
    ctx.emit(*isa.config_st(bundle.n*4))
    descriptor = [None]
    def emit(instruction, arguments):
        a = {key:bridge.value(value) for key,value in arguments.items()}
        if instruction == 'merlin_dense_panel_drain':
            ctx.fence()
        elif instruction.startswith('row_bundle_lhs_'):
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
            descriptor[0] = a['cols']
            acc = fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),
                     _pack(fb,acc,a['rows'],a['cols']))
        elif instruction == 'gemmini_compute_preloaded':
            if descriptor[0] is None: raise ValueError('retained bundle compute lacks descriptor')
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),
                     _pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
        elif instruction.startswith('row_bundle_out_'):
            index = int(instruction.removeprefix('row_bundle_out_'))
            row = bundle.members[index]
            ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index')
            ctx.emit(*isa.mvout(ctx.ptr(row['dst'],a['dram_col']),
                isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
        else: raise ValueError('unknown retained bundle instruction '+instruction)
    bridge = FnBuilderIndexAdapter(fb)
    emission = emit_index_program(bundle.kernel,bridge,emit)
    ctx.fence()
    receipt = dict(mechanism='shared_rhs_row_bundle_retained_full_k',products=len(bundle.members),
        bundles=[bundle.receipt], emitted_kernel_sha256=[bundle.kernel.digest()],
        retained_plan=law, emission=emission,
        completion='original selected ordered pure-compute epochs; real drain per split DMA and raw row readout',
        qualification='checked retained virtual dense source and exact original-pointer mapping; independent execution required')
    ctx.metadata.setdefault('rhs_row_bundles_retained',[]).append(receipt)
    return receipt


# Same two-argument optional-lowering convention as the incumbent adapter.
lower = lower_rhs_row_bundle_retained
