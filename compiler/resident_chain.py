"""Source-bound independent products sharing one immutable complete RHS.

The caller owns grouping and external alias admission. This module owns the
checked local-storage epoch and lowers its actual schedule through Encoder.
Unsupported groups decline before issuing any machine effects.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import isqrt
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Load, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
from merlin.compile.scheduling.dense_shared_rhs_check import DenseRHSReuseLaw
from merlin.sched.ir import Ptr, as_expr
from merlin.transforms.dense_planning import DenseProblem, DenseResources

from dense import _capacities, _execution_laws, dense_adapter


def _readout(product):
    """A source-declared unit-scale clamp, with no floating reassociation."""
    dtype = product.get('out_dtype', 'i32')
    scale = product.get('scale', 1.0)
    relu = product.get('relu', False)
    if product.get('bias') is not None or type(scale) not in (float, int) or scale != 1:
        raise ValueError('resident product needs zero seed and original unit scale')
    if dtype == 'i32' and not relu:
        return None
    if dtype != 'i8':
        raise ValueError('resident output requires raw i32 or saturating i8')
    low = 0 if relu else -128
    scalar = '''module {
      func.func @readout(%x: i32) -> i8 {
        %low = arith.constant LOW : i32
        %high = arith.constant 127 : i32
        %a = arith.maxsi %x, %low : i32
        %b = arith.minsi %a, %high : i32
        %y = arith.trunci %b : i32 to i8
        func.return %y : i8
      }
    }'''.replace('LOW', str(low))
    return DenseIntegerReadout(scalar, 'i8', 'store', ())


def plan_resident_products(products, *, source_identity, external_buffers_disjoint,
                           ready_frontier=False,compute_order=None):
    """Return a checked shared-RHS product or None, without touching Encoder.

    Products are dictionaries with lhs/out pointers, positive m/k/n, and the
    ordinary out_dtype/scale/relu/bias readout options. They must be independent
    original source products using the identical immutable source RHS handle.
    The caller must retain command order and exclude intervening local effects,
    dependent members, RHS writes and external aliases before setting admission.
    """
    if (external_buffers_disjoint is not True or type(source_identity) is not str
            or not source_identity or not 2 <= len(products) <= 128):
        return None
    try:
        dimensions = tuple((p['m'], p['k'], p['n']) for p in products)
        if any(type(d) is not int or d <= 0 for shape in dimensions for d in shape):
            return None
        if len({(k, n) for _, k, n in dimensions}) != 1:
            return None
        readouts = tuple(_readout(p) for p in products)
    except (KeyError, TypeError, ValueError):
        return None
    kr, edge, operand, rows, rowbytes, accumulator, accrows, accbytes = _capacities()
    if kr != edge:
        return None
    lhsrows = rows // 2
    storage = DenseLoopStorage(
        DenseMemoryView('lhs', operand, rowbytes, 0, lhsrows),
        DenseMemoryView('rhs', operand, rowbytes, lhsrows, rows-lhsrows),
        DenseMemoryView('acc', accumulator, accbytes, 0, accrows))
    problems = tuple(DenseProblem(m, k, n, edge, kr, edge, 1, 1, 4)
                     for m, k, n in dimensions)
    resources = DenseResources(rows*rowbytes, accrows*accbytes,
        max_m_blocks=min(max(p.tiles[0] for p in problems), isqrt(accrows//edge)),
        max_n_blocks=min(problems[0].tiles[2], isqrt(accrows//edge)),
        max_k_blocks=problems[0].tiles[1], partial_output_tiles=True,
        partial_output_blocks=True)
    adapter = replace(dense_adapter(), qualification_identity='gemmini-resident-actual-fields:' +
        sha256((Path(__file__).with_name('encoding.py')).read_bytes()).hexdigest())

    def encode_member(member, op):
        call = adapter.encode(op)
        # The member is instruction metadata, never a runtime selector. This
        # preserves the exact pointer binding all the way to Encoder.
        tag = 'shared' if member is None else str(member)
        args = call.args
        if isinstance(op, Load):
            weight = op.role == 'weight'
            pitch = problems[0].n if weight else problems[member].k
            args += (('ptr', Ptr('rhs' if weight else f'lhs{member}',
                as_expr(op.dram_row)*pitch + op.dram_col)),)
        elif isinstance(op, Store):
            width = 1 if readouts[member] is not None else 4
            args += (('ptr', Ptr(f'output{member}',
                (as_expr(op.dram_row)*problems[member].n+op.dram_col)*width)),)
        return replace(call, instr=f'{call.instr}__{tag}', args=args)

    def project_member(call):
        instruction, tag = call.instr.rsplit('__', 1)
        member = None if tag == 'shared' else int(tag)
        projected = adapter.project(replace(call, instr=instruction,
            args=tuple((k, v) for k, v in call.args if k != 'ptr')))
        if isinstance(projected, Store) and readouts[member] is not None:
            projected = replace(projected, readout=readouts[member].identity)
        return member, projected

    _, reduction, epoch = _execution_laws(adapter)
    law = DenseRHSReuseLaw(adapter.configuration_identity,
        adapter.numerical_policy_identity, adapter.qualification_identity,
        'source-independent-products-immutable-rhs-private-local-epoch-v1',
        rhs_unchanged_by_member_operations=True, independent_member_interleaving=True,
        suppressed_loads_no_faults=True, suppressed_loads_no_extra_observations=True)
    result = schedule_dense_shared_rhs(problems, resources, (storage,)*len(problems), adapter,
        encode_member=encode_member, project_member=project_member,
        source_identity=source_identity, name='resident_products',
        dtypes=('i8', 'i8', 'i32'), static_budget=256,
        external_buffers_disjoint=True, retention_law=law,
        epoch_admission=reduction, epoch_law=epoch,
        output_readouts=readouts,
        readout_qualification_identities=tuple(None if r is None else
            'gemmini-unit-scale-rne-saturating-readout:'+r.identity for r in readouts),
        verification_timeout_ms=20000)
    if result is not None and compute_order is not None:
        if ready_frontier:raise ValueError('compute order alternative requires separate readout')
        if any(r is not None for r in readouts):return None
        from shared_rhs_order import compose_shared_rhs_order
        result=compose_shared_rhs_order(result,problems,resources,(storage,)*len(problems),
            adapter,encode_member,project_member,reduction,epoch,order=compute_order)
    if result is not None and ready_frontier:
        if any(r is not None for r in readouts):return None
        from shared_rhs_ready import compose_shared_rhs_ready
        result=compose_shared_rhs_ready(result,problems,resources,(storage,)*len(problems),
            adapter,encode_member,project_member,reduction,epoch)
    return None if result is None else (problems, result)


def emit_resident_products(enc, rhs, products, *, source_identity,
                           external_buffers_disjoint=False,ready_frontier=False,compute_order=None):
    """Emit the checked original group; None means the caller uses its fallback."""
    planned = plan_resident_products(products, source_identity=source_identity,
        external_buffers_disjoint=external_buffers_disjoint,ready_frontier=ready_frontier,
        compute_order=compute_order)
    if planned is None:
        return None
    problems, result = planned
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from merlin.sched.ir import Loop

    def call_count(body):
        # Bound code size before expansion; never walk a dynamic iteration
        # domain to decide whether to retain its counted representation.
        count = 0
        for item in body:
            count += item.extent * call_count(item.body) if isinstance(item, Loop) else 1
            if count > 256:
                return 257
        return count

    class StaticControl(FnBuilderControlAdapter):
        """Constant-fold the exact checked tiny schedule, including its drains."""
        def loop(self, start, stop, step, callback):
            for index in range(start, stop, step):
                callback(index)

        def op(self, opcode, lhs, rhs, dtype='i64'):
            if type(lhs) is int and type(rhs) is int and opcode in ('add', 'mul'):
                value = lhs + rhs if opcode == 'add' else lhs * rhs
                if not -(1 << 63) <= value < (1 << 63):
                    raise ValueError('resident static index exceeds signed i64')
                return value
            return super().op(opcode, lhs, rhs, dtype)

        def cmp(self, predicate, lhs, rhs):
            if type(lhs) is int and type(rhs) is int and predicate == 'eq':
                return lhs == rhs
            return super().cmp(predicate, lhs, rhs)

        def select(self, condition, lhs, rhs):
            return (lhs if condition else rhs) if type(condition) is bool else super().select(condition, lhs, rhs)

    def effect(instruction, fields):
        instruction, tag = instruction.rsplit('__', 1)
        if tag == 'shared':
            pointers = dict(rhs=rhs)
            problem = problems[0]
        else:
            member = int(tag)
            product, problem = products[member], problems[member]
            pointers = dict(lhs=product['lhs'], rhs=rhs, out=product['out'],
                out_dtype=product.get('out_dtype', 'i32'),
                scale=product.get('scale', 1.0), relu=product.get('relu', False))
        previous=getattr(enc.fn,'current_source_index',None)
        if tag!='shared' and 'source_op_index' in product:
            enc.fn.current_source_index=product['source_op_index']
        enc.dense_effect(instruction, fields, pointers, problem)
        enc.fn.current_source_index=previous

    static = call_count(result.kernel.body) <= 256
    control = StaticControl(enc.fn) if static else FnBuilderControlAdapter(enc.fn)
    receipt = result.emit_index_program(control, effect,
        complete_epoch=enc.fence)
    enc.fence()
    return {**receipt, 'selected_schedule': 'source_resident_products',
        'index_emission': 'bounded_static' if static else 'counted',
        'source_sha256': source_identity, 'members': len(products),
        'schedule_plan_sha256': result.realization['schedule_sha256'],
        'shared_rhs_realization': result.realization}
