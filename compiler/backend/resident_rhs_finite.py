"""Bounded exact integer proof for resident operand address rewrites.

Every quantified point belongs to an actual counted-loop domain. Unsupported
control flow or budgets refuse before the caller emits any target effects.
"""
from itertools import product
from math import prod
from merlin.sched.ir import Call, Loop, evaluate

MAX_COMMANDS = 1_000_000
MAX_ADDRESS_CHECKS = 1_000_000

class Predicates:
    @staticmethod
    def And(*conditions): return all(conditions)
    @staticmethod
    def If(condition, yes, no): return yes if condition else no

class FiniteProof:
    def __init__(self, kernel):
        self.points = 0
        def count(body, names=()):
            result = 0
            for item in body:
                if isinstance(item, Loop):
                    if (type(item.extent) is not int or item.extent <= 0
                            or item.var in names or item.iter_args or item.yields or item.results):
                        raise ValueError('unsupported resident finite loop domain')
                    result += item.extent * count(item.body, (*names, item.var))
                elif isinstance(item, Call): result += 1
                else: raise ValueError('unsupported resident finite command')
                if result > MAX_COMMANDS:
                    raise ValueError('resident finite command proof budget exceeded')
            return result
        self.commands = count(kernel.body)
        if not self.commands: raise ValueError('empty resident finite program')
    def prove(self, scopes, condition):
        cost = prod(extent for _, extent in scopes)
        if self.points + cost > MAX_ADDRESS_CHECKS:
            raise ValueError('resident finite address proof budget exceeded')
        self.points += cost
        for values in product(*(range(extent) for _, extent in scopes)):
            env = dict(zip((name for name, _ in scopes), values))
            if not condition(env):
                raise ValueError('unproved resident finite source/address relation')
    def record(self):
        return {'method':'bounded_exhaustive_actual_loop_domains',
                'actual_commands':self.commands, 'address_points':self.points,
                'command_budget':MAX_COMMANDS, 'address_budget':MAX_ADDRESS_CHECKS,
                'optional_solver_required':False}

expr = evaluate

def verify_member(schedule, plan):
    """Reprove the current source kernel, even after a refreshed custody digest."""
    from merlin.compile.scheduling.compact_dense import DenseMemoryView, DenseLoopStorage, DenseBiasSource
    from merlin.transforms.dense_planning import DenseProblem, DenseResources
    from merlin.compile.scheduling.dense_finite_certificate import verify_finite_dense
    from .dense import _adapter
    from . import isa
    d = isa.DIM
    # Only the established one-tile streamed member adapter is consumed here.
    # Bias/initializer lowering shares the member's explicit i32 seed operation.
    def seeded_body(body):
        return any(seeded_body(item.body) if isinstance(item, Loop)
                   else item.instr == 'gemmini_bias_mvin' for item in body)
    seeded = seeded_body(schedule.kernel.body)
    problem = DenseProblem(plan.m, plan.k, plan.n, d, d, d, 1, 1, 4, bias=seeded)
    storage = DenseLoopStorage(DenseMemoryView('lhs','spad',d,0,d),
        DenseMemoryView('rhs','spad',d,isa.SPAD_ROWS//2,d),
        DenseMemoryView('acc','acc',4*d,0,d))
    resources = DenseResources(2*d*d,4*d*d,max_m_blocks=1,max_n_blocks=1,
        max_k_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    checked = verify_finite_dense(schedule.kernel,problem,resources,storage,_adapter(),
        static_budget=128,timeout_ms=2000,output_readout=None,
        bias_source=DenseBiasSource(plan.n,'i32') if seeded else None,mode='serial')
    if checked is None:
        raise ValueError('resident source exceeds finite member proof domain')
    return checked
