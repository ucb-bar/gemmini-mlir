"""Source-bound raw-i32 store epochs for sampled pointwise dense schedules.

The source adapter proves actual counted calls against the original stride-two
row domain and its chosen tile_m. Existing dense_store_epochs independently
proves disjoint actual ACC spans and supplies the consumed store-only rewrite.
No primitive-row alignment or configuration snapshot is assumed.
"""
from __future__ import annotations

from merlin.common.symbolic_lock import serialized_symbolic_check
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var
from . import isa
from .dense_store_epochs import select_store_epochs


def _binding(window,problem):
    return sha256(json.dumps([asdict(window),asdict(problem)],sort_keys=True).encode()).hexdigest()


@serialized_symbolic_check
def _check_geometry(kernel, window, problem):
    import z3
    batch,height,width,channels=window.input_shape
    oh,ow=window.output_spatial
    if (window.kernel!=(1,1) or window.stride!=(2,2)
            or window.dilation!=(1,1) or any(window.padding)
            or (problem.m,problem.k)!=(batch*oh*ow,channels)
            or not 0<problem.tile_m<=isa.DIM or ow%problem.tile_m):
        raise ValueError('pointwise store epoch source/tile_m geometry changed')
    shapes={a.name:(a.shape,a.dtype) for a in kernel.args}
    if any(shapes.get(name)!=(shape,dtype) for name,shape,dtype in (
            ('lhs',(problem.m,problem.k),'i8'),('rhs',(problem.k,problem.n),'i8'),
            ('output',(problem.m,problem.n),'i32'))):
        raise ValueError('pointwise store epoch dense arguments differ from sampled source')
    queries=0
    def expression(e,env):
        if type(e)is Const:return z3.IntVal(e.value)
        if type(e)is Var:return env[e.name]
        if type(e)is Sum:return z3.Sum([expression(t,env) for t in e.terms])
        if type(e)is Scaled:return expression(e.expr,env)*e.factor
        if type(e)is Select:return z3.If(env[e.var]==e.equals,expression(e.then,env),expression(e.other,env))
        raise ValueError('pointwise store epoch unsupported counted coordinate')
    def walk(body,env,domain):
        nonlocal queries
        for node in body:
            if type(node)is Loop:
                if node.iter_args or node.yields or node.results:
                    raise ValueError('pointwise store epochs retain original token loops')
                var=z3.Int(node.var)
                walk(node.body,{**env,node.var:var},[*domain,var>=0,var<node.extent])
            elif node.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
                a={key:expression(value,env) for key,value in node.args}
                columns=problem.k if node.instr=='gemmini_mvin_lhs' else problem.n
                invalid=[a['dram_row']<0,a['rows']<=0,a['rows']>problem.tile_m,
                    a['dram_row']+a['rows']>problem.m,
                    a['dram_row']%ow+a['rows']>ow,
                    a['dram_col']<0,a['cols']<=0,a['cols']>isa.DIM,
                    a['dram_col']+a['cols']>columns]
                if node.instr=='gemmini_mvin_lhs':invalid.append(a['step']!=1)
                solver=z3.Solver();solver.set(timeout=2000)
                solver.add(*domain,z3.Or(invalid));queries+=1
                if solver.check()!=z3.unsat:
                    raise ValueError('pointwise store epoch source/output coordinate domain is unproved')
    walk(kernel.body,{},[])
    return dict(tile_m=problem.tile_m,output_width=ow,solver_queries=queries,
        input_row_stride_elements=2*channels,
        source='actual LHS and store operands; all counted loop values; sampled original NHWC rows',
        physical_source='((image*H+2*y)*W+2*x)*CI+channel',
        accumulator_geometry='actual shared store operands; no DIM row-spacing substitution')


@dataclass(frozen=True)
class PointwiseStoreEpochResult:
    selected: object
    window: object
    problem: object
    source_kernel_sha256: str
    geometry_sha256: str

    @property
    def kernel(self):return self.selected.kernel
    @property
    def command(self):return self.selected.command
    @property
    def realization(self):return self.selected.realization

    def revalidate(self):
        self.selected.revalidate()
        if _binding(self.window,self.problem)!=self.geometry_sha256:
            raise ValueError('pointwise store epoch sampled source geometry changed')
        if self.selected.original.kernel.digest()!=self.source_kernel_sha256:
            raise ValueError('pointwise store epoch original counted source changed')
        _check_geometry(self.selected.original.kernel,self.window,self.problem)

    def emit_llvm(self,builder,lower_call):
        self.revalidate()
        result=self.selected.emit_llvm(builder,lower_call)
        self.revalidate()
        return result


def select_pointwise_store_epochs(schedule,window,problem,attributes,dtype):
    """Opt-in only for the existing producer-only raw i32 composition."""
    option=attributes.get('strided_pointwise_store_epochs','off')
    if option not in ('off','on'):
        raise ValueError('strided_pointwise_store_epochs must be off or on')
    if option=='off':return schedule,None
    if dtype!='i32' or not attributes.get('strided_pointwise_raw_producer',False):
        return schedule,dict(selection='on',applied=False,refusal='requires original producer-only raw i32 output')
    schedule.revalidate()
    try:geometry=_check_geometry(schedule.kernel,window,problem)
    except ValueError as exc:
        return schedule,dict(selection='on',applied=False,refusal=str(exc))
    selected,report=select_store_epochs(schedule,{**attributes,'dense_store_epochs':'on'},problem.n,dtype)
    if not report or not report.get('applied'):
        return schedule,report
    result=PointwiseStoreEpochResult(selected,window,problem,schedule.kernel.digest(),_binding(window,problem))
    result.revalidate()
    return result,dict(report,sampled_source=geometry,
        producer_boundary='raw i32 only; bias and runtime seed producer-only; device column RNE unchanged')
