"""Solver-free structural induction for one primitive tile per dense panel.

This checker does not call a scheduler or regenerate target instructions. It
recognizes the projected actual program's restricted loop/control grammar and
checks every source/physical-address equation against the source contraction.

Induction: each M/N iteration owns one disjoint output rectangle. Its K loop
loads the two current source rectangles into disjoint assigned windows, then
immediately consumes them with a matching preload/compute pair. K=0 overwrites
(or consumes the explicit bias seed); successive K values continue precisely
the same accumulator. The sole following store observes complete K. Tile-loop
bounds and the last-tile extents partition all three original source axes.
Other loop/issue forms are unavailable, never assumed equivalent.
"""
from __future__ import annotations

from dataclasses import fields
import time

from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure
from merlin.transforms.dense_planning import DenseBlock
from .block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from .compact_dense import DenseBiasSource
from .compact_verification import DenseVerificationUnavailable, _project, _requested_elements


def verify_compact_rectangular(kernel, problem, resources, storage, adapter, *,
        static_budget, timeout_ms=2000, output_readout=None, bias_source=None):
    """Check actual primitive compact source/state semantics without SMT.

    Noncanonical schedules/readouts decline with DenseVerificationUnavailable.
    A recognized canonical schedule with wrong operand/state equations fails
    with ValueError. No dynamic tile/product expansion or larger budget occurs.
    """
    if (type(static_budget) is not int or static_budget <= 0
            or type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError('rectangular source proof needs original positive checking budgets')
    deadline=time.monotonic()+timeout_ms/1000
    def budget():
        if time.monotonic()>deadline:
            raise DenseVerificationUnavailable('rectangular source verification budget exhausted')
    if output_readout is not None:
        raise DenseVerificationUnavailable('rectangular source induction supports raw accumulator readout only')
    if (not adapter.synchronous_completion or not adapter.exact_accumulator_continuation
            or storage.accumulator is None):
        raise ValueError('rectangular source induction lacks exact synchronous accumulator semantics')
    if check_structure(kernel):
        raise ValueError('invalid actual rectangular dense kernel structure')
    if problem.bias != (bias_source is not None):
        raise ValueError('rectangular bias presence differs from original source')
    if bias_source is not None:
        if type(bias_source) is not DenseBiasSource:
            raise ValueError('rectangular bias requires its exact typed source')
        bias_source.validate(problem,(kernel.args[0].dtype,kernel.args[1].dtype,kernel.args[2].dtype))
    expected_args=[('lhs',(problem.m,problem.k),problem.lhs_bytes,'read'),
                   ('rhs',(problem.k,problem.n),problem.rhs_bytes,'read')]
    if bias_source is not None:
        expected_args.append(('bias',(problem.n,),problem.accumulator_bytes,'read'))
    expected_args.append(('output',(problem.m,problem.n),problem.accumulator_bytes,'write'))
    if (kernel.attrs or len(kernel.args)!=len(expected_args)
            or any((arg.name,arg.shape,DTYPE_BYTES.get(arg.dtype),arg.access)!=expected
                or type(arg.shape) is not tuple
                or any(type(dimension) is not int or dimension <= 0 for dimension in arg.shape)
                or arg.dtype not in ('i8','i16','i32','i64')
                for arg,expected in zip(kernel.args,expected_args))):
        raise ValueError('actual rectangular boundary differs from complete integer contraction')
    if not resources.partial_output_tiles and (problem.m%problem.tile_m or problem.n%problem.tile_n):
        raise ValueError('rectangular source tails exceed selected resource contract')
    mt,kt,nt=problem.tiles
    # A source-derived primitive footprint, independent of the emitter's block
    # choice. This checks assigned byte extents, capacity, backing aliasing,
    # strides/alignment, and total live operand/accumulator resources.
    primitive=DenseBlock(1,1,1,
        problem.tile_m*problem.tile_k*problem.lhs_bytes+problem.tile_k*problem.tile_n*problem.rhs_bytes,
        problem.tile_m*problem.tile_n*problem.accumulator_bytes,
        mt*kt*nt,mt*kt*nt,mt*nt if bias_source else 0,mt*nt,
        nt*problem.m*problem.k*problem.lhs_bytes+mt*problem.k*problem.n*problem.rhs_bytes)
    if not storage.admits(problem,primitive,resources):
        raise ValueError('primitive rectangular windows exceed assigned resource ownership')
    if min(resources.max_m_blocks,resources.max_k_blocks,resources.max_n_blocks)<1:
        raise ValueError('primitive rectangular block exceeds selected resource bounds')
    # Expressions normalize to an integer affine combination of variables and
    # explicit equality-select atoms. No evaluation at sampled loop points is
    # used. Equality of these normal forms implies equality at every iteration.
    serial=0
    nodes=0
    def linear(constant=0, terms=()):
        merged={}
        for atom,coefficient in terms:
            merged[atom]=merged.get(atom,0)+coefficient
        return (constant,tuple(sorted(((a,c) for a,c in merged.items() if c),key=lambda p:repr(p[0]))))
    def add(values):
        return linear(sum(v[0] for v in values),tuple(p for v in values for p in v[1]))
    def expression(value,env):
        budget()
        if type(value) is int:return linear(value)
        if type(value) is Const:
            if type(value.value) is not int:
                raise ValueError('rectangular index constants require exact integers')
            return linear(value.value)
        if type(value) is Var:
            if value.name not in env:raise ValueError('unbound actual rectangular source index')
            return env[value.name]
        if type(value) is Sum:return add([expression(t,env) for t in value.terms])
        if type(value) is Scaled:
            if type(value.factor) is not int:
                raise ValueError('rectangular index factors require exact integers')
            e=expression(value.expr,env)
            return linear(e[0]*value.factor,((a,c*value.factor) for a,c in e[1]))
        if type(value) is Select:
            if type(value.equals) is not int:
                raise ValueError('rectangular index comparisons require exact integers')
            condition=expression(Var(value.var),env)
            # Validate both actual branches even if a singleton loop folds the
            # condition. Direct dataclass construction need not use as_expr.
            then,other=expression(value.then,env),expression(value.other,env)
            if not condition[1]:
                return then if condition[0]==value.equals else other
            if then==other:return then
            return linear(0,((('select',condition,value.equals,then,other),1),))
        raise ValueError('unsupported actual rectangular arithmetic expression')
    def value(v,env):
        if isinstance(v,(Const,Var,Sum,Scaled,Select)) or type(v) is int:
            return ('integer',expression(v,env))
        if type(v) is tuple:return ('tuple',tuple(value(x,env) for x in v))
        if v is None or type(v) in (str,bool):return (type(v).__name__,v)
        raise ValueError('unmodeled actual rectangular operand')
    def operation(op,env):
        if type(op) not in (Load,Preload,Compute,Store,InitializeAccumulator):
            raise DenseVerificationUnavailable('rectangular source induction does not model this actual operation')
        return (type(op).__name__,tuple((f.name,value(getattr(op,f.name),env)) for f in fields(op)))
    def normalize(body,env,actual):
        nonlocal serial,nodes
        normalized=[]
        for node in body:
            nodes+=1;budget()
            if nodes>static_budget*16:
                raise DenseVerificationUnavailable('rectangular static structure exceeds original checking budget')
            if type(node) is Loop:
                if type(node.extent) is not int or node.extent <= 0:
                    raise ValueError('rectangular loop bounds require exact positive integers')
                if node.iter_args or node.yields or node.results:
                    raise ValueError('rectangular verification cannot erase loop completion-token effects')
                if node.extent==1:
                    normalized.extend(normalize(node.body,{**env,node.var:linear(0)},actual))
                else:
                    token=serial;serial+=1
                    nested=normalize(node.body,{**env,node.var:linear(0,((('var',token),1),))},actual)
                    normalized.append(('loop',node.extent,nested))
            else:
                if actual:
                    if (type(node) is not Call or node.produces is not None or node.awaits
                            or node.unit or node.stages or node.sets or node.assumes):
                        raise ValueError('rectangular verification cannot erase actual configuration or completion effects')
                    node=adapter.project(node)
                normalized.append(('operation',operation(node,env)))
        return tuple(normalized)
    actual=normalize(kernel.body,{},True)
    # The following is a source/state proof schema, not an emitted kernel.
    # Only projected arithmetic effects are specified; no target calls, encoder
    # or scheduling generator participates in constructing the obligations.
    m,k,n=Var('m'),Var('k'),Var('n')
    def edge(axis,tiles,extent,tile):
        return Const(tile) if extent%tile==0 else Select(axis,tiles-1,Const(extent%tile),Const(tile))
    mr=edge('m',mt,problem.m,problem.tile_m)
    kd=edge('k',kt,problem.k,problem.tile_k)
    nc=edge('n',nt,problem.n,problem.tile_n)
    lhs,rhs,acc=storage.lhs.base,storage.rhs.base,storage.accumulator.base
    reduction=(Load('lhs',(m,k),m*problem.tile_m,k*problem.tile_k,mr,kd,Const(lhs),1),
        Load('weight',(k,n),k*problem.tile_k,n*problem.tile_n,kd,nc,Const(rhs),1),
        Preload(Const(rhs),(k,n),Const(acc),Const(1) if bias_source else Select('k',0,Const(0),Const(1)),mr,nc,kd),
        Compute(Const(lhs),(m,k),mr,kd,True))
    seed=() if bias_source is None else (InitializeAccumulator((m,n),n*problem.tile_n,mr,nc,Const(acc)),)
    schema=(Loop('m',mt,(Loop('n',nt,(*seed,Loop('k',kt,reduction),
        Store(m*problem.tile_m,n*problem.tile_n,mr,nc,Const(acc)))),)),)
    serial=0;nodes=0
    required=normalize(schema,{},False)
    def skeleton(body):
        return tuple(('loop',node[1],skeleton(node[2])) if node[0]=='loop' else ('operation',node[1][0]) for node in body)
    if skeleton(actual)!=skeleton(required):
        raise DenseVerificationUnavailable('actual schedule is outside primitive rectangular source-induction normal form')
    if actual!=required:
        raise ValueError('actual canonical rectangular source/address/tail/initialization/continuation equations differ')
    # The complete source-domain partition proves all dynamic cardinalities and
    # traffic algebraically. The helper below independently counts actual
    # projected tail expressions, without enumerating loop instances/products.
    sites=_project(kernel,adapter)
    if len(sites)>static_budget:
        raise ValueError('actual rectangular instructions exceed original static budget')
    budget()
    loads=[s for s in sites if type(s.operation) is Load]
    initializers=[s for s in sites if type(s.operation) is InitializeAccumulator]
    stores=[s for s in sites if type(s.operation) is Store]
    lhs_bytes=sum(_requested_elements(s,s.operation.rows,s.operation.cols)*problem.lhs_bytes for s in loads if s.operation.role=='lhs')
    rhs_bytes=sum(_requested_elements(s,s.operation.rows,s.operation.cols)*problem.rhs_bytes for s in loads if s.operation.role=='weight')
    bias_bytes=sum(_requested_elements(s,s.operation.cols)*problem.accumulator_bytes for s in initializers)
    store_bytes=sum(_requested_elements(s,s.operation.rows,s.operation.cols)*problem.accumulator_bytes for s in stores)
    if (lhs_bytes!=nt*problem.m*problem.k*problem.lhs_bytes
            or rhs_bytes!=mt*problem.k*problem.n*problem.rhs_bytes
            or bias_bytes!=(mt*problem.n*problem.accumulator_bytes if bias_source else 0)
            or store_bytes!=problem.m*problem.n*problem.accumulator_bytes):
        raise ValueError('actual rectangular traffic differs from complete source partition')
    high_water={storage.lhs.name:min(problem.tile_m,problem.m)*storage.lhs.unit_bytes,
        storage.rhs.name:min(problem.tile_k,problem.k)*storage.rhs.unit_bytes,
        storage.accumulator.name:min(problem.tile_m,problem.m)*storage.accumulator.unit_bytes}
    maximum=max(problem.m*problem.k*problem.lhs_bytes,
        problem.k*problem.n*problem.rhs_bytes,problem.m*problem.n*problem.accumulator_bytes,
        *((view.base+view.capacity)*view.unit_bytes for view in (storage.lhs,storage.rhs,storage.accumulator)))
    if maximum >= 1 << (adapter.index_bits-1):
        raise ValueError('actual rectangular source/index bounds exceed selected signed index arithmetic')
    budget()
    return dict(schema='merlin.compact_dense_realization.v1',kernel_sha256=kernel.digest(),
        configuration_identity=adapter.configuration_identity,numerical_policy_identity=adapter.numerical_policy_identity,
        c_index_bits=adapter.index_bits,static_instruction_calls=len(sites),
        executed_instruction_calls=sum(s.instances for s in sites),
        lhs_loads=sum(s.instances for s in loads if s.operation.role=='lhs'),
        rhs_loads=sum(s.instances for s in loads if s.operation.role=='weight'),
        bias_initializations=sum(s.instances for s in initializers),
        bias_source=None if bias_source is None else dict(columns=bias_source.columns,dtype=bias_source.dtype,
            arithmetic=bias_source.arithmetic,row_stride=0,column_stride=1),
        requested_bias_load_bytes=bias_bytes,requested_lhs_bytes=lhs_bytes,requested_rhs_bytes=rhs_bytes,
        requested_load_bytes=lhs_bytes+rhs_bytes+bias_bytes,requested_store_bytes=store_bytes,
        padded_high_water_bytes=high_water,output_tiles=sum(s.instances for s in stores),solver_queries=0,
        proof_kind='primitive_rectangular_structural_induction',completion='declared synchronous',
        machine_emission='not_established',timing='UNKNOWN')
