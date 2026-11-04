"""Dependency-free proof of the primitive, synchronous dense loop family.

This is a deliberately narrower checker than the shared SMT checker. It proves
an exact loop schema by structural induction; it admits no panel reuse, overlap,
reordered reductions, output transformations, or unmodelled instruction effects.
"""
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from merlin.sched.ir import Call, Const, Var, Loop, Kernel, TensorArg, select
from merlin.sched.ir.kernel import check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.transforms.dense_planning import choose_dense_block
from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import _project, _requested_elements
from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store, InitializeAccumulator


def _reference(p, s):
    """Independent source partition and sequential-reduction specification.

    The half-open intervals [i*T,min((i+1)*T,D)) are disjoint and
    exhaustive for 0<=i<ceil(D/T). The innermost loop visits K intervals
    in increasing order. Each iteration loads both operands, preloads the
    matching weight, and computes; only iteration zero clears the accumulator.
    Bias instead initializes once and every product continues that seed.
    The sole store follows the entire reduction, before any slot is reused.
    """
    mt, kt, nt = p.tiles
    mi, ki, nj = Var('m'), Var('k'), Var('n')
    def edge(var, count, tile):
        return select(var, (count-1)//tile, count % tile, tile) if count % tile else Const(tile)
    rows, depth, cols = edge('m',p.m,p.tile_m),edge('k',p.k,p.tile_k),edge('n',p.n,p.tile_n)
    operations=[]
    def call(op):
        operations.append(op)
        return Call(str(len(operations)-1), ())
    before=()
    if p.bias:
        before=(call(InitializeAccumulator((mi,nj),nj*p.tile_n,rows,cols,s.accumulator.base,0,1)),)
    reduction=(
        call(Load('lhs',(mi,ki),mi*p.tile_m,ki*p.tile_k,rows,depth,s.lhs.base,1)),
        call(Load('weight',(ki,nj),ki*p.tile_k,nj*p.tile_n,depth,cols,s.rhs.base,1)),
        call(Preload(s.rhs.base,(ki,nj),s.accumulator.base,Const(1) if p.bias else select('k',0,0,1),rows,cols,depth)),
        call(Compute(s.lhs.base,(mi,ki),rows,depth,True)),
    )
    after=call(Store(mi*p.tile_m,nj*p.tile_n,rows,cols,s.accumulator.base))
    args=(TensorArg('lhs',(p.m,p.k),'i8','read'),TensorArg('rhs',(p.k,p.n),'i8','read'),
          *((TensorArg('bias',(p.n,),'i32','read'),) if p.bias else ()),
          TensorArg('output',(p.m,p.n),'i32','write'))
    kernel=Kernel('reference',args,(Loop('m',mt,(Loop('n',nt,(*before,Loop('k',kt,reduction),after)),)),))
    return kernel,SimpleNamespace(project=lambda c:operations[int(c.instr)])


def verify_primitive_dense(kernel,p,r,s,adapter,choice):
    if (p.tile_m,p.tile_k,p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)!=(16,16,1,1,4) or p.tile_n not in (1,16):
        raise ValueError('structural dense proof requires the exact DIM16 i8/i8/i32 primitive geometry')
    if (choice.block.m,choice.block.k,choice.block.n)!=(1,1,1):
        raise ValueError('structural dense proof requires one primitive tile per slot')
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation or adapter.index_bits!=64:
        raise ValueError('structural dense proof requires exact synchronous i64 adapter')
    if adapter.configuration_identity!='gemmini-fire-aligned-dim16-ws' or adapter.numerical_policy_identity!='i8-i8-i32-modular':
        raise ValueError('structural dense proof has an unsupported target contract')
    if not s.admits(p,choice.block,r):
        raise ValueError('primitive dense physical windows overlap or exceed assigned capacity')
    errors=check_structure(kernel)
    if errors: raise ValueError('invalid primitive dense structure: '+str(errors))
    names=set()
    def loops(body):
        for stmt in body:
            if isinstance(stmt,Loop):
                if stmt.iter_args or stmt.yields or stmt.results or not stmt.body or stmt.var in names:
                    raise ValueError('unsupported primitive dense loop effects or reused binder')
                names.add(stmt.var);loops(stmt.body)
            elif type(stmt)is not Call or adapter.encode(adapter.project(stmt))!=stmt:
                raise ValueError('primitive dense actual call has unprojected fields')
    loops(kernel.body)
    reference,semantic=_reference(p,s)
    if kernel.args!=reference.args or kernel.attrs:
        raise ValueError('primitive dense tensor/source contract changed')
    actual=_project(kernel,adapter)
    expected=_project(reference,semantic)
    # Normalize binder names, retaining loop identity, bounds, nesting and order.
    # Exact per-site scope equality rejects splitting/interleaving reductions.
    def canonical(sites):
        binders={}
        def value(v,rename):
            if type(v) is Var:return ('var',rename[v.name])
            from merlin.sched.ir import Select
            if type(v) is Select:return ('select',rename[v.var],v.equals,value(v.then,rename),value(v.other,rename))
            if type(v) is int:return ('integer',v)
            if type(v) is Const:return ('integer',v.value)
            if type(v) is tuple:return tuple(value(x,rename)for x in v)
            if is_dataclass(v):return (type(v).__name__,tuple((f.name,value(getattr(v,f.name),rename))for f in fields(v)))
            return (type(v).__name__,v)
        result=[]
        for site in sites:
            scope=[];rename={}
            for serial,name,extent in site.scopes:
                identity=binders.setdefault(serial,len(binders));scope.append((identity,extent));rename[name]=identity
            result.append((tuple(scope),value(site.operation,rename)))
        return tuple(result)
    if canonical(actual)!=canonical(expected):
        raise ValueError('actual primitive dense operands/order do not match the source induction schema')
    check_integer_domain(kernel,64)
    counts={name:0 for name in ('lhs','weight','bias','output')}
    for site in actual:
        op=site.operation
        role=op.role if type(op)is Load else 'bias' if type(op)is InitializeAccumulator else 'output' if type(op)is Store else None
        if role:counts[role]+=site.instances
    # The target uses padded physical rows even for one-column readout.
    # DenseEmission's packed-byte high-water convention does not describe
    # those rows; physical capacity/disjointness is checked by storage.admits.
    for role,field in [('lhs','lhs_loads'),('weight','rhs_loads'),('bias','bias_loads'),('output','output_tiles')]:
        if counts[role]!=getattr(choice.block,field):
            raise ValueError('actual primitive dense transfer multiplicity changed')
    if len(actual)>min(128,choice.static_commands):
        raise ValueError('actual primitive dense static budget exceeded')
    stores=[x for x in actual if type(x.operation)is Store]
    if sum(_requested_elements(x,x.operation.rows,x.operation.cols)for x in stores)!=p.m*p.n:
        raise ValueError('primitive dense exact output cover changed')
    return {'verification':'primitive-loop structural induction (no SMT)',
            'kernel_sha256':kernel.digest(),'source_shape':[p.m,p.k,p.n],
            'block':[1,1,1],'static_commands':len(actual),'lhs_loads':counts['lhs'],
            'rhs_loads':counts['weight'],'bias_loads':counts['bias'],'output_tiles':counts['output'],
            'completion':'full terminal drains before primitive operand/accumulator reuse',
            'proof_obligations':['exact source partition and tails','ordered complete K reduction',
                'zero or explicit modular seed','actual projected operands and loop scopes',
                'disjoint assigned physical windows','one store per output tile','signed i64 intermediates']}


@dataclass(frozen=True)
class PrimitiveDenseResult:
    kernel: object
    choice: object
    problem: object
    resources: object
    storage: object
    adapter: object
    index_bits: int=64
    @property
    def realization(self):
        return verify_primitive_dense(self.kernel,self.problem,self.resources,self.storage,self.adapter,self.choice)
    def revalidate(self):
        self.realization
    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        return emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)


def schedule_primitive_dense(problem,resources,storage,adapter,*,bias_source=None):
    choice=choose_dense_block(problem,resources,static_budget=128,require_reuse=False,
        price=lambda block:5+int(problem.bias),admits=lambda block:storage.admits(problem,block,resources))
    if choice is None:raise ValueError('no legal primitive dense storage assignment')
    kernel=emit_compact_dense(problem,choice.block,storage,adapter,name='dense',dtypes=('i8','i8','i32'),bias_source=bias_source)
    result=PrimitiveDenseResult(kernel,choice,problem,resources,storage,adapter)
    result.revalidate()
    return result
