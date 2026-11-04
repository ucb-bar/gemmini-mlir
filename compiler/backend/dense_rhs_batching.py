"""Optional ordered batching of primitive RHS transfers before terminal emission.

This changes host counted loops only. Every original MVIN, operand, completion
and ordering annotation remains exact; no wide-load or concurrent-copy law is
assumed. Shared loop proposals are independently checked as ordered refinements.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.sched.ir import Call, Loop, Const, Var, Sum, Scaled, Select, add, mul, select, check_structure

MAX_STATIC=512


def _sub(e,var,factor,offset,newvar=None):
    if type(e) is Const:return e
    if type(e) is Var:return (Const(offset) if newvar is None else Var(newvar)*factor+offset) if e.name==var else e
    if type(e) is Sum:return add(*(_sub(t,var,factor,offset,newvar) for t in e.terms))
    if type(e) is Scaled:return mul(_sub(e.expr,var,factor,offset,newvar),e.factor)
    if type(e) is Select:
        yes=_sub(e.then,var,factor,offset,newvar);no=_sub(e.other,var,factor,offset,newvar)
        if e.var!=var:return select(e.var,e.equals,yes,no)
        if newvar is None:return yes if offset==e.equals else no
        q,r=divmod(e.equals-offset,factor)
        return no if r else select(newvar,q,yes,no)
    raise ValueError('RHS batching refuses unknown operand expression')


def _body_sub(body,var,factor,offset,newvar=None):
    out=[]
    for node in body:
        if type(node) is Loop:
            if node.var==var:raise ValueError('RHS batching refuses shadowed induction')
            out.append(replace(node,body=_body_sub(node.body,var,factor,offset,newvar)))
        elif type(node) is Call:
            out.append(replace(node,args=tuple((k,_sub(v,var,factor,offset,newvar)) for k,v in node.args)))
        else:raise ValueError('RHS batching refuses unknown statement')
    return tuple(out)


def _plain_rhs(body):
    return bool(body) and all((_plain_rhs(x.body) if type(x) is Loop else
        type(x) is Call and x.instr in ('gemmini_mvin_weight','gemmini_mvin_rhs')
        and not (x.unit or x.produces or x.awaits or x.stages or x.sets or x.assumes)) for x in body)


def _size(body):return sum(1+(_size(x.body) if type(x) is Loop else 0) for x in body)


def _replace(body,path,replacement):
    result=[];found=0
    for node in body:
        if type(node) is Loop and node.var==path[0]:
            found+=1
            result.extend(replacement if len(path)==1 else (replace(node,body=_replace(node.body,path[1:],replacement)),))
        else:result.append(node)
    if found!=1:raise ValueError('RHS batching path is not unique')
    return tuple(result)


def _candidates(body,path=(),multiplicity=1):
    out=[]
    for node in body:
        if type(node) is not Loop:continue
        nested=_candidates(node.body,(*path,node.var),multiplicity*node.extent)
        if nested:out.extend(nested)
        elif node.extent>=2 and _plain_rhs(node.body):out.append(((*path,node.var),node,multiplicity))
    return out


@dataclass(frozen=True)
class DenseRhsBatchResult:
    original:object
    kernel:object
    transforms:tuple
    realization:dict
    checked_identity:str
    implementation_identity:str

    @property
    def command(self):
        # Store-epoch BEGIN is emitted by the unchanged terminal callback.
        return self.original.command

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest()!=self.checked_identity or
            sha256(Path(__file__).read_bytes()).hexdigest()!=self.implementation_identity or
            propose_loop_transforms(self.original.kernel,self.transforms,max_static_operations=MAX_STATIC)!=self.kernel):
            raise ValueError('RHS batching checked source, proposal or implementation changed')

    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)
        self.revalidate();return result

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate();return materialize_scf_commands(self.kernel)


def select_rhs_batching(schedule,*,factor=4):
    """Batch deepest pure RHS transfer loops using factors 2,4,8; keep tails."""
    if factor not in (2,4,8):raise ValueError('RHS batch factor must be 2, 4 or 8')
    schedule.revalidate();actual=schedule.kernel;transforms=[];changes=[]
    for index,(path,loop,multiplicity) in enumerate(_candidates(actual.body)):
        if len(transforms)>=32:break
        width=min(factor,loop.extent);full,tail=divmod(loop.extent,width)
        name=f'dense_rhs_batch_{index}'
        # Construct an independent exact ordered refinement. Select predicates
        # are substituted by integer quotient/residue, including ragged tails.
        if loop.extent==width:
            replacement=tuple(x for j in range(width) for x in _body_sub(loop.body,loop.var,1,j))
            request=LoopTransform('unroll',path,factor=width)
        else:
            replacement=(Loop(name,full,tuple(x for j in range(width) for x in _body_sub(loop.body,loop.var,width,j,name))),)
            if tail:replacement+=(Loop(loop.var,tail,_body_sub(loop.body,loop.var,1,full*width,loop.var)),)
            request=LoopTransform('unroll',path,factor=width,names=(name,name+'_within'))
        expected=replace(actual,body=_replace(actual.body,path,replacement))
        if _size(expected.body)>MAX_STATIC:continue
        proposal=propose_loop_transforms(actual,(request,),max_static_operations=MAX_STATIC)
        if proposal!=expected:raise ValueError('shared RHS batching proposal differs from exact ordered refinement')
        if check_structure(proposal):raise ValueError('invalid RHS batching result')
        transforms.append(request);actual=proposal
        changes.append(dict(path=list(path),original_trips=loop.extent,factor=width,tail_trips=tail,
            dynamic_loop_iterations_before=multiplicity*loop.extent,
            dynamic_loop_iterations_after=multiplicity*(0 if loop.extent==width else full+tail)))
    receipt=dict(applied=bool(changes),factor=factor,changes=changes,
        original_kernel_sha256=schedule.kernel.digest(),kernel_sha256=actual.digest(),
        primitive_commands='identical ordered MVINs and drains; no wide transfer',
        proof='independent exact symbolic ordered refinement of shared split/unroll',
        max_static_nodes=MAX_STATIC)
    if not changes:return schedule,receipt
    result=DenseRhsBatchResult(schedule,actual,tuple(transforms),dict(schedule.realization,kernel_sha256=actual.digest(),rhs_batching=receipt),
        actual.digest(),sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate();return result,receipt
