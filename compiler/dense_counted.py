"""Bounded shared partial unroll with an exact ordered-call refinement proof.

No effect is suppressed, moved across another effect, or completed differently.
The proof examines every static cloned command and every operand; its work is
bounded by the explicit factor and program size, independent of M/K/N.
"""
from dataclasses import dataclass, replace
from merlin.sched.ir import Loop, Call, Var, Const
from merlin.sched.ir.expr import Sum, Scaled, Select, add
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.panel_tile_loads import _map_call, _same_band_call


def _index(expr, source, quotient, factor, offset):
    if type(expr) is Const: return expr
    if type(expr) is Var:
        return Var(quotient)*factor+offset if expr.name == source else expr
    if type(expr) is Sum:
        return add(*(_index(x,source,quotient,factor,offset) for x in expr.terms))
    if type(expr) is Scaled:
        return _index(expr.expr,source,quotient,factor,offset)*expr.factor
    if type(expr) is Select:
        yes=_index(expr.then,source,quotient,factor,offset)
        no=_index(expr.other,source,quotient,factor,offset)
        if expr.var != source: return replace(expr,then=yes,other=no)
        eq=expr.equals-offset
        return Select(quotient,eq//factor,yes,no) if eq%factor == 0 else no
    raise ValueError('dense issue proof refuses an unknown index expression')


def _replace_at(body, path, replacement):
    out = list(body)
    index = path[0]
    if len(path) == 1:
        out[index:index+1] = replacement
    else:
        out[index] = replace(out[index], body=_replace_at(out[index].body, path[1:], replacement))
    return tuple(out)


def _at(body, path):
    for index in path:
        item = body[index]
        body = item.body
    return item


def _calls_equal(left, right):
    return len(left) == len(right) and all(type(a) is Call and type(b) is Call and
        _same_band_call(a, b) for a,b in zip(left,right))


@dataclass(frozen=True)
class DenseCountedResult:
    original: object
    kernel: object
    realization: dict
    check: object

    def revalidate(self):
        self.original.revalidate()
        self.check(self.kernel)

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        result = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return result


def unroll_dense_issue(original, *, factor=4):
    """Unroll counted innermost preload/compute pairs, preserving all effects."""
    if factor not in (2,4):
        raise ValueError('dense issue factor must be bounded two or four')
    original.revalidate()
    paths = []
    def discover(body, positions=(), names=()):
        for index,item in enumerate(body):
            if type(item) is Loop:
                here = (*positions,index)
                if (item.extent > factor and not item.iter_args and not item.yields and not item.results
                    and tuple(x.instr if type(x) is Call else None for x in item.body) == ('preload','compute')
                    and all(not x.produces and not x.awaits and not x.stages and not x.sets and not x.assumes
                            for x in item.body)):
                    paths.append((here,(*names,item.var),item))
                else:
                    discover(item.body,here,(*names,item.var))
    discover(original.kernel.body)
    if not paths or len(paths)>32:
        raise ValueError('dense counted issue needs one to 32 plain preload/compute loops')
    # Reverse sibling order makes structural indices independent of inserted tails.
    paths.sort(reverse=True)
    requests=[]
    witnesses=[]
    for serial,(position,names,source) in enumerate(paths):
        outer=f'dense_issue_q{serial}'
        inner=f'dense_issue_r{serial}'
        requests.append(LoopTransform('unroll',names,factor=factor,names=(outer,inner)))
        witnesses.append((position,source,outer))
    proposal = propose_loop_transforms(original.kernel,tuple(requests),index_bits=64,
                                      max_static_operations=4096)

    def check(kernel):
        actual=kernel
        # Undo in opposite order, checking each concrete cloned operand first.
        for position,source,outer in reversed(witnesses):
            q,tail=divmod(source.extent,factor)
            bulk=_at(actual.body,position)
            if (type(bulk) is not Loop or bulk != replace(bulk,var=outer,extent=q,
                    iter_args=(),yields=(),results=())):
                raise ValueError('dense issue quotient/control differs from exact source domain')
            expected=tuple(_map_call(call,lambda expr,r=r: _index(expr,source.var,outer,factor,r)) for r in range(factor) for call in source.body)
            if not _calls_equal(bulk.body,expected):
                raise ValueError('dense issue cloned calls differ from ordered source refinement')
            count=1
            if tail:
                sibling=position[:-1]+(position[-1]+1,)
                remainder=_at(actual.body,sibling)
                expected_tail=tuple(_map_call(call,lambda expr: _index(expr,source.var,source.var,1,q*factor)) for call in source.body)
                if (type(remainder) is not Loop or remainder.var != source.var or remainder.extent != tail
                    or remainder.iter_args or remainder.yields or remainder.results
                    or not _calls_equal(remainder.body,expected_tail)):
                    raise ValueError('dense issue ordered remainder differs from source domain')
                count=2
            # Replace checked quotient+tail by original counted loop. This leaves
            # all surrounding calls, stages and completion boundaries untouched.
            parent=actual.body
            for index in position[:-1]: parent=parent[index].body
            restored=(*parent[:position[-1]],source,*parent[position[-1]+count:])
            if len(position)==1:actual=replace(actual,body=restored)
            else:
                owner=_at(actual.body,position[:-1])
                actual=replace(actual,body=_replace_at(actual.body,position[:-1],
                    (replace(owner,body=restored),)))
        if actual != original.kernel:
            raise ValueError('dense issue changes effects outside the proved source loops')

    check(proposal)
    result=DenseCountedResult(original,proposal,dict(original.realization,
        kernel_sha256=proposal.digest(),counted_issue=dict(factor=factor,
            parent_sha256=original.kernel.digest(),loops=len(paths),
            rule='ordered Euclidean quotient/remainder; exact operand and completion trace')),check)
    result.revalidate()
    return result
