"""Bounded pure-load issue refinement after original checked dense transport.

No load, compute, configuration or completion is removed or moved. This is a
separate issue domain from existing preload/compute unroll experiments.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from merlin.sched.ir import Loop,Call,Const
from merlin.compile.scheduling.tile_load_coalescing import _substitute
from merlin.compile.scheduling.loop_schedule import LoopTransform,propose_loop_transforms
from merlin.compile.scheduling.panel_tile_loads import _map_call
from dense_counted import DenseCountedResult,_index,_replace_at,_at
from dense_transfer import _trace_equal


def unroll_dense_load_issue(original,*,factor):
    if type(factor)is not int or factor not in (2,4):raise ValueError('load issue factor must be two or four')
    original.revalidate();paths=[]
    if not hasattr(original, "emit_index_program"):
        return replace(original, realization={**original.realization, "load_issue": dict(
            selected=False, factor=factor, reason="incumbent emitter has no index-effect program; incumbent retained",
            kernel_sha256=original.kernel.digest())})
    def pure(body):
        return bool(body)and all((not x.iter_args and not x.yields and not x.results and pure(x.body))if type(x)is Loop else
            type(x)is Call and x.instr in ('lhs_load','weight_load')and not(x.produces or x.awaits or x.stages or x.sets or x.assumes)for x in body)
    def discover(body,positions=(),names=()):
        for i,item in enumerate(body):
            if type(item)is not Loop:continue
            here=(*positions,i);before=len(paths);discover(item.body,here,(*names,item.var))
            if len(paths)==before and item.extent>=factor and not(item.iter_args or item.yields or item.results)and pure(item.body):
                paths.append((here,(*names,item.var),item))
    discover(original.kernel.body)
    if not paths or len(paths)>32:
        return replace(original,realization={**original.realization,'load_issue':dict(selected=False,factor=factor,
            reason='requires one to32 bounded pure-load counted domains; incumbent retained',kernel_sha256=original.kernel.digest())})
    def mapped(body,var,quotient,multiple,offset):
        return tuple(replace(x,body=mapped(x.body,var,quotient,multiple,offset))if type(x)is Loop else
            _map_call(x,lambda e:_substitute(e,var,Const(offset))if multiple==0 else _index(e,var,quotient,multiple,offset))for x in body)
    paths.sort(reverse=True);requests=[];witnesses=[];named=original.kernel
    all_names=set()
    def collect(body):
        for x in body:
            if type(x)is Loop:all_names.add(x.var);collect(x.body)
    collect(named.body)
    for serial,(position,names,source)in enumerate(paths):
        quotient=f'dense_load_q{serial}';renamed=f'dense_load_source{serial}';inner=f'dense_load_r{serial}'
        if any(x in all_names for x in (quotient,renamed,inner)):raise ValueError('load issue fresh control name collides')
        # Exact alpha-renaming distinguishes coalesced bulk/tail siblings with
        # the same original name. The final inverse restores the original axes.
        bound=replace(source,var=renamed,body=mapped(source.body,source.var,renamed,1,0))
        named=replace(named,body=_replace_at(named.body,position,(bound,)))
        requests.append(LoopTransform('unroll',(*names[:-1],renamed),factor=factor,names=()if source.extent==factor else (quotient,inner)))
        witnesses.append((position,source,quotient,renamed))
    try:
        actual=propose_loop_transforms(named,tuple(requests),index_bits=64,max_static_operations=4096)
    except ValueError as exc:
        return replace(original, realization={**original.realization, "load_issue": dict(
            selected=False, factor=factor, reason="shared loop transform refused: "+str(exc),
            kernel_sha256=original.kernel.digest())})
    root=Path(__file__).parent
    owners={p:sha256((root/p).read_bytes()).hexdigest()for p in ('dense_issue_policy.py','dense_counted.py','dense_transfer.py','compiler_library/merlin/compile/scheduling/loop_schedule.py')}
    def check(kernel):
        if any(sha256((root/p).read_bytes()).hexdigest()!=pin for p,pin in owners.items()):raise ValueError('load issue proof owner changed')
        restored=kernel
        for position,source,quotient,renamed in reversed(witnesses):
            parent=restored.body
            for i in position[:-1]:parent=parent[i].body
            q,tail=divmod(source.extent,factor)
            if source.extent==factor:
                expected=tuple(x for offset in range(factor)for x in mapped(source.body,source.var,source.var,0,offset))
                count=len(expected)
                if not _trace_equal(parent[position[-1]:position[-1]+count],expected):raise ValueError('load issue complete bounded domain differs from original source')
            else:
                bulk=_at(restored.body,position)
                if type(bulk)is not Loop or bulk!=replace(bulk,var=quotient,extent=q,iter_args=(),yields=(),results=()):raise ValueError('load issue quotient differs from original source')
                expected=tuple(x for offset in range(factor)for x in mapped(source.body,source.var,quotient,factor,offset))
                if not _trace_equal(bulk.body,expected):raise ValueError('load issue cloned operands/effects differ from original ordered source')
                count=1
                if tail:
                    remainder=_at(restored.body,position[:-1]+(position[-1]+1,))
                    expected_tail=mapped(source.body,source.var,renamed,1,q*factor)
                    if type(remainder)is not Loop or remainder.var!=renamed or remainder.extent!=tail or remainder.iter_args or remainder.yields or remainder.results or not _trace_equal(remainder.body,expected_tail):raise ValueError('load issue remainder differs from original source')
                    count=2
            body=(*parent[:position[-1]],source,*parent[position[-1]+count:])
            if len(position)==1:restored=replace(restored,body=body)
            else:
                owner=_at(restored.body,position[:-1]);restored=replace(restored,body=_replace_at(restored.body,position[:-1],(replace(owner,body=body),)))
        if restored!=original.kernel:raise ValueError('load issue changes original effects outside checked domains')
    check(actual)
    record=dict(selected=True,factor=factor,loops=len(paths),parent_sha256=original.kernel.digest(),parent_owner=type(original).__module__+'.'+type(original).__name__,actual_sha256=actual.digest(),source_owners=owners,
        rule='bounded ordered Euclidean quotient/remainder of actual pure-load nests; exact inverse restores entire original fully selected checked product',max_static_operations=4096)
    result=DenseCountedResult(original,actual,{**original.realization,'kernel_sha256':actual.digest(),'load_issue':record},check);result.revalidate();return result
