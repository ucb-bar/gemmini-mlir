"""Prove reused operand writes are ordered through actual intervening reads.

No load-WAW completion law is selected.  For EVERY overlapping pair of load
instances in one merged K stream, each overwritten cell must be covered by an
actual read issued between them. Selected local RAW followed by WAR then orders
the writes. Counted domains stay symbolic; no unrolling or elapsed-delay proof.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, replace
from time import monotonic

from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var
from .block_schedule import Compute, Load, Preload
from .compact_dense import _digest
from .compact_verification import _defined_exists, _project
from .dense_panel_pipeline import _serial_projection
from .dense_completion import CompletionRefusal

_PROOFS = OrderedDict()


def verify_load_reuse_order(kernel, groups, storage, adapter, *, block, timeout_ms=30000):
    import z3
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise CompletionRefusal('completion: invalid local-reuse checking budget')
    deadline = monotonic()+timeout_ms/1000
    checked, queries = 0, 0
    group_pins = []

    def x(value, env):
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(x(t,env) for t in value.terms))
        if type(value) is Scaled:
            return x(value.expr,env)*value.factor
        if type(value) is Select:
            return z3.If(env[value.var]==value.equals,x(value.then,env),x(value.other,env))
        raise CompletionRefusal('completion: unknown expression in actual load reuse')

    def domain(site, tag):
        env = {name:z3.Int(f'{tag}:{scope}:{name}') for scope,name,_ in site.scopes}
        return env, [z3.And(env[name]>=0,env[name]<extent) for _,name,extent in site.scopes]

    def before(a, ae, b, be):
        left=tuple(z3.IntVal(v) if type(v) is int else ae[v[1]] for v in a.path)
        right=tuple(z3.IntVal(v) if type(v) is int else be[v[1]] for v in b.path)
        return z3.Or(*(z3.And(*(left[j]==right[j] for j in range(i)),left[i]<right[i])
                       for i in range(min(len(left),len(right)))))

    def unsat(claim):
        nonlocal queries
        remaining=int((deadline-monotonic())*1000)
        if remaining<=0:
            raise CompletionRefusal('completion: operand reuse checking budget exhausted')
        solver=z3.Solver();solver.set(timeout=remaining);solver.add(claim);queries+=1
        return solver.check()==z3.unsat

    for gi,group in enumerate(groups):
        body=kernel.body; ancestors=[]
        for index in group['path'][:-1]:
            parent=body[index]
            if type(parent) is not Loop:
                raise ValueError('completion: scope path no longer names original counted loops')
            ancestors.append(parent);body=parent.body
        start=group['path'][-1]
        selected=body[start:start+group['siblings']]
        # Wrap the selected siblings in the ORIGINAL enclosing loop domains,
        # so every expression is bound and outer invocation equality is exact.
        for parent in reversed(ancestors):
            selected=(replace(parent,body=tuple(selected)),)
        subkernel=replace(kernel,body=tuple(selected))
        sites=_project(_serial_projection(subkernel),adapter,fold_singleton_loops=False)
        pin=_digest(dict(sites=[asdict(s) for s in sites],storage=asdict(storage),
                         outer=[(p.var,p.extent) for p in ancestors],block=asdict(block)))
        group_pins.append(pin)
        if pin in _PROOFS:
            checked+=_PROOFS[pin];_PROOFS.move_to_end(pin);continue
        outer=tuple(p.var for p in ancestors)
        loads=[s for s in sites if type(s.operation) is Load]
        reads={'lhs':[],'weight':[]}
        for site in sites:
            if type(site.operation) is not Compute:
                continue
            if site.ordinal==0 or type(pre:=sites[site.ordinal-1].operation) is not Preload:
                raise CompletionRefusal('completion: actual compute has no preceding preload descriptor')
            previous=sites[site.ordinal-1]
            if previous.scopes!=site.scopes:
                raise CompletionRefusal('completion: preload and compute cross counted scopes')
            op=site.operation
            reads['lhs'].append((site,op.input_block,op.input_row,op.rows,op.cols))
            reads['weight'].append((previous,pre.weight_block,pre.weight_row,pre.weight_rows,pre.cols))
        # One K stream must stay inside one ORIGINAL logical output block.
        # This protects output-block boundaries even on valid source programs
        # whose stores are placed later, outside their compute-only loops.
        computes=[s for s in sites if type(s.operation) is Compute]
        reference=computes[0]
        reference_pre=sites[reference.ordinal-1].operation
        for ci,compute in enumerate(computes):
            env,constraints=domain(compute,f'g{gi}:owner{ci}')
            witness={name:env[name] if name in outer else z3.IntVal(0)
                     for _,name,_ in reference.scopes}
            pre=sites[compute.ordinal-1].operation
            different=z3.Or(
                x(compute.operation.input_block[0],env)/block.m != x(reference.operation.input_block[0],witness)/block.m,
                x(pre.weight_block[1],env)/block.n != x(reference_pre.weight_block[1],witness)/block.n)
            if not unsat(z3.And(*constraints,different)):
                raise CompletionRefusal('completion: merged stream has unknown or changing original output-block ownership')
        group_checked=0
        for ai,a in enumerate(loads):
            for bi,b in enumerate(loads):
                ao,bo=a.operation,b.operation
                # Source checking proves LHS/RHS assigned byte views disjoint.
                if ao.role!=bo.role:
                    continue
                ae,ac=domain(a,f'g{gi}:a{ai}');be,bc=domain(b,f'g{gi}:b{bi}')
                row=z3.Int(f'g{gi}:cell_row');col=z3.Int(f'g{gi}:cell_col')
                ar,br=x(ao.row,ae),x(bo.row,be)
                overlap=z3.And(row>=ar,row<ar+x(ao.rows,ae),row>=br,row<br+x(bo.rows,be),
                               col>=0,col<x(ao.cols,ae),col<x(bo.cols,be))
                pending=z3.And(*ac,*bc,*(ae[n]==be[n] for n in outer),before(a,ae,b,be),overlap)
                if unsat(pending):
                    continue
                covered=[];witnesses=[]
                for ci,(read,owner,rr,rn,cn) in enumerate(reads[ao.role]):
                    re,rc=domain(read,f'g{gi}:r{ci}')
                    # Equal logical tile coordinates provide constructive
                    # integer witnesses, while actual physical row/column
                    # coverage and issue order remain independent conjuncts.
                    match=[x(v,re)==x(w,ae) for v,w in zip(owner,ao.block,strict=True)]
                    claim=z3.And(*rc,*(re[n]==ae[n] for n in outer),*match,
                        before(a,ae,read,re),before(read,re,b,be),
                        row>=x(rr,re),row<x(rr,re)+x(rn,re),col<x(cn,re))
                    full,witness=_defined_exists(tuple(re.values()),claim,
                        instances=tuple((v,z3.IntVal(0)) for v in re.values()))
                    covered.append(full);witnesses.append(witness)
                if not unsat(z3.And(pending,z3.Not(z3.Or(*witnesses)))):
                    if not unsat(z3.And(pending,z3.Not(z3.Or(*covered)))):
                        raise CompletionRefusal('completion: overlapping operand writes lack a proved intervening RAW/WAR read')
                group_checked+=1
        _PROOFS[pin]=group_checked
        while len(_PROOFS)>64:
            _PROOFS.popitem(last=False)
        checked+=group_checked
    return dict(schema='merlin.dense_completion_load_order.v1',
        projected_scopes_sha256=tuple(group_pins),overlapping_site_pairs_proved=checked,
        proof='every overlapping load-write cell has an intervening actual read; local RAW then WAR',
        load_waw_ordering='not_assumed', output_block_ownership='uniform original logical M/N block per merged scope',
        dynamic_domains='symbolic counted loops')
