"""Compose checked shared RHS source ownership with selected ready readout.

The original shared proof owns all external pointers and independent members.
Each proposed member consumes the same dense source/hazard proof engine and
the separately selected final-K visibility law. The checked actual program,
including its retained prefix and original member pointers, reaches Encoder.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call,Loop,TensorArg
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN,_verify_dense_epochs
from merlin.compile.scheduling.dense_ready_readout import (
    propose_dense_ready_readout,DenseReadyReadoutCheckingProjection)
from merlin.compile.scheduling.dense_shared_rhs import DenseSharedRHSResult
from merlin.sched.primitives import Cursor,unroll


@dataclass(frozen=True)
class SharedRHSReadyResult:
    original: object
    kernel: object
    realization: dict
    _guard: object

    @property
    def blocks(self):return self.original.blocks

    def revalidate(self):self._guard(self)

    emit_index_program=DenseSharedRHSResult.emit_index_program


def compose_shared_rhs_ready(original,problems,resources,storages,adapter,
                             encode_member,project_member,reduction,epoch):
    """Return an actual checked frontier; unsupported layouts raise ValueError."""
    from dense_readout import resident_readout_laws
    original.revalidate()
    owner=sha256(Path(__file__).read_bytes()).hexdigest()
    if any(p.bias for p in problems):raise ValueError('shared ready requires zero-seeded products')

    def canonical(items):
        return tuple(replace(s,body=canonical(s.body)) if type(s) is Loop else
            s if s.instr==DRAIN else adapter.encode(project_member(s)[1]) for s in items)

    prefix=replace(original.kernel.body[0],body=canonical(original.kernel.body[0].body))
    sections=[original.kernel.body[0]];proofs=[];records=[]
    for member,(p,b,storage) in enumerate(zip(problems,original.blocks,storages,strict=True)):
        section=original.kernel.body[member+1]
        base=replace(original.kernel,args=(TensorArg('lhs',(p.m,p.k),'i8','read'),
            TensorArg('rhs',(p.k,p.n),'i8','read'),TensorArg('output',(p.m,p.n),'i32','write')),
            body=(prefix,replace(section,body=canonical(section.body))))
        # Compact shared members retain a singleton full-K panel wrapper.
        # Consume the library's exact unroll rewrite so the ready pass sees its
        # required three original physical M/K/N axes, including real tails.
        def wrappers(items,path=()):
            for s in items:
                if type(s) is not Loop:continue
                if s.extent==1 and s.var.startswith('panel'):
                    return (*path,s.var)
                found=wrappers(s.body,(*path,s.var))
                if found:return found
            return None
        unit_rewrites=[]
        while (path:=wrappers(base.body)) is not None:
            base=unroll(base,Cursor(loops=path))
            unit_rewrites.append(path)
        actual,changes=propose_dense_ready_readout(base,p,adapter)
        bound,admission=resident_readout_laws(adapter,b,reduction,epoch)
        projection=DenseReadyReadoutCheckingProjection(base,p)
        report=_verify_dense_epochs(actual,p,resources,storage,adapter,bound,
            static_budget=256,timeout_ms=20000,ordered_accumulator_commits=True,
            source_phase_fission=True,readout_compute_admission=admission,
            readout_source_projection=projection)

        def tagged(items):
            return tuple(replace(s,body=tagged(s.body)) if type(s) is Loop else
                s if s.instr==DRAIN else encode_member(member,adapter.project(s)) for s in items)

        # The shared RHS prefix remains exactly the original checked prefix.
        # Only the member region is substituted, after actual epoch verification.
        if actual.body[0]!=prefix:raise ValueError('ready rewrite changed shared RHS prefix')
        sections.extend(replace(s,body=tagged(s.body)) for s in actual.body[1:])
        proofs.append((actual,p,storage,bound,admission,projection))
        records.append(dict(member=member,unit_panel_unrolls=unit_rewrites,
            transformations=changes,proof=report))
    kernel=replace(original.kernel,body=tuple(sections))
    digest=kernel.digest()
    def drains(items):
        return sum(s.extent*drains(s.body) if type(s) is Loop else int(s.instr==DRAIN) for s in items)

    def guard(result):
        original.revalidate()
        if (sha256(Path(__file__).read_bytes()).hexdigest()!=owner
            or result.kernel.digest()!=digest or result.kernel!=kernel):
            raise ValueError('shared ready owner or actual checked program changed')
        for _,_,_,bound,admission,_ in proofs:admission.validate_for(adapter,bound)

    result=SharedRHSReadyResult(original,kernel,{**original.realization,
        'kernel_sha256':digest,'schema':'merlin.shared_rhs_ready_composition.v1',
        'original_kernel_sha256':original.kernel.digest(),
        'original_dynamic_epoch_drains':original.realization['dynamic_epoch_drains'],
        'dynamic_epoch_drains':drains(kernel.body),
        'ready_members':records,'owner_sha256':owner},guard)
    result.revalidate()
    return result
