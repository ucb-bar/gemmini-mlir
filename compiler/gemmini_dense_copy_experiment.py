"""Isolated private-copy pipeline experiment; production remains unchanged.

Calls the real shared pass on the checked dense stream and, within a finite
budget, on its exact concrete expansion. Unknown completion/descriptor effects
refuse motion. The incumbent source owner and terminal remain authoritative;
there is no metadata-only substitute for an admitted transformed artifact.

Opt-in ordinary manifest entry: replace a frozen variant's gemmini-opt with
``from gemmini_dense_copy_experiment import main; main()``. No production hook.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import runpy
import sys

from merlin.compile.scheduling.copy_pipeline import pipeline_private_copies,CopyPipelinePolicy
from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable
from merlin.sched.ir import evaluate,instances,as_expr
from gemmini_dense_copy_adapter import copy_storage_boundary


def try_pipeline_private_dense(incumbent,adapter,problem,geometry,facts_path,*,distance=8,max_calls=65536):
    """Attempt a real bounded pass; retain the exact incumbent on safe refusal.

    No token, wait, private allocation, configuration or arithmetic instruction
    is invented. A newly qualified transformed candidate would additionally
    require admission by its original dense source owner before publication.
    """
    if distance not in (1,8,32):raise ValueError('unregistered bounded copy motion distance')
    incumbent.revalidate();before=incumbent.kernel.digest()
    storage,objects,banks,windows,admission=copy_storage_boundary(adapter,problem,geometry,facts_path)
    policy=CopyPipelinePolicy(distance=distance,max_moves=32,max_pending_copies=1)
    options=dict(policy=policy,external_windows=windows,private_ownership=True,max_calls=max_calls)
    attempts=[]
    def attempt(kernel,kind):
        try:
            candidate,report=pipeline_private_copies(kernel,objects,banks,storage,admission,**options)
            attempts.append(dict(kind=kind,status=report['status'],proof=report,
                candidate_sha256=candidate.digest()))
            if candidate!=kernel:
                raise StorageOptimizationUnavailable('original dense owner has no admission for copy-pipeline motion')
        except StorageOptimizationUnavailable as error:
            attempts.append(dict(kind=kind,status='refused',reason=str(error)))
    attempt(incumbent.kernel,'original-counted-kernel')
    calls=[];counts=Counter();sample=[];exhausted=False
    for call,env in instances(incumbent.kernel):
        if len(calls)>=max_calls:
            exhausted=True;break
        concrete=replace(call,args=tuple((key,as_expr(evaluate(value,env))) for key,value in call.args))
        calls.append(concrete);counts[call.instr]+=1
        if len(sample)<8:
            effect=storage.project(concrete)
            sample.append(dict(instruction=concrete.instr,operands={k:evaluate(v,{}) for k,v in concrete.args},
                effect=asdict(effect)))
    if not exhausted:
        attempt(replace(incumbent.kernel,body=tuple(calls)),'full-exact-concrete-expansion')
    else:
        attempts.append(dict(kind='full-exact-concrete-expansion',status='refused',
            reason='complete original source exceeds finite expansion budget; no prefix is scheduled'))
    incumbent.revalidate()
    if incumbent.kernel.digest()!=before:raise ValueError('copy attempt mutated original checked source')
    return incumbent,dict(schema='gemmini.dense-copy-attempt.v1',status='refused',
        source_problem=asdict(problem),before_effect_sha256=before,after_effect_sha256=before,
        original_owner_retained=True,policy=asdict(policy),admission=asdict(admission),
        fixed_objects=[asdict(x) for x in objects],banks=[asdict(x) for x in banks],
        attempts=attempts,expansion_budget=max_calls,expanded_calls=len(calls),
        expansion_complete=not exhausted,expanded_instruction_counts=dict(counts),actual_effect_sample=sample,
        limitations=['no actual per-copy token waits in dense source',
            'global drain also completes execute and readout; not token-completion-only',
            'compute/preload descriptors and asynchronous readout are outside copy_pipeline semantics',
            'strided source rows and banded local destinations are not one contiguous exact copy',
            'counted dense loop motion is outside the bounded straight-line shared pass'],
        timing='UNKNOWN; unchanged program, no timing request')


@contextmanager
def install_copy_attempt(records,*,distance=8):
    """Opt-in frozen-package bridge at the real long-K planner return boundary."""
    import gemmini_dense_longk_experiment as owner
    original=owner.plan_longk_output
    def plan(*args,**kwargs):
        checked,receipt=original(*args,**kwargs)
        result,proof=try_pipeline_private_dense(checked,args[1],args[2],args[3],args[4],distance=distance)
        records.append(proof)
        return result,{**receipt,'copy_pipeline_experiment':proof}
    owner.plan_longk_output=plan
    try:yield
    finally:owner.plan_longk_output=original


def main():
    records=[]
    output=next((x.split('=',1)[1] for x in sys.argv if x.startswith('--copy-report=')),None)
    sys.argv=[x for x in sys.argv if not x.startswith('--copy-report=')]
    with install_copy_attempt(records):
        runpy.run_path(str(Path(__file__).resolve().parent/'gemmini-opt-original'),run_name='__main__')
    if output:Path(output).write_text(json.dumps(records,indent=2)+'\n')
