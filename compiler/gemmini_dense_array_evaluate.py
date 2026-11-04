#!/usr/bin/env python3
"""Freeze ordinary panel/array-retained dense products; request target checks only explicitly.

Counts never rank alternatives. The replay observes the actual checked kernel
consumed by the existing index emitter and reproduces ordinary manifest bytes.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'compiler_library')]
from gemmini_schedule_evaluate import source_record, evaluate
from gemmini_conv_evaluate import copy_package, digest, tree_identity, manifest_emit


def local_checks(output):
    """Execute actual projected fresh/STAY effects on independent host integers.

    This is validation only; production lowering contains no tensor arithmetic.
    Both initializer components enter the source comparison independently.
    """
    from dataclasses import replace
    from merlin.sched.ir import instances, evaluate as eval_expr
    from merlin.compile.scheduling.block_schedule import Load, Preload, Compute, Store, InitializeAccumulator
    from gemmini_dense import plan_dense, geometry, FACTS_PATH, dense_terminal
    from gemmini_dense_array import plan_array, wrap_terminal
    from merlin.llvmlower.host_builder import FnBuilder, PTR
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from gemmini_support import finish_module
    import gemmini_encode as encoder
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    cases=[((17,1,17),False,False),((33,17,19),False,False),((49,33,35),False,False),
        ((129,64,65),False,False),((128,16,128),False,False),
        ((33,17,19),True,False),((33,17,19),False,True),((33,17,19),True,True)]
    for shape,bias,initial in cases:
        incumbent,adapter,p=plan_dense(*shape,bias=bias,initial=initial)
        result,receipt,retained=plan_array(incumbent,adapter,p,geometry(),FACTS_PATH,bias=bias,initial=initial)
        result.materialize_scf().verify()
        lhs=[[((i*p.k+k)*3)%13-6 for k in range(p.k)] for i in range(p.m)]
        rhs=[[((k*p.n+j)*5)%17-8 for j in range(p.n)] for k in range(p.k)]
        seed=lambda i,j: ((i*11+j*7)%19-9 if initial else 0)+((j*3)%11-5 if bias else 0)
        expected=[[seed(i,j)+sum(lhs[i][k]*rhs[k][j] for k in range(p.k)) for j in range(p.n)] for i in range(p.m)]
        out=[[None]*p.n for _ in range(p.m)]
        memory={};acc={};pending=None;array=None;counts={}
        for call,env in instances(result.kernel):
            counts[call.instr]=counts.get(call.instr,0)+1
            if call.instr=='merlin_dense_panel_drain':continue
            op=retained.project(replace(call,args=tuple((key,eval_expr(value,env)) for key,value in call.args)))
            if type(op) is Load:
                source=lhs if op.role=='lhs' else rhs
                for i in range(op.rows):
                    for j in range(op.cols):memory[op.role,op.row+i,j]=source[op.dram_row+i][op.dram_col+j]
            elif type(op) is InitializeAccumulator:
                for i in range(op.rows):
                    for j in range(op.cols):
                        source_index=op.dram_row*op.source_row_stride+op.dram_col+i*op.source_row_stride+j*op.source_col_stride
                        value=((source_index//p.n*11+source_index%p.n*7)%19-9 if initial else (source_index*3)%11-5)
                        if op.additional_bias:value+=((op.dram_col+j)*3)%11-5
                        acc[op.row+i,j]=value
            elif type(op) is Preload:pending=op
            elif type(op) is Compute:
                if op.fresh_weights:
                    array=[[memory['weight',pending.weight_row+k,j] for j in range(pending.cols)] for k in range(pending.weight_rows)]
                for i in range(op.rows):
                    for j in range(pending.cols):
                        old=acc[pending.accumulator_row+i,j] if pending.accumulate else 0
                        acc[pending.accumulator_row+i,j]=old+sum(memory['lhs',op.input_row+i,k]*array[k][j] for k in range(op.cols))
                pending=None
            elif type(op) is Store:
                for i in range(op.rows):
                    for j in range(op.cols):out[op.dram_row+i][op.dram_col+j]=acc[op.accumulator_row+i,j]
            else:raise ValueError('unhandled actual validation effect')
        if out!=expected:raise AssertionError('actual fresh/STAY seed/source integer mismatch')
        mutated=replace(result,proposal=replace(result.proposal,kernel=replace(result.kernel,name='tampered')))
        try:mutated.revalidate()
        except ValueError:pass
        else:raise AssertionError('mutated actual kernel was accepted')
        rows.append(dict(shape=shape,bias=bias,initial=initial,block=receipt['block'],counts=counts,
            actual_kernel_sha256=result.kernel.digest(),checking_kernel_sha256=receipt['source_schedule_sha256'],
            retained_pairs=receipt['array_proof']['retained_pairs'],exact_integer='PASS',
            complete_source_and_epochs='PASS',mutation_refusal='PASS'))
    # Exercise actual common terminal packing, including dynamic SSA operands.
    incumbent,adapter,p=plan_dense(33,17,19)
    result,receipt,retained=plan_array(incumbent,adapter,p,geometry(),FACTS_PATH)
    fn=FnBuilder([PTR]*3);lhs,rhs,dst=fn.entry.args
    lower=wrap_terminal(fn,dense_terminal(fn,lhs,rhs,dst,p,asynchronous=True),p)
    result.emit_index_program(FnBuilderControlAdapter(fn),lower,complete_epoch=lambda:encoder.fence(fn))
    encoder.fence(fn);llvm=finish_module(fn)
    if '.insn r 0x7b, 3, 5,' not in llvm:raise AssertionError('actual terminal did not emit funct5')
    (output/'terminal.mlir').write_text(llvm+'\n')
    report=dict(cases=rows,terminal_sha256=digest((llvm+'\n').encode()),
        terminal='same encoder actual funct4/funct5/preload output descriptor PASS',
        execution_qualification=None,cycles='UNKNOWN')
    (output/'local_checks.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def consume_replay(package, source, output, expected, mode):
    script = r'''
import sys,json,re
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package),str(package/'compiler_library')]
import gemmini_dense_array as dense_array
from merlin.compile.scheduling import llvm_program
plans=[];consumed=[];effects=[]
def norm(v):
 if v is None or type(v) in (str,int,float,bool):return v
 if isinstance(v,dict):return {str(k):norm(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [norm(x) for x in v]
 return re.sub(r'0x[0-9a-fA-F]+','<address>',str(v))
original_plan=dense_array.plan_array
def plan(*args,**kwargs):
 result,receipt,adapter=original_plan(*args,**kwargs)
 plans.append(dict(kernel_sha256=result.kernel.digest(),receipt=norm(receipt)))
 return result,receipt,adapter
dense_array.plan_array=plan
original_emit=llvm_program.emit_index_effect_program
def emit(kernel,control,lower_effect,**kwargs):
 consumed.append(dict(kernel_sha256=kernel.digest()))
 def lower(kind,fields):
  effects.append(dict(kind=kind,fields=norm(fields)))
  return lower_effect(kind,fields)
 return original_emit(kernel,control,lower,**kwargs)
llvm_program.emit_index_effect_program=emit
from gemmini_backend import compile_source
_,out=compile_source(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_bytes((out+'\n').encode())
Path(sys.argv[4]).write_text(json.dumps(dict(plans=plans,consumed=consumed,effects=effects),sort_keys=True,indent=2)+'\n')
'''
    replay, receipt = output/'replay.mlir', output/'consumed.json'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package/'compiler_library'))
    run = subprocess.run([sys.executable, '-c', script, str(package), str(source), str(replay), str(receipt)],
        env=env, cwd=package, capture_output=True, timeout=180)
    (output/'replay.stderr').write_bytes(run.stderr)
    if run.returncode:
        raise RuntimeError('consumption replay failed: '+str(output/'replay.stderr'))
    if replay.read_bytes() != expected:
        raise ValueError('observing actual checked consumption changed manifest product')
    observed = json.loads(receipt.read_text())
    if mode == 'off':
        if observed['plans']:
            raise ValueError('incumbent unexpectedly entered array-retained dense')
    else:
        if not observed['plans'] or not any(row['kind']=='gemmini.compute.array' for row in observed['effects']):
            raise ValueError('array-retained dense was not actually planned and emitted')
        emitted = [row['kernel_sha256'] for row in observed['consumed']]
        if any(row['kernel_sha256'] not in emitted for row in observed['plans']):
            raise ValueError('planned dense array kernel was not consumed by actual index emitter')
    return dict(path=str(receipt), planned_kernel_sha256=[r['kernel_sha256'] for r in observed['plans']],
        consumed_kernel_sha256=[r['kernel_sha256'] for r in observed['consumed']],
        actual_effect_sha256=digest(observed['effects']), replay_byte_identical=True)


def prepare(*, submission, output, prefix, training, held_out):
    from merlin.transforms.pass_exploration import (
        PassFamily, EmittedProgram, ValidationPartition, generate_candidates, explore_candidates)
    from merlin.transforms.pipeline_policy import RegisteredPass, PolicyFacts
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext
    submission, output = Path(submission).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('use a fresh output outside submission')
    if not prefix or len(prefix)>40 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in prefix):
        raise ValueError('use a bounded alphanumeric variant prefix')
    output.mkdir(parents=True); (output/'sources').mkdir()
    records=[]; groups=[]
    members=json.loads((WORKSPACE/'PUBLIC_CAPABILITIES.json').read_text())['members']
    for split, path in (('training',training),('held_out',held_out)):
        geometry, record=source_record(path,split,output/'sources')
        if len(record['shapes'])!=2 or record['shapes'][0][0]<=16:
            raise ValueError('freeze one original repeated-M contraction per complete source')
        original=Path(path);case=original.parent.name if original.name.startswith('capsule.') else original.stem
        if record['source_sha256']!=members[case]['source_analysis']['sha256']:
            raise ValueError('source differs from public declaration')
        record['capsule']=case; records.append(record); groups.append((geometry,))
    partition=ValidationPartition(*groups)
    before=tree_identity(submission);frozen=output/'compiler';copy_package(submission,frozen)
    if tree_identity(submission)!=before or tree_identity(frozen)!=before:
        raise ValueError('compiler changed during snapshot; retry fresh output')
    modes=('off','array_epochs');variants={}
    for mode in modes:
        path=WORKSPACE/'variants'/(prefix+'-'+mode)
        if path.exists():raise ValueError('variant exists: '+str(path))
        marker=path.parent/('.library-pass-pending-'+path.name)
        marker.write_text('dense array preparation pending; do not evaluate.\n')
        copy_package(frozen,path)
        decl=json.loads((path/'TUNING_KNOBS.json').read_text())
        value='panel_epochs' if mode=='off' else 'array_epochs'
        if value not in decl['knobs']['dense_schedule']['values']:
            raise ValueError('dense array option is not declared')
        settings_path=path/decl['settings_file']
        settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings['dense_schedule']=value
        settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        variants[mode]=dict(package=str(path),compiler_sha256=tree_identity(path),settings=settings)
    emitted=[]
    def select(knobs):
        def apply(session):
            session.module['mode']=knobs['mode'];return dict(settings=knobs['mode'])
        return RegisteredPass(PassSpec('dense-array-retention',0,produces=('settings',),ir_level='authored_source'),apply)
    def emit(_):
        def apply(session):
            mode=session.module['mode'];record=session.module['source']
            package=Path(variants[mode]['package']);location=output/'products'/record['source_sha256']/mode
            payload,receipt=manifest_emit(package,Path(record['frozen_path']),location,timeout=180)
            observed=consume_replay(package,Path(record['frozen_path']),location,payload,mode)
            emitted.append(dict(mode=mode,source_sha256=record['source_sha256'],split=record['split'],
                capsule=record['capsule'],consumption=observed,**receipt))
            return dict(program=dict(data=payload,schedule_sha256=digest(observed)))
        return RegisteredPass(PassSpec('ordinary-manifest-emitter',1,requires=('settings',),
            produces=('program',),mutates_ir=False,ir_level='authored_source'),apply)
    space=generate_candidates((PassFamily('dense-array-retention',select,(('mode',modes),)),
        PassFamily('ordinary-manifest-emitter',emit)),output='program',max_candidates=2)
    context=PipelineContext(hardware=digest((frozen/'data/rtl_facts.json').read_bytes()),
        numerical_policy=digest((frozen/'data/target_contract.yaml').read_bytes()),library=tree_identity(frozen/'compiler_library'))
    pipelines=[]
    for record in records:
        for row in explore_candidates({'source':record,'mode':'off'},space,
                lambda session:PolicyFacts(session.stamp(),shapes=tuple(map(tuple,record['shapes']))),
                output='program',emission=lambda result:EmittedProgram(result.artifact['data'],'llvm-mlir',result.artifact['schedule_sha256']),
                context=context,ir_level='authored_source',clone=copy.deepcopy,fingerprint=digest,
                verify_ir=lambda module:None):
            pipelines.append({**row.summary(),'source_sha256':record['source_sha256']})
    (output/'pipeline_diagnostics.json').write_text(json.dumps(pipelines,indent=2,default=str)+'\n')
    for record in records:
        pair=[r for r in emitted if r['source_sha256']==record['source_sha256']]
        if len(pair)!=2 or len({r['program_sha256'] for r in pair})!=2:
            raise ValueError('array-retained dense did not change actual ordinary product')
    report=dict(schema='gemmini.dense_array_experiment.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=before,partition=partition.manifest(),variants=variants,sources=records,emissions=emitted,
        baseline='off',scope=dict(capsules=','.join(r['capsule'] for r in records),suite='scalability',sim='spike'),
        requests={},pipeline_reports=pipelines,cycles='UNKNOWN',
        selection='retain incumbent; no matched complete-call cycle evidence',
        hypothesis='checked retained full-K panels and WS array reuse amortize original weight loads and fresh captures across M')
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    for variant in variants.values():
        path=Path(variant['package'])
        (path.parent/('.library-pass-pending-'+path.name)).unlink()
    return report


def native_custody(report):
    """Bind already returned owner numerics to actual ordinary LLVM/build bytes."""
    rows=[]
    for mode, request in report['requests'].items():
        rid=request.get('request_id')
        if not rid:continue
        for source in report['sources']:
            emission=next(e for e in report['emissions'] if e['mode']==mode and e['source_sha256']==source['source_sha256'])
            roots=list((WORKSPACE/'.qa_channel').glob('.resp_'+rid+'.*.scalability/runs/runs/gemmini-capsule-bench/'+source['capsule']))
            if len(roots)!=1:continue
            root=roots[0];generated=root/'generated'
            paths=[root/'capsule_result.json',generated/'lowered.llvm.mlir',generated/'readback_build.json',generated/'lowering_recipe.json',generated/'kernel.ll']
            if not all(path.is_file() for path in paths):continue
            owner=json.loads(paths[0].read_text());build=json.loads(paths[2].read_text());recipe=json.loads(paths[3].read_text())
            if digest(paths[1].read_bytes())!=emission['program_sha256']:
                raise ValueError('native owner executed a different ordinary dense array product')
            if recipe['returned_llvm_ir']['sha256']!=digest(paths[4].read_bytes()):
                raise ValueError('native returned LLVM identity differs')
            rows.append(dict(mode=mode,capsule=source['capsule'],request_id=rid,
                source_sha256=source['source_sha256'],ordinary_llvm_sha256=emission['program_sha256'],
                actual_llvm_sha256=digest(paths[1].read_bytes()),returned_llvm_sha256=digest(paths[4].read_bytes()),
                kernel_object_sha256=build['kernel_object_sha256'],elf_sha256=build['elf_sha256'],
                owner_path=str(root),numeric=owner.get('numeric'),owner_qualification=owner.get('status'),
                owner_failure=owner.get('failure'),elf_binary_retained=False))
    return rows


def collect(report_path):
    """Resume only existing jobs, then bind owner LLVM/object/ELF and complete calls."""
    from gemmini_residual_evaluate import _body, owner_product_custody, compare_complete_calls
    evaluate(report_path,collect=True)
    path=Path(report_path).resolve();report=json.loads(path.read_text())
    modes=('off',next(mode for mode in report['variants'] if mode!='off'))
    normalized={**report,'variants':{mode:report['variants'][mode] for mode in modes},'requests':{}}
    for mode in modes:
        request=report['requests'][mode]
        response=Path(request['response_path'])
        body=json.loads(response.read_text()) if response.exists() else _body((path.parent/('selfcheck-'+mode+'.stdout')).read_text())
        normalized['requests'][mode]={**request,'route':mode,'suite':report['scope']['suite'],'evidence':body}
    comparison=compare_complete_calls(normalized)
    # UNKNOWN timing has no ELF identity and must stay outside performance
    # custody/ranking; numerical build custody is still checked independently.
    timed=copy.deepcopy(normalized)
    for request in timed['requests'].values():
        body=request['evidence'];key='cases' if 'cases' in body else 'per_capsule'
        rows=[]
        for row in body.get(key,[]):
            detail=row.get('detail',{});location=WORKSPACE/detail.get('path','')
            performance=json.loads(location.read_text()).get('performance',{}) if detail.get('sha256') and location.is_file() else row.get('performance',{})
            if performance.get('elf_sha256'):rows.append(row)
        body[key]=rows
    owner=owner_product_custody(timed)
    result=dict(comparisons=comparison,owner_product_custody=owner,native_numeric_custody=native_custody(report),
                evidence='modeled Spike complete calls; no measured speedup established',
                default_promotion=False)
    (path.parent/'owner_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--submission',default='submission');p.add_argument('--output',required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--train',required=True);p.add_argument('--held-out',required=True)
    p=sub.add_parser('local-checks');p.add_argument('--output',required=True)
    for name in ('evaluate','collect'):
        p=sub.add_parser(name);p.add_argument('--report',required=True)
    a=parser.parse_args()
    if a.command=='local-checks':print(json.dumps(local_checks(a.output),indent=2))
    elif a.command=='prepare':
        r=prepare(submission=a.submission,output=a.output,prefix=a.prefix,training=a.train,held_out=a.held_out)
        print(json.dumps(dict(report=str(Path(a.output)/'report.json'),products=len(r['emissions']))))
    elif a.command=='collect':print(json.dumps(collect(a.report),indent=2))
    else:print(json.dumps(evaluate(a.report),indent=2))
