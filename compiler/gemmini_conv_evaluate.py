#!/usr/bin/env python3
"""Freeze ordinary convolution compiler products and request matched broker checks.

prepare registers settings with PassFamily and consumes actual transformed target
products. evaluate/collect use evaluate_emitted with ordinary selfcheck ownership.
No missing cycle, input-custody or complete-ROI evidence becomes a ranking.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
WORKSPACE=PACKAGE.parent
sys.path[:0]=[str(PACKAGE/'compiler_library'),str(PACKAGE)]


def digest(value):
    if not isinstance(value,bytes): value=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(value).hexdigest()


IGNORED={'__pycache__','.git','qa','selfcheck_out','.qa_channel','build','out','evidence'}


def tree_identity(root):
    rows={str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(root.rglob('*'))
        if p.is_file() and not any(part in IGNORED for part in p.relative_to(root).parts)}
    return digest(rows)


def copy_package(source,destination):
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns(*IGNORED))


def source_record(path,split,directory):
    from gemmini_frontend import normalize_source
    from merlin.transforms.pass_exploration import ExplorationGeometry
    path=Path(path).resolve();payload=path.read_bytes();model=normalize_source(payload.decode())
    # Preserve complete convolution geometry, including source stride/halo and readout.
    import re
    shapes=tuple(tuple(map(int,m.split('x'))) for m in re.findall(r'tensor<([0-9x]+)xi(?:8|32)>',payload.decode()))
    if 'conv2d' not in payload.decode() or len(shapes)<2:
        raise ValueError('expected a declared conv2d source with native tensor shapes')
    geometry=ExplorationGeometry('convolution',shapes)
    saved=directory/(split+'-'+digest(payload)+'.mlir');saved.write_bytes(payload)
    return geometry,dict(path=str(path),frozen_path=str(saved),source_sha256=digest(payload),
        split=split,geometry_sha256=geometry.identity,shapes=shapes)


def manifest_emit(package,source,output,*,timeout=120):
    """Invoke exactly the ordinary declared target emitter, without a shell."""
    import yaml
    manifest=yaml.safe_load((package/'manifest.yaml').read_text())
    declaration=manifest['commands']['lower_target_to_llvm']['argv']
    tool=package/manifest['entrypoints']['tool']
    bindings={'tool':str(tool),'input_mlir':str(source)}
    argv=[token.format(**bindings) for token in declaration]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(package/'compiler_library'))
    # Strides/order are compiler settings, never ambient exploration state.
    for key in ('GEMMINI_DENSE_BLOCK_LIMIT','GEMMINI_DENSE_OUTER_ORDER'):
        env.pop(key,None)
    process=subprocess.run(argv,cwd=package,env=env,capture_output=True,timeout=timeout)
    output.mkdir(parents=True,exist_ok=False)
    (output/'stdout.mlir').write_bytes(process.stdout);(output/'stderr.txt').write_bytes(process.stderr)
    if process.returncode or not process.stdout:
        raise RuntimeError('ordinary manifest emission failed: '+str(output/'stderr.txt'))
    return process.stdout,dict(argv=argv,cwd=str(package),returncode=process.returncode,
        product_path=str(output/'stdout.mlir'),program_sha256=digest(process.stdout))


def prepare(*,submission,output,variant_prefix,training,held_out,capsules,suite='core',sim='spike',
            modes=('off1','input_rhs1','input_rhs8','rolling_rhs8')):
    """Return a reviewable experiment using PassFamily and actual target bytes."""
    from merlin.transforms.pass_exploration import (PassFamily,EmittedProgram,ExplorationRefused,
        ValidationPartition,generate_candidates,explore_candidates)
    from merlin.transforms.pipeline_policy import RegisteredPass,PolicyFacts
    from merlin.transforms.pipeline_state import PassSpec,PipelineContext
    submission=Path(submission).resolve();output=Path(output).resolve()
    if output.exists(): raise ValueError('use a new experiment output directory')
    if output.is_relative_to(submission): raise ValueError('experiment output must be outside submission')
    if (not variant_prefix or not variant_prefix[0].isalnum() or len(variant_prefix)>40
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in variant_prefix)):
        raise ValueError('variant prefix must start with a letter or digit, use letters/digits/underscore/hyphen, and fit 40 characters')
    if tuple(modes)[0]!='off1' or set(modes)-{'off1','input_rhs1','input_rhs8','rolling_rhs8'}:
        raise ValueError('include off1 first and use only implemented convolution alternatives')
    output.mkdir(parents=True);sources=output/'sources';sources.mkdir()
    groups=[];records=[]
    for split,paths in (('training',training),('held_out',held_out)):
        group=[]
        for path in paths:
            geometry,record=source_record(path,split,sources);group.append(geometry);records.append(record)
        groups.append(tuple(group))
    partition=ValidationPartition(*groups)
    before=tree_identity(submission)
    frozen=output/'compiler';copy_package(submission,frozen)
    if tree_identity(submission)!=before or tree_identity(frozen)!=before:
        raise RuntimeError('compiler changed while freezing; retry in a new experiment directory')
    variants={};variant_root=WORKSPACE/'variants';variant_root.mkdir(exist_ok=True)
    for mode in modes:
        variant=variant_root/(variant_prefix+'-'+mode)
        if variant.exists(): raise ValueError('variant already exists: '+str(variant))
        marker=variant_root/('.library-pass-pending-'+variant.name)
        marker.write_text('Exploration preparation pending; do not evaluate incomplete variants.\n')
        copy_package(frozen,variant)
        declaration=json.loads((variant/'TUNING_KNOBS.json').read_text())
        retention, pairs = ('off',1) if mode=='off1' else (mode[:-1],int(mode[-1]))
        if retention not in declaration['knobs']['conv_retention']['values'] or pairs not in declaration['knobs']['conv_epoch_pairs']['values']:
            raise ValueError('dense_schedule alternative is not declared')
        settings_path=variant/declaration['settings_file']
        settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings['conv_retention']=retention;settings['conv_epoch_pairs']=pairs;settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        variants[mode]={'package':str(variant),'compiler_sha256':tree_identity(variant),'settings':settings}
    emitted=[]
    def select(knobs):
        def action(session):
            session.module['mode']=knobs['mode'];return {'settings':knobs['mode']}
        return RegisteredPass(PassSpec('conv-settings',0,produces=('settings',),ir_level='authored_source'),action)
    def emit(_):
        def action(session):
            mode=session.module['mode'];variant=Path(variants[mode]['package']);source=session.module['source']
            location=output/'products'/source['source_sha256']/mode
            try:product,receipt=manifest_emit(variant,Path(source['frozen_path']),location)
            except RuntimeError as exc:
                raise ExplorationRefused(str(exc),layer='emission') from exc
            emitted.append(dict(mode=mode,split=source['split'],source_sha256=source['source_sha256'],
                geometry_sha256=source['geometry_sha256'],compiler_sha256=variants[mode]['compiler_sha256'],pipeline_receipt=None,**receipt))
            replay=consume_replay(variant,Path(source['frozen_path']),location,product)
            emitted[-1].update(replay)
            # Artifact is the actual ordinary emitter stdout, with its source and settings identity.
            return {'program':{'data':product,'schedule_sha256':digest({'source':source['source_sha256'],
                'settings':variants[mode]['settings'],'compiler':variants[mode]['compiler_sha256'],'effects':replay['effect_sha256']})}}
        return RegisteredPass(PassSpec('ordinary-manifest-emitter',1,requires=('settings',),produces=('program',),
            mutates_ir=False,ir_level='authored_source'),action)
    space=generate_candidates((PassFamily('conv-settings',select,(('mode',tuple(modes)),)),
        PassFamily('ordinary-manifest-emitter',emit)),output='program',max_candidates=16)
    candidates=[]
    context=PipelineContext(hardware=digest((frozen/'data/rtl_facts.json').read_bytes()),
        numerical_policy=digest((frozen/'data/target_contract.yaml').read_bytes()),
        execution='ordinary-manifest-compiler;target-numerics-and-cycles-pending',library=tree_identity(frozen/'compiler_library'))
    for record in records:
        state={'source':record,'mode':'off1'}
        for row in explore_candidates(state,space,lambda s:PolicyFacts(s.stamp(),shapes=tuple(map(tuple,record['shapes']))),
                output='program',emission=lambda r:EmittedProgram(r.artifact['data'],'llvm-mlir',r.artifact['schedule_sha256']),
                context=context,ir_level='authored_source',clone=copy.deepcopy,fingerprint=digest,
                verify_ir=lambda m:None):
            if row.emission is not None:
                for e in emitted:
                    if e['source_sha256']==record['source_sha256'] and e['program_sha256']==digest(row.emission.data):
                        e['pipeline_receipt']=row.evaluation.result.receipt
            candidates.append({'source_sha256':record['source_sha256'],'split':record['split'],**row.summary()})
    changed=[]
    for record in records:
        rows={r['mode']:r for r in emitted if r['source_sha256']==record['source_sha256']}
        base=rows.get('off1')
        for mode in modes:
            if mode=='off1':continue
            changed.append(dict(source_sha256=record['source_sha256'],mode=mode,
                actual_emitted_change=None if base is None or mode not in rows else base['program_sha256']!=rows[mode]['program_sha256']))
    report=dict(schema='merlin.gemmini.ordinary_schedule_exploration.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=before,partition=partition.manifest(),sources=records,
        variants=variants,candidates=candidates,emissions=emitted,changes=changed,
        scope={'capsules':capsules,'suite':suite,'sim':sim},requests={},cycles='UNKNOWN',
        selection='retain off/1; no matched target cycles yet',
        qualification='actual ordinary manifest emission; target numerics and cycles require selfcheck',
        compare_command='python .agents/skills/ab-library-pass/scripts/ab_compare.py ALTERNATIVE_RESPONSE BASELINE_RESPONSE')
    path=output/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for mode,variant in variants.items():
        successes=[r for r in emitted if r['mode']==mode]
        if len(successes)==len(records):
            (variant_root/('.library-pass-pending-'+Path(variant['package']).name)).unlink()
    return report


def broker_evaluate(report_path,*,collect=False):
    """Submit or resume ordinary checks, preserving identical requested scope."""
    path=Path(report_path).resolve();report=json.loads(path.read_text());scope=report['scope']
    if not scope['capsules']:raise ValueError('provide an explicit matched capsule scope at prepare time')
    for mode,variant in report['variants'].items():
        package=Path(variant['package'])
        if tree_identity(package)!=variant['compiler_sha256']:
            raise ValueError('frozen variant changed: '+str(package))
        response=path.parent/('selfcheck-'+mode+'.json');previous=report['requests'].get(mode)
        argv=[sys.executable,str(WORKSPACE/'agent_selfcheck.py')]
        if collect:
            if not previous:raise ValueError('submit evaluation before collecting')
            rid=previous.get('request_id')
            if not rid:continue
            argv+=['--attach',rid]
        else:
            if previous:raise ValueError('request already exists; use collect')
            argv+=['--submission',str(package),'--capsules',scope['capsules'],'--suite',scope['suite'],'--sim',scope['sim']]
            if scope['suite']=='programs':argv+=['--tiers','L0,L1,L2']
        argv+=['--wait-budget','5','--out',str(response)]
        done=subprocess.run(argv,cwd=WORKSPACE,capture_output=True,timeout=60)
        (path.parent/('selfcheck-'+mode+'.stdout')).write_bytes(done.stdout)
        (path.parent/('selfcheck-'+mode+'.stderr')).write_bytes(done.stderr)
        body=json.loads(response.read_text()) if response.exists() else {}
        if not body or collect:
            # In-progress agent_selfcheck responses are stdout-only even with --out.
            stdout=done.stdout.decode(errors='replace')
            for index,char in enumerate(stdout):
                if char!='{': continue
                try: candidate,_=json.JSONDecoder().raw_decode(stdout[index:])
                except ValueError: continue
                if isinstance(candidate,dict) and 'request_id' in candidate:
                    body=candidate;break
        report['requests'][mode]={'argv':argv,'response_path':str(response),'returncode':done.returncode,
            'request_id':body.get('request_id',previous.get('request_id') if previous else None),
            'status':body.get('status','complete' if 'all_pass' in body else 'unknown'),
            'all_pass':body.get('all_pass')}
        path.write_text(json.dumps(report,indent=2)+'\n')
        completed_failure=done.returncode==1 and body.get('all_pass') is False
        if done.returncode not in (0,3) and not completed_failure:
            raise RuntimeError('ordinary selfcheck failed; inspect saved stderr and response')
        if done.returncode==3 and not report['requests'][mode]['request_id']:
            raise RuntimeError('running selfcheck returned no resumable request identity')
    return report['requests']


def consume_replay(package,source,location,expected):
    """Observe the actual consumed terminal effects without changing emitted IR."""
    script=r'''
import sys,json,hashlib,re
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package),str(package/'compiler_library')]
import gemmini_conv
receipts=[];effects=[]
original=gemmini_conv.emit_conv
factory=gemmini_conv.make_conv_lowerer
def normalize(value):
    if value is None or type(value) in (str,int,float,bool):return value
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
    return re.sub(r'0x[0-9a-fA-F]+','<address>',str(value))
def bind(*args,**kwargs):
    lower=factory(*args,**kwargs)
    def consume(kind,fields):
        effects.append(dict(kind=kind,fields=normalize(fields)))
        return lower(kind,fields)
    return consume
def emit(*args,**kwargs):
    receipt=original(*args,**kwargs);receipts.append(normalize(receipt));return receipt
gemmini_conv.make_conv_lowerer=bind;gemmini_conv.emit_conv=emit
from gemmini_backend import compile_source
cb,llvm=compile_source(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_bytes((llvm+'\n').encode())
Path(sys.argv[4]).write_text(json.dumps(dict(consumed_schedules=receipts,terminal_effect_sites=effects),sort_keys=True,indent=2)+'\n')
'''
    replay=location/'replay.mlir';evidence=location/'consumed.json'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(package/'compiler_library'))
    done=subprocess.run([sys.executable,'-c',script,str(package),str(source),str(replay),str(evidence)],
        env=env,cwd=package,capture_output=True,timeout=120)
    (location/'replay.stderr').write_bytes(done.stderr)
    if done.returncode:raise RuntimeError('consumption replay failed: '+str(location/'replay.stderr'))
    if replay.read_bytes()!=expected:raise ValueError('observing consumed effects changed manifest bytes')
    body=json.loads(evidence.read_text())
    if not body['consumed_schedules']:raise ValueError('source did not consume a whole convolution schedule')
    return dict(effect_sha256=digest(body['terminal_effect_sites']),consumption_path=str(evidence),
        schedule_sha256=digest(body['consumed_schedules']),replay_byte_identical=True,
        effect_scope='static counted-loop terminal effect sites; no dynamic multiplicity or cycle inference')


def evaluate(report_path,*,collect=False):
    """Ordinary broker requests, then shared evaluation records for actual products.

    Input and provider contract identities are request pins, not claimed native
    input-blob custody. Consequently all timings remain explicitly unqualified.
    """
    from merlin.transforms.evaluated_exploration import ExplorationIdentity,EvaluationRequest,evaluate_emitted,CompletedObservation
    from merlin.transforms.pass_exploration import EmittedProgram
    path=Path(report_path).resolve()
    result=broker_evaluate(path,collect=collect)
    report=json.loads(path.read_text());requests=[];owners={}
    for row in report['emissions']:
        payload=Path(row['product_path']).read_bytes()
        if digest(payload)!=row['program_sha256']:raise ValueError('frozen target product changed')
        receipt=row['pipeline_receipt']
        if not receipt:raise ValueError('missing real pass pipeline receipt')
        variant=report['variants'][row['mode']];package=Path(variant['package'])
        domain=ExplorationIdentity(row['source_sha256'],digest((package/'manifest.yaml').read_bytes()),
            digest({'native_inputs_owner':'ordinary selfcheck','scope':report['scope']}),
            digest({'reference_owner':'ordinary selfcheck','scope':report['scope']}),
            digest((package/'data/rtl_facts.json').read_bytes()),tree_identity(package/'compiler_library'),
            digest({'provider':'ordinary broker','sim':report['scope']['sim']}),digest(report['scope']),
            digest({'roi':'complete ordinary capsule call','qualification':'pending custody and provider evidence'}),
            digest(receipt['after']['context']),'llvm-mlir','modeled_roi')
        request=EvaluationRequest(EmittedProgram(payload,'llvm-mlir',row['schedule_sha256']),receipt,domain)
        requests.append(request)
        original=Path(next(s['path'] for s in report['sources'] if s['source_sha256']==row['source_sha256']))
        capsule=original.parent.name if original.name.startswith('capsule.') else original.stem
        owners[request.key]=(row['mode'],capsule)
    completions=[]
    def owner(program,receipt,identity):
        # The broker is the sole external process owner. These observations do
        # not recycle partial timing, command counts, or queue wall time.
        mode,capsule=owners[(identity.digest(),digest(program.data))];status=result[mode]
        response=Path(status['response_path'])
        if not response.exists():return None
        feedback=response.read_bytes();body=json.loads(feedback)
        cases=body.get('per_capsule') or body.get('cases') or ()
        case=next((c for c in cases if c.get('capsule',c.get('name'))==capsule),None)
        if case is None:return None
        # Suite qualification (all_pass/complete) is independent of this
        # program's numerical screen. An incomplete oracle/timing tier cannot
        # turn the case's explicit L2 pass into a numerical failure.
        numeric='unknown'
        flag=case.get('numerical_execution_passed')
        tier=case.get('tiers',{}).get('L2')
        tier=tier.get('status') if isinstance(tier,dict) else tier
        if type(flag) is bool:numeric='pass' if flag else 'fail'
        elif tier in ('pass','fail'):numeric=tier
        elif case.get('status')=='pass':numeric='pass'
        return CompletedObservation(identity,digest(program.data),digest(feedback),
            numeric,'unknown',False,None)
    list(evaluate_emitted(requests,owner,max_in_flight=1,on_completion=lambda r,c:completions.append(c.record())))
    report['evaluation_completions']=completions
    report['identity_qualification']='source/manifest/library/LLVM and actual consumed effects pinned; native input blobs and cycle-provider custody remain ordinary broker owned and unqualified here'
    report['cycles']='UNKNOWN';report['selection']='retain off/1; no qualified matched complete-call timing'
    path.write_text(json.dumps(report,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--submission',default='submission');p.add_argument('--output',required=True)
    p.add_argument('--variant-prefix',required=True);p.add_argument('--train',action='append',required=True)
    p.add_argument('--held-out',action='append',required=True);p.add_argument('--capsules',required=True)
    p.add_argument('--suite',choices=('core','scalability','programs'),default='core');p.add_argument('--sim',choices=('spike','gsim','verilator','vcs'),default='spike')
    for command in ('evaluate','collect'):
        p=subs.add_parser(command);p.add_argument('--report',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        report=prepare(submission=args.submission,output=args.output,variant_prefix=args.variant_prefix,
            training=args.train,held_out=args.held_out,capsules=args.capsules,suite=args.suite,sim=args.sim)
        print(json.dumps({'report':str(Path(args.output).resolve()/'report.json'),'actual_emitted_changes':report['changes'],
            'selection':report['selection']},indent=2))
    else:print(json.dumps(evaluate(args.report,collect=args.command=='collect'),indent=2))

if __name__=='__main__':main()
