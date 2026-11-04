#!/usr/bin/env python3
"""Explore real compiler schedules, then request matched ordinary selfchecks.

Examples (run from workspace):
  python submission/gemmini_schedule_evaluate.py prepare --output ab/dense-search \
    --variant-prefix dense-search --train layers/G_matmul_m1k48n64_raw/capsule.interface.mlir \
    --held-out layers/SY_int_mm_m3_k16_n4_8da67d4209/capsule.linalg.mlir \
    --capsules G_matmul_m1k48n64_raw,SY_int_mm_m3_k16_n4_8da67d4209
  python submission/gemmini_schedule_evaluate.py evaluate --report ab/dense-search/report.json
  python submission/gemmini_schedule_evaluate.py collect --report ab/dense-search/report.json

No program counts rank a candidate. Preparation records actual manifest-emitted
bytes and frozen source/compiler identities. Evaluation uses the ordinary broker
and identical capsule/suite/simulator scope for every frozen variant. Collection
resumes each request once; unknown cycles stay unknown. Training and holdout are
frozen before settings exploration and cannot alias in source geometry.
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
    operations=model['operations'] if model['grammar']=='merlin_iface' else model['inventory']['ops']
    products=[op for op in operations if op.get('category')=='product' or op.get('family')=='contraction']
    canonical=[]
    for operation in products:
        extents=operation.get('extents',{})
        if not all(key in extents for key in ('m','k','n')):
            raise ValueError('exploration requires canonical dense source axes')
        batch=tuple(operation.get('batch_shape',()))
        canonical.extend(((*batch,extents['m'],extents['k']),(*batch,extents['k'],extents['n'])))
    if not canonical: raise ValueError('source contains no dense contraction')
    shapes=tuple(canonical)
    geometry=ExplorationGeometry('dense',shapes)
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
            modes=None):
    """Return a reviewable experiment using PassFamily and actual target bytes."""
    from merlin.transforms.pass_exploration import (PassFamily,EmittedProgram,ExplorationRefused,
        ValidationPartition,generate_candidates,explore_candidates)
    from merlin.transforms.pipeline_policy import RegisteredPass,PolicyFacts
    from merlin.transforms.pipeline_state import PassSpec,PipelineContext
    submission=Path(submission).resolve();output=Path(output).resolve()
    if modes is None:
        modes=tuple(json.loads((submission/'TUNING_KNOBS.json').read_text())['knobs']['dense_schedule']['values'])
    if output.exists(): raise ValueError('use a new experiment output directory')
    if output.is_relative_to(submission): raise ValueError('experiment output must be outside submission')
    if (not variant_prefix or not variant_prefix[0].isalnum() or len(variant_prefix)>40
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in variant_prefix)):
        raise ValueError('variant prefix must start with a letter or digit, use letters/digits/underscore/hyphen, and fit 40 characters')
    if not modes or set(modes)-{'auto','auto_ready','compact','panel_drained','panel_epochs','ready_readout','segmented_ready'}:
        raise ValueError('use only implemented schedule alternatives; first mode is the paired baseline')
    baseline=tuple(modes)[0]
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
        if mode not in declaration['knobs']['dense_schedule']['values']:
            raise ValueError('dense_schedule alternative is not declared')
        settings_path=variant/declaration['settings_file']
        settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings['dense_schedule']=mode;settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        variants[mode]={'package':str(variant),'compiler_sha256':tree_identity(variant),'settings':settings}
    emitted=[]
    def select(knobs):
        def action(session):
            session.module['mode']=knobs['mode'];return {'settings':knobs['mode']}
        return RegisteredPass(PassSpec('dense-settings',0,produces=('settings',),ir_level='authored_source'),action)
    def emit(_):
        def action(session):
            mode=session.module['mode'];variant=Path(variants[mode]['package']);source=session.module['source']
            location=output/'products'/source['source_sha256']/mode
            try:product,receipt=manifest_emit(variant,Path(source['frozen_path']),location)
            except RuntimeError as exc:
                raise ExplorationRefused(str(exc),layer='emission') from exc
            emitted.append(dict(mode=mode,split=source['split'],source_sha256=source['source_sha256'],
                geometry_sha256=source['geometry_sha256'],compiler_sha256=variants[mode]['compiler_sha256'],**receipt))
            # Artifact is the actual ordinary emitter stdout, with its source and settings identity.
            return {'program':{'data':product,'schedule_sha256':digest({'source':source['source_sha256'],
                'settings':variants[mode]['settings'],'compiler':variants[mode]['compiler_sha256']})}}
        return RegisteredPass(PassSpec('ordinary-manifest-emitter',1,requires=('settings',),produces=('program',),
            mutates_ir=False,ir_level='authored_source'),action)
    space=generate_candidates((PassFamily('dense-settings',select,(('mode',tuple(modes)),)),
        PassFamily('ordinary-manifest-emitter',emit)),output='program',max_candidates=16)
    candidates=[]
    context=PipelineContext(hardware=digest((frozen/'data/rtl_facts.json').read_bytes()),
        numerical_policy=digest((frozen/'data/target_contract.yaml').read_bytes()),
        execution='ordinary-manifest-compiler;target-numerics-and-cycles-pending',library=tree_identity(frozen/'compiler_library'))
    for record in records:
        state={'source':record,'mode':baseline}
        for row in explore_candidates(state,space,lambda s:PolicyFacts(s.stamp(),shapes=tuple(map(tuple,record['shapes']))),
                output='program',emission=lambda r:EmittedProgram(r.artifact['data'],'llvm-mlir',r.artifact['schedule_sha256']),
                context=context,ir_level='authored_source',clone=copy.deepcopy,fingerprint=digest,
                verify_ir=lambda m:None):
            candidates.append({'source_sha256':record['source_sha256'],'split':record['split'],**row.summary()})
    changed=[]
    for record in records:
        rows={r['mode']:r for r in emitted if r['source_sha256']==record['source_sha256']}
        base=rows.get(baseline)
        for mode in modes:
            if mode==baseline:continue
            changed.append(dict(source_sha256=record['source_sha256'],mode=mode,baseline=baseline,
                actual_emitted_change=None if base is None or mode not in rows else base['program_sha256']!=rows[mode]['program_sha256']))
    report=dict(schema='merlin.gemmini.ordinary_schedule_exploration.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=before,partition=partition.manifest(),sources=records,
        variants=variants,candidates=candidates,emissions=emitted,changes=changed,
        scope={'capsules':capsules,'suite':suite,'sim':sim},requests={},cycles='UNKNOWN',
        baseline=baseline,selection='retain '+baseline+'; no matched target cycles yet',
        qualification='actual ordinary manifest emission; target numerics and cycles require selfcheck',
        compare_command='python .agents/skills/ab-library-pass/scripts/ab_compare.py ALTERNATIVE_RESPONSE BASELINE_RESPONSE')
    path=output/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for mode,variant in variants.items():
        successes=[r for r in emitted if r['mode']==mode]
        if len(successes)==len(records):
            (variant_root/('.library-pass-pending-'+Path(variant['package']).name)).unlink()
    return report


def evaluate(report_path,*,collect=False):
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--submission',default='submission');p.add_argument('--output',required=True)
    p.add_argument('--variant-prefix',required=True);p.add_argument('--train',action='append',required=True)
    p.add_argument('--held-out',action='append',required=True);p.add_argument('--capsules',required=True)
    p.add_argument('--suite',choices=('core','scalability','programs'),default='core');p.add_argument('--sim',choices=('spike','gsim','verilator','vcs'),default='spike')
    p.add_argument('--mode',action='append',help='explicit dense mode; otherwise explore all declared dense_schedule values')
    for command in ('evaluate','collect'):
        p=subs.add_parser(command);p.add_argument('--report',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        report=prepare(submission=args.submission,output=args.output,variant_prefix=args.variant_prefix,
            training=args.train,held_out=args.held_out,capsules=args.capsules,suite=args.suite,sim=args.sim,modes=args.mode)
        print(json.dumps({'report':str(Path(args.output).resolve()/'report.json'),'actual_emitted_changes':report['changes'],
            'selection':report['selection']},indent=2))
    else:print(json.dumps(evaluate(args.report,collect=args.command=='collect'),indent=2))

if __name__=='__main__':main()
