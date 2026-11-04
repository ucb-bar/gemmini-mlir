#!/usr/bin/env python3
"""Matched ordinary manifest products for shared convolution array retention.

The shared pass-exploration pipeline freezes default regional schedules with
array off/on; actual transformed target bytes and terminal consumption are
replayed before broker evaluation. No target timing is inferred from effects.
"""
import argparse,copy,json,time
from pathlib import Path
from gemmini_conv_evaluate import (digest,tree_identity,copy_package,source_record,
    manifest_emit,consume_replay,evaluate,WORKSPACE)

def prepare(*,submission,output,variant_prefix,training,held_out,capsules,suite='core',sim='spike',
            modes=('off','on')):
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
    if tuple(modes) not in (('off','on'),('auto','multirow')):
        raise ValueError('use matched off/on or incumbent auto/multirow alternatives')
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
        if mode not in declaration['knobs']['conv_array_retention']['values']:
            raise ValueError('array alternative is not declared')
        settings_path=variant/declaration['settings_file']
        settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings['conv_policy']='regional';settings['conv_array_retention']=mode
        settings_path.write_text(json.dumps(settings,indent=2)+'\n')
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
        state={'source':record,'mode':'off'}
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
        base=rows.get(modes[0])
        for mode in modes:
            if mode==modes[0]:continue
            changed.append(dict(source_sha256=record['source_sha256'],mode=mode,
                actual_emitted_change=None if base is None or mode not in rows else base['program_sha256']!=rows[mode]['program_sha256']))
    report=dict(schema='merlin.gemmini.ordinary_schedule_exploration.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=before,partition=partition.manifest(),sources=records,
        variants=variants,candidates=candidates,emissions=emitted,changes=changed,
        scope={'capsules':capsules,'suite':suite,'sim':sim},requests={},cycles='UNKNOWN',
        selection='retain regional array off; no matched target cycles yet',
        qualification='actual ordinary manifest emission; target numerics and cycles require selfcheck',
        compare_command='python .agents/skills/ab-library-pass/scripts/ab_compare.py ALTERNATIVE_RESPONSE BASELINE_RESPONSE')
    path=output/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for mode,variant in variants.items():
        successes=[r for r in emitted if r['mode']==mode]
        if len(successes)==len(records):
            (variant_root/('.library-pass-pending-'+Path(variant['package']).name)).unlink()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--submission',default='submission');q.add_argument('--output',required=True)
    q.add_argument('--variant-prefix',required=True);q.add_argument('--train',action='append',required=True)
    q.add_argument('--held-out',action='append',required=True);q.add_argument('--capsules',required=True)
    q.add_argument('--suite',default='scalability',choices=('core','scalability'));q.add_argument('--sim',default='spike')
    for command in ('evaluate','collect'):
        q=sub.add_parser(command);q.add_argument('--report',required=True)
    a=p.parse_args()
    if a.command=='prepare':
        r=prepare(submission=a.submission,output=a.output,variant_prefix=a.variant_prefix,training=a.train,
            held_out=a.held_out,capsules=a.capsules,suite=a.suite,sim=a.sim)
        print(json.dumps(dict(report=str(Path(a.output).resolve()/'report.json'),changes=r['changes']),indent=2))
    else:
        result=evaluate(a.report,collect=a.command=='collect')
        path=Path(a.report);report=json.loads(path.read_text())
        report['selection']='retain regional array off; paired target evidence remains pending independent qualification'
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
