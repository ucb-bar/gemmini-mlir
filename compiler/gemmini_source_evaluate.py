#!/usr/bin/env python3
"""Freeze ordinary source-composition A/B programs and collect owner evidence.

Requests contain preferences only. Membership, numerical boundaries, storage
permissions and actual application remain with the original program owner.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'compiler_library')]
from gemmini_conv_evaluate import WORKSPACE,copy_package,digest,tree_identity

SCHEMA='merlin.source-composition-preferences.v1'
PROGRAMS=('PG_source_attention_cache_representative',
          'PG_source_repeated_inference_representative','PG_source_residual_block_medium')


def _write(path,body):
    Path(path).write_text(json.dumps(body,sort_keys=True,indent=2)+'\n')


def prepare(*,submission,output,prefix,programs=PROGRAMS,heads=False):
    source=Path(submission).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('use a fresh experiment directory outside the compiler')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',prefix):
        raise ValueError('invalid variant prefix')
    suite=json.loads((WORKSPACE/'public-program-sources/suite.json').read_text())
    roster={row['name']:row for row in suite['members']}
    # Only explicit public program directories can enter the experiment.
    records=[]
    for name in programs:
        if not re.fullmatch(r'PG_source_[A-Za-z0-9_]+',name):
            raise ValueError('invalid public program name')
        if name not in roster:raise ValueError('program is outside the selected public suite')
        directory=WORKSPACE/'public-program-sources'/name
        paths=[directory/'capsule.interface.mlir',directory/'capsule.pytorch.py',directory/'profile.json']
        if not all(p.is_file() for p in paths):raise ValueError('missing public program '+name)
        members={p.name:digest(p.read_bytes()) for p in paths}
        for member in roster[name]['sources'].values():
            if members[Path(member['path']).name]!=member['sha256']:
                raise ValueError('public source differs from selected suite identity')
        records.append(dict(program=name,members=members))
    output.mkdir(parents=True)
    before=tree_identity(source);frozen=output/'compiler';copy_package(source,frozen)
    if tree_identity(source)!=before or tree_identity(frozen)!=before:
        raise ValueError('compiler changed during snapshot; retry with a fresh output')
    request_path=frozen/'source_composition.json'
    incumbent_request=json.loads(request_path.read_text()) if request_path.exists() else None
    if incumbent_request is not None and incumbent_request.get('schema')!=SCHEMA:
        raise ValueError('incumbent source composition schema differs')
    # Preserve all incumbent host/numeric choices. Each alternative requests
    # only its stated preparation/storage delta under the original owner.
    bundles={'baseline':incumbent_request,'preparation_storage':dict(incumbent_request or {},schema=SCHEMA,
        shared_input_siblings=False,prepack_weights=True,
        share_exact_tensor_preparations=True,dense_input_views=True,
        reuse_intermediate_storage=True)}
    if heads:
        bundles['heads_siblings']={**bundles['preparation_storage'],
            'independent_heads':True,'shared_input_siblings':True}
    variants={}
    for mode,request in bundles.items():
        path=WORKSPACE/'variants'/(prefix+'-'+mode)
        if path.exists():raise ValueError('variant already exists: '+str(path))
        copy_package(frozen,path)
        request_path=path/'source_composition.json'
        if request_path.exists():request_path.unlink()
        if request is not None:_write(request_path,request)
        variants[mode]=dict(path=str(path),compiler_sha256=tree_identity(path),request=request,
                            request_sha256=digest(request_path.read_bytes()) if request else None)
    # Audit every non-request source byte across comparison arms.
    def compiler_members(path):
        return {str(p.relative_to(path)):digest(p.read_bytes()) for p in path.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name!='source_composition.json'}
    common=compiler_members(Path(variants['baseline']['path']))
    for item in variants.values():
        if compiler_members(Path(item['path']))!=common:
            raise ValueError('comparison changes compiler bytes outside source request')
    report=dict(schema='gemmini.source_composition_experiment.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=before,incumbent_request=incumbent_request,
        common_compiler_members_sha256=digest(common),
        public_suite_sha256=digest((WORKSPACE/'public-program-sources/suite.json').read_bytes()),programs=records,variants=variants,requests={},
        hypothesis='exact preparation sharing and proven final allocation reuse reduce completed repeated-program latency',
        selection='retain baseline; UNKNOWN cycles remain unranked',observations={},comparisons=[])
    _write(output/'report.json',report)
    return report


def _run(argv,path):
    result=subprocess.run(argv,cwd=WORKSPACE,text=True,capture_output=True,timeout=45)
    Path(str(path)+'.stdout').write_text(result.stdout)
    Path(str(path)+'.stderr').write_text(result.stderr)
    try:body=json.loads(result.stdout)
    except ValueError:raise RuntimeError('ordinary checker returned no JSON; inspect '+str(path)+'.stdout')
    if result.returncode and not body.get('request_id') and not body.get('selfcheck_request_id'):
        raise RuntimeError('ordinary checker refused request: '+json.dumps(body))
    return body


def evaluate(path,*,collect=False):
    path=Path(path).resolve();report=json.loads(path.read_text());directory=path.parent
    for program in report['programs']:
        for name,expected in program['members'].items():
            member=WORKSPACE/'public-program-sources'/program['program']/name
            if digest(member.read_bytes())!=expected:
                raise ValueError('public source changed since preparation: '+str(member))
    for mode,item in report['variants'].items():
        variant=Path(item['path'])
        if tree_identity(variant)!=item['compiler_sha256']:
            raise ValueError('immutable variant changed: '+mode)
        if not collect and mode not in report['requests']:
            argv=[sys.executable,str(WORKSPACE/'agent_selfcheck.py'),'--submission',
                  str(variant.relative_to(WORKSPACE)),'--suite','programs','--sim','spike',
                  '--capsules',','.join(p['program'] for p in report['programs']),
                  '--wait-budget','5']
            body=_run(argv,directory/(mode+'-submit'))
            rid=body.get('request_id') or body.get('selfcheck_request_id')
            if not rid:raise ValueError('ordinary checker did not return request identity')
            report['requests'][mode]=dict(request_id=rid,argv=argv,response=body)
            _write(path,report)
    if collect:
        for mode,request in report['requests'].items():
            rid=request['request_id'];response=WORKSPACE/'.qa_channel'/('resp_'+rid+'.json')
            if not response.is_file():
                report['observations'][mode]=dict(status='pending',request_id=rid);continue
            body=json.loads(response.read_text())
            expected_submission=str(Path(report['variants'][mode]['path']).relative_to(WORKSPACE))
            if body.get('selfcheck_submission')!=expected_submission:
                raise ValueError('owner response names another immutable variant')
            if (body.get('requested_submission_sha256') and body.get('submission_sha256')
                    and body['requested_submission_sha256']!=body['submission_sha256']):
                raise ValueError('owner executed another compiler snapshot')
            # Preserve actual owner response; never replace it by a catalog plan.
            saved=directory/(mode+'-owner-response.json');_write(saved,body)
            effects=[];cycles=[];identities=[];case_outcomes=[]
            def walk(value,location='$'):
                if isinstance(value,dict):
                    composition=value.get('source_composition')
                    if (isinstance(composition,dict) and composition.get('schema')=='merlin.source_composition_feedback.v1'
                            and isinstance(value.get('capsule'),str)):
                        case_outcomes.append(dict(capsule=value['capsule'],status=value.get('status'),
                            tiers=value.get('tiers'),source_composition=composition,performance=value.get('performance')))
                    for key,member in value.items():
                        place=location+'.'+key
                        if (key=='source_composition' and isinstance(member,dict)
                                and member.get('schema')=='merlin.source_composition_feedback.v1'):
                            effects.append(dict(path=place,evidence=member))
                        if key in ('estimated_whole_soc_cycles','barrier_cycles','complete_call_cycles'):
                            cycles.append(dict(path=place,value=member))
                        if key.endswith('sha256') and isinstance(member,str):
                            identities.append(dict(path=place,value=member))
                        if key=='detail' and isinstance(member,dict) and isinstance(member.get('path'),str):
                            detail=(WORKSPACE/member['path']).resolve()
                            if detail.is_relative_to(WORKSPACE) and detail.is_file():
                                payload=detail.read_bytes()
                                if member.get('sha256') and digest(payload)!=member['sha256']:
                                    raise ValueError('owner detail identity mismatch: '+str(detail))
                                walk(json.loads(payload),place+'.hydrated')
                        walk(member,place)
                elif isinstance(value,list):
                    for i,member in enumerate(value):walk(member,location+'['+str(i)+']')
            walk(body)
            report['observations'][mode]=dict(request_id=rid,response_path=str(saved),
                response_sha256=digest(body),all_pass=body.get('all_pass',body.get('passed')),
                actual_source_composition=effects,case_outcomes=case_outcomes,cycle_observations=cycles,
                identity_observations=identities,
                application_status='owner_evidence_observed' if effects else 'application_evidence_unavailable',
                cycle_rank='UNKNOWN; requires matched complete-call scope and product/source identities')
        # Deliberately no automatic ranking of arbitrary counter ticks or partial
        # operation measurements. Owner outcomes and exact bytes remain reviewable.
        comparisons=[]
        observed=report['observations']
        baseline={r['capsule']:r for r in observed.get('baseline',{}).get('case_outcomes',[])}
        for mode in report['variants']:
            if mode=='baseline':continue
            candidate={r['capsule']:r for r in observed.get(mode,{}).get('case_outcomes',[])}
            for program in report['programs']:
                name=program['program'];a=baseline.get(name);b=candidate.get(name)
                row=dict(baseline='baseline',candidate=mode,program=name,status='unranked')
                pa=(a or {}).get('performance') or {};pb=(b or {}).get('performance') or {}
                fields=('capsule_sha256','facts_sha256','comparison_domain_sha256','model_sha256','scope_sha256','scope')
                usable=(a and b and a.get('status')=='pass' and b.get('status')=='pass'
                    and all(p.get('status') in ('PROVISIONAL','QUALIFIED')
                        and p.get('complete_event_accounting') is True and p.get('elf_sha256')
                        and isinstance(p.get('estimated_whole_soc_cycles'),(int,float))
                        and not isinstance(p['estimated_whole_soc_cycles'],bool)
                        and math.isfinite(p['estimated_whole_soc_cycles'])
                        and p['estimated_whole_soc_cycles']>0 for p in (pa,pb))
                    and all(pa.get(k) and pa.get(k)==pb.get(k) for k in fields))
                if usable:
                    old=pa['estimated_whole_soc_cycles'];new=pb['estimated_whole_soc_cycles']
                    row.update(status='provisional_modeled_comparison',baseline_cycles=old,
                        candidate_cycles=new,candidate_over_baseline=new/old,
                        basis='same model/facts/source/complete ROI; modeled, not measured speedup')
                else:row['reason']='UNKNOWN or unmatched complete-call/model/source/executable evidence'
                comparisons.append(row)
        report['comparisons']=comparisons
        _write(path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--submission',default='submission');p.add_argument('--output',required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--program',action='append');p.add_argument('--heads',action='store_true')
    for name in ('evaluate','collect'):
        p=commands.add_parser(name);p.add_argument('--report',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='prepare':
        args['programs']=args.pop('program') or PROGRAMS
        report=prepare(**args)
        print(json.dumps(dict(report=str(Path(args['output']).resolve()/'report.json'),variants=report['variants']),indent=2))
    else:
        report=evaluate(args['report'],collect=command=='collect')
        print(json.dumps(dict(requests=report['requests'],observations=report['observations'],comparisons=report['comparisons']),indent=2))

if __name__=='__main__':main()
