#!/usr/bin/env python3
"""Production owner for heavy-pass-recipes/scripts/parallel_schedule_recipe.py.

Prepare freezes the ordinary compiler, original sources and declared knob overlays.
The exact requested recipe branches actual checked dense results, consumes each
result through the production encoder, checks byte equality with manifest stdout,
and deduplicates identical products. Only ready requests new ordinary selfchecks;
incumbents with no exact completed custody remain UNKNOWN, never ranked.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

PACKAGE=Path(__file__).resolve().parent
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'compiler_library')]
from gemmini_schedule_evaluate import digest,prepare as prepare_packages,tree_identity,evaluate as ordinary_evaluate

MODES={'original':'compact','batched':'panel_epochs','ready':'ready_readout','identical':'compact'}


def load_recipe(workspace):
    path=Path(workspace)/'.agents/skills/heavy-pass-recipes/scripts/parallel_schedule_recipe.py'
    spec=importlib.util.spec_from_file_location('gemmini_requested_parallel_recipe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


class Owner:
    """Actual source/effect/program owner; callbacks use ordinary broker leases."""
    def __init__(self,report,*,submit=False):
        self.path=Path(report).resolve();self.report=json.loads(self.path.read_text())
        self.output=self.path.parent;self.workspace=Path(self.report['workspace'])
        import gemmini_schedule_evaluate as ordinary_owner
        ordinary_owner.WORKSPACE=self.workspace
        self.submit=submit;self.lock=Lock();self.bindings={};self.domains={};self.programs={}
        self.ready_report=self.output/'ordinary-ready.json'
        ready=dict(self.report,variants={'ready_readout':self.report['variants']['ready_readout']},requests={})
        if not self.ready_report.exists():self.ready_report.write_text(json.dumps(ready,indent=2)+'\n')

    def binding(self,shape,caps):
        from gemmini_dense import plan_dense,geometry,FACTS_PATH
        from gemmini_dense_pipeline import plan_pipeline
        from gemmini_dense_readout import plan_ready_readout
        from gemmini_frontend import normalize_source
        from merlin.transforms.pipeline_state import PipelineContext
        matches=[r for r in self.report['sources'] if tuple(r['shapes'][0])==(shape[0],shape[1])
                 and tuple(r['shapes'][1])==(shape[1],shape[2])]
        if len(matches)!=1:raise ValueError('shape requires exactly one original source binding')
        record=matches[0];source=Path(record['frozen_path']).read_text();model=normalize_source(source)
        products=[o for o in model.get('operations',[]) if o['category']=='product']
        readouts=[o for o in model.get('operations',[]) if o['category']=='readout']
        if len(products)!=1 or products[0]['opcode'] not in ('MATMUL','MATMUL_RESIDENT') or len(readouts)!=1:
            raise ValueError('parallel owner currently admits a single rank-two source dense product and commit')
        attrs=readouts[0]['attributes']
        if attrs.get('output_dtype','i32')!='i32' or attrs.get('epilogue',[]) or any(k in attrs for k in ('scale','acc_scale','relu','bias')):
            raise ValueError('this exploration domain requires original raw i32 readout without initializer')
        readout=json.dumps({'dtype':'i32','scale':1.0,'relu':False},sort_keys=True)
        original,adapter,problem=plan_dense(*shape,readout_identity=readout)
        results={original.kernel.digest():('compact',None)}
        transformed={}
        def transform(result,mode):
            if result is not original:raise ValueError('registered branches require the bound source artifact')
            if mode in ('original','identical'):return result
            if mode not in transformed:
                if mode=='batched':candidate,evidence=plan_pipeline(result,adapter,problem,geometry(),FACTS_PATH,mode='panel_epochs')
                elif mode=='ready':candidate,evidence=plan_ready_readout(result,adapter,problem,geometry(),FACTS_PATH)
                else:raise ValueError(mode)
                candidate.revalidate();transformed[mode]=candidate
                results[candidate.kernel.digest()]=(MODES[mode],evidence)
            return transformed[mode]
        def emit(result):
            from gemmini_dense import consume_checked_dense
            from gemmini_backend import compile_source
            mode,evidence=results[result.kernel.digest()]
            # This exact artifact is passed into production, not replaced by a
            # recreated plan or merely mentioned in an evidence receipt.
            with consume_checked_dense(result,adapter,problem,mode=mode,readout_identity=readout) as consumed:
                _,lowered=compile_source(source)
            payload=(lowered+'\n').encode()
            ordinary=next(r for r in self.report['emissions'] if r['source_sha256']==record['source_sha256'] and r['mode']==mode)
            expected=Path(ordinary['product_path']).read_bytes()
            if payload!=expected:raise ValueError('consumed artifact differs from ordinary manifest product: '+mode)
            row=dict(shape=shape,split=record['split'],source_sha256=record['source_sha256'],
                original_effect_sha256=original.kernel.digest(),effect_sha256=result.kernel.digest(),
                program_sha256=digest(payload),mode=mode,consumed=consumed['consumed'],
                manifest_product_path=ordinary['product_path'],manifest_byte_equal=True,
                admission=evidence,compiler_sha256=self.report['variants'][mode]['compiler_sha256'])
            self.programs[(record['source_sha256'],digest(payload))]=row
            return payload.decode(),row
        binding=dict(source=original,transform=transform,emit=emit,shapes=((shape[0],shape[1]),(shape[1],shape[2])),
            context=PipelineContext(hardware=digest(FACTS_PATH.read_bytes()),resources=digest(asdict(geometry()[1])),
                numerical_policy=adapter.numerical_policy_identity,execution='ordinary-selfcheck:spike:scalability',
                library=tree_identity(PACKAGE/'compiler_library')),record=record,model=model)
        self.bindings[tuple(shape)]=binding
        return binding

    def identity(self,shape,binding):
        from merlin.transforms.evaluated_exploration import ExplorationIdentity,exploration_context_identity
        record=binding['record'];model=binding['model'];scope=self.report['scope']
        # Before broker execution these pin the original mathematical input and
        # reference declarations, not fabricated numerical input/reference bytes.
        declarations=dict(source_sha256=record['source_sha256'],logical_abi=model['logical_args'],scope=scope,
                          custody='original public source declarations; broker execution custody pending')
        identity=ExplorationIdentity(record['source_sha256'],digest(model['logical_args']),digest(declarations),
            digest(dict(reference='ordinary capsule mathematical source',**declarations)),
            digest((PACKAGE/'data/rtl_facts.json').read_bytes()),tree_identity(PACKAGE/'compiler_library'),
            digest((self.workspace/'agent_selfcheck.py').read_bytes()),digest(scope),
            digest(dict(roi='ordinary complete source kernel',source=record['source_sha256'],scope=scope)),
            exploration_context_identity(binding['context']),'llvm-mlir','modeled_roi')
        self.domains[identity.digest()]=binding
        (self.output/('identity-'+record['source_sha256']+'.json')).write_text(json.dumps(asdict(identity),indent=2)+'\n')
        return identity

    def hold_inputs(self,request):
        # Frozen package/source ownership lasts through recipe.run and broker
        # submission; the ordinary broker owns its immutable execution snapshot.
        return nullcontext()

    def evaluate(self,program,receipt,identity):
        from merlin.transforms.evaluated_exploration import CompletedObservation
        binding=self.domains[identity.digest()];row=self.programs[(identity.source_sha256,digest(program.data))]
        if Path(row['manifest_product_path']).read_bytes()!=program.data:
            raise ValueError('evaluation product custody changed')
        mode=row['mode']
        with self.lock:
            if self.submit and mode=='ready_readout':
                ready=json.loads(self.ready_report.read_text())
                if not ready['requests']:ordinary_evaluate(self.ready_report)
            request=json.loads(self.ready_report.read_text())['requests'].get('ready_readout') if mode=='ready_readout' else None
        feedback=dict(**row,request=request,numerics='UNKNOWN pending exact ordinary execution artifact custody',
            cycles='UNKNOWN',incumbent_reuse='none; no exact matching completed product evidence supplied',
            inputs='source declarations pinned; actual broker inputs/reference remain unqualified')
        feedback_path=self.output/('observation-'+identity.digest()+'-'+digest(program.data)+'.json')
        feedback_path.write_text(json.dumps(feedback,indent=2)+'\n')
        return CompletedObservation(identity,digest(program.data),digest(feedback_path.read_bytes()),'unknown','unknown',False,None)


def frozen_run(report,*,submit=False):
    owner=Owner(report,submit=submit);recipe,path=load_recipe(owner.workspace)
    shapes=[(r['shapes'][0][0],r['shapes'][0][1],r['shapes'][1][1]) for r in owner.report['sources']]
    with ThreadPoolExecutor(max_workers=2) as executor:
        result=recipe.run(shapes,{},executor=executor,max_in_flight=2,
            directory=owner.output/'parallel',owner=owner)
    evidence=dict(schema='merlin.gemmini.parallel_owner.v1',recipe_path=str(path),recipe_sha256=digest(path.read_bytes()),
        recipe_report=str(owner.output/'parallel/report.json'),ordinary_ready_report=str(owner.ready_report),
        products=list(owner.programs.values()),scope=owner.report['scope'],
        mapping=MODES,source_and_manifest_bound=True,selection='retain incumbent; UNKNOWN execution and cycles',
        identities={source:json.loads((owner.output/('identity-'+source+'.json')).read_text())
                    for source in {r['source_sha256'] for r in owner.programs.values()}},
        completion_count=len(result['completions']))
    (owner.output/'owner.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def collect_feedback(report_path):
    """Join only exact request-specific LLVM with ordinary numerical/model rows."""
    from merlin.transforms.evaluated_exploration import ExplorationIdentity,CompletedObservation
    output=Path(report_path).resolve().parent
    report=json.loads(Path(report_path).read_text());workspace=Path(report['workspace'])
    import gemmini_schedule_evaluate as ordinary_owner
    ordinary_owner.WORKSPACE=workspace
    ready_path=output/'ordinary-ready.json'
    ordinary_evaluate(ready_path,collect=True)
    request=json.loads(ready_path.read_text())['requests']['ready_readout']
    response_path=Path(request['response_path'])
    if not response_path.exists():
        return {'request':request,'observations':[],'selection':'retain incumbent; ordinary feedback pending'}
    response=json.loads(response_path.read_text())
    if response.get('selfcheck_request_id')!=request['request_id']:
        raise ValueError('broker response does not belong to the original ready request')
    owner=json.loads((output/'owner.json').read_text())
    observations=[]
    for product in owner['products']:
        if product['mode']!='ready_readout':continue
        source=next(r for r in report['sources'] if r['source_sha256']==product['source_sha256'])
        capsule=Path(source['path']).stem
        row=next((r for r in (response.get('per_capsule') or response.get('cases') or []) if r.get('capsule')==capsule),None)
        executed=workspace/'selfcheck_out/requests'/request['request_id']/capsule/'lowered.llvm.mlir'
        variant=report['variants']['ready_readout']
        native_export_match=executed.is_file() and digest(executed.read_bytes())==product['program_sha256']
        # Scalability may expose no LLVM copy. Its ordinary broker still owns
        # the immutable producer snapshot, whose unchanged source-to-manifest
        # equality was checked before request submission. Preserve that weaker
        # explicit custody label, rather than claim an exported execution file.
        frozen_owner_match=(tree_identity(Path(variant['package']))==variant['compiler_sha256']
            and digest(Path(product['manifest_product_path']).read_bytes())==product['program_sha256'])
        if row is None or not (native_export_match or frozen_owner_match):
            observations.append({'capsule':capsule,'qualification':'UNKNOWN: exact producer/execution custody unavailable'})
            continue
        # Hydrate only the exact content-addressed row named by this response.
        pointer=row.get('detail',{})
        if pointer.get('path'):
            detail_path=workspace/pointer['path']
            if detail_path.is_file() and digest(detail_path.read_bytes())==pointer.get('sha256'):
                detail=json.loads(detail_path.read_text())
                row={**detail,**row,'performance':{**row.get('performance',{}),**detail.get('performance',{})}}
        numerical=('pass' if row.get('numerical_execution_passed') is True or row.get('tiers',{}).get('L2')=='pass' else 'unknown')
        performance=row.get('performance',{})
        cycles=performance.get('estimated_whole_soc_cycles')
        pins=('selection_sha256','model_sha256','facts_sha256','comparison_domain_sha256','scope_sha256','capsule_sha256')
        completed=(numerical=='pass' and type(cycles) is int and cycles>0
            and performance.get('complete_event_accounting') is True
            and performance.get('status') in ('PROVISIONAL','QUALIFIED')
            and all(type(performance.get(k)) is str and len(performance[k])==64 for k in pins))
        identity_data=owner.get('identities',{}).get(product['source_sha256'])
        identity_file=output/('identity-'+product['source_sha256']+'.json')
        if identity_data is None and identity_file.exists():identity_data=json.loads(identity_file.read_text())
        if identity_data is None:
            observations.append({'capsule':capsule,'qualification':'UNKNOWN: original exploration identity unavailable'})
            continue
        observation=CompletedObservation(ExplorationIdentity(**identity_data),product['program_sha256'],
            digest(response_path.read_bytes()),numerical,'complete' if completed else 'unknown',completed,
            cycles if completed else None)
        observations.append(dict(capsule=capsule,observation=asdict(observation),eligible=observation.eligible,
            executed_program=str(executed) if native_export_match else None,
            custody='request-specific LLVM' if native_export_match else 'unchanged immutable ordinary producer + exact manifest bytes',
            comparison_identity={k:performance.get(k) for k in (*pins,'timing_mode','timing_epoch')},
            execution_digest=row.get('execution_digest'),
            timing_status=performance.get('status'),timing_qualified=performance.get('timing_qualified',False)))
    result=dict(request=request,observations=observations,
        selection='retain incumbent; require exact paired incumbent observations before schedule selection')
    (output/'collected-observations.json').write_text(json.dumps(result,indent=2)+'\n')
    result['joined_rankings']=join_completed_rankings(report_path)
    return result


def join_completed_rankings(report_path):
    """Rank exact recipe products only after native complete paired feedback.

    Reuse joins original source/program bytes and native model identities. It
    does not rewrite the recipe's original unknown completions or infer timing.
    """
    from merlin.transforms.pass_exploration import ObservedMetrics,rank_performance_variants
    output=Path(report_path).resolve().parent
    owner=json.loads((output/'owner.json').read_text())
    native=json.loads((output/'collected-observations.json').read_text())
    reuse_path=output/'incumbent-reuse.json'
    reuse=json.loads(reuse_path.read_text()).get('reused',[]) if reuse_path.exists() else []
    groups={}
    pins=('selection_sha256','model_sha256','facts_sha256','comparison_domain_sha256','scope_sha256','capsule_sha256','timing_mode','timing_epoch')
    for product in owner['products']:
        source=product['source_sha256'];program=product['program_sha256']
        group=groups.setdefault(source,dict(source_sha256=source,shape=product['shape'],rows=[],unranked=[]))
        completed=next((r for r in native['observations'] if r.get('observation',{}).get('program_sha256')==program
            and r['observation']['identity']['source_sha256']==source),None)
        previous=next((r for r in reuse if r['source_sha256']==source and r['program_sha256']==program),None)
        if completed and completed.get('eligible'):
            group['rows'].append(dict(mode=product['mode'],program_sha256=program,
                cycles=completed['observation']['cycles'],comparison_identity=[completed['comparison_identity'].get(k) for k in pins],
                feedback_sha256=completed['observation']['feedback_sha256']))
        elif previous and previous.get('numerical_passed') and previous.get('cycles'):
            if digest(Path(previous['response_path']).read_bytes())!=previous['response_sha256']:
                raise ValueError('original reused native response changed')
            group['rows'].append(dict(mode=product['mode'],program_sha256=program,cycles=previous['cycles'],
                comparison_identity=previous['comparison_identity'],feedback_sha256=previous['response_sha256']))
        else:
            group['unranked'].append(dict(mode=product['mode'],program_sha256=program,
                reason='UNKNOWN incomplete or missing native complete same-ROI timing'))
    records=[]
    for group in groups.values():
        rows=group['rows'];domains={digest(row['comparison_identity']) for row in rows}
        paired=len(rows)>=2 and len(domains)==1 and any(r['mode']=='compact' for r in rows)
        ranked=rank_performance_variants(tuple((r['mode'],ObservedMetrics('modeled_roi',next(iter(domains)),
            (('cycles',r['cycles']),))) for r in rows)) if paired else None
        records.append(dict(**group,ranking=ranked,selection='exact paired provisional model comparison' if paired
            else 'retain incumbent; missing complete exact incumbent pair'))
    result=dict(schema='gemmini.parallel.completed_join.v1',rankings=records,
        recipe_report=str(output/'parallel/report.json'),
        qualification='same source and product bytes; original native paired model pins; provisional model only')
    (output/'joined-rankings.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def join_reused(report_path,previous_reports):
    """Reuse only an unchanged frozen owner and exact source/program match."""
    path=Path(report_path).resolve();report=json.loads(path.read_text());workspace=Path(report['workspace'])
    recipe,_=load_recipe(workspace)
    compare=recipe.load(workspace/'.agents/skills/ab-library-pass/scripts/ab_compare.py','gemmini_exact_compare')
    reused=[]
    for previous in previous_reports:
        previous=Path(previous).resolve();prior=json.loads(previous.read_text())
        for product in report['emissions']:
            mode=product['mode'];variant=prior['variants'].get(mode)
            if variant is None or tree_identity(Path(variant['package']))!=variant['compiler_sha256']:continue
            old=next((r for r in prior['emissions'] if r['source_sha256']==product['source_sha256']
                and r['mode']==mode and r['program_sha256']==product['program_sha256']),None)
            request=prior.get('requests',{}).get(mode)
            if old is None or request is None:continue
            if digest(Path(old['product_path']).read_bytes())!=product['program_sha256']:continue
            response_path=Path(request['response_path'])
            if not response_path.exists():continue
            response=compare._load(response_path)
            if response.get('selfcheck_request_id')!=request['request_id']:continue
            source=next(r for r in report['sources'] if r['source_sha256']==product['source_sha256'])
            capsule=Path(source['path']).stem;row=compare.rows(response).get(capsule)
            if row is None:continue
            evidence=compare.evidence(row).get('modeled')
            reused.append(dict(capsule=capsule,mode=mode,source_sha256=product['source_sha256'],
                program_sha256=product['program_sha256'],frozen_owner=str(variant['package']),
                response_path=str(response_path),response_sha256=digest(response_path.read_bytes()),
                request_id=request['request_id'],numerical_passed=compare.passed(row),
                cycles=None if evidence is None else evidence[0],comparison_identity=None if evidence is None else evidence[1],
                evidence=None if evidence is None else evidence[2],
                qualification='ordinary immutable owner; source and manifest products match exactly; provisional model evidence'))
    result=dict(reused=reused,selection='retain incumbent pending boundary panel; provisional model results only')
    (path.parent/'incumbent-reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--submission',default='submission');p.add_argument('--output',required=True)
    p.add_argument('--variant-prefix',required=True);p.add_argument('--train',action='append',required=True)
    p.add_argument('--held-out',action='append',required=True);p.add_argument('--capsules',required=True)
    p.add_argument('--suite',choices=('core','scalability'),default='scalability');p.add_argument('--sim',choices=('spike',),default='spike')
    p.add_argument('--submit-ready',action='store_true')
    p=commands.add_parser('_run');p.add_argument('--report',required=True);p.add_argument('--submit-ready',action='store_true')
    p=commands.add_parser('collect');p.add_argument('--report',required=True)
    p=commands.add_parser('join');p.add_argument('--report',required=True);p.add_argument('--reuse-report',action='append',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        report=prepare_packages(submission=args.submission,output=args.output,variant_prefix=args.variant_prefix,
            training=args.train,held_out=args.held_out,capsules=args.capsules,suite=args.suite,sim=args.sim,
            modes=('compact','panel_epochs','ready_readout'))
        report['workspace']=str(PACKAGE.parent)
        path=Path(args.output).resolve()/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
        argv=[sys.executable,str(path.parent/'compiler/gemmini_parallel_explore.py'),'_run','--report',str(path)]
        if args.submit_ready:argv+=['--submit-ready']
        subprocess.run(argv,check=True)
    elif args.command=='_run':
        result=frozen_run(args.report,submit=args.submit_ready)
        print(json.dumps({k:result[k] for k in ('recipe_report','ordinary_ready_report','completion_count','selection')},indent=2))
    elif args.command=='join':print(json.dumps(join_reused(args.report,args.reuse_report),indent=2))
    else:
        print(json.dumps(collect_feedback(args.report),indent=2))

if __name__=='__main__':main()
