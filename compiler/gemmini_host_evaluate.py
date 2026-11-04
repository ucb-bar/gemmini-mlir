#!/usr/bin/env python3
"""Freeze ordinary host-optimization variants and compare complete target calls.

Only exact owner artifacts permit baseline reuse. Missing timing, utilization,
or source ownership never becomes a speedup claim or an automatic default edit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'compiler_library')]
from gemmini_conv_evaluate import digest, tree_identity, copy_package, manifest_emit
from gemmini_pool_evaluate import _body


def command_buffer(package, source, location):
    import yaml
    manifest = yaml.safe_load((package/'manifest.yaml').read_text())
    output = location/'command_buffer.json'
    bindings = dict(tool=str(package/manifest['entrypoints']['tool']), input_mlir=str(source), output_json=str(output))
    argv = [arg.format(**bindings) for arg in manifest['commands']['emit_command_buffer']['argv']]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package/'compiler_library'))
    run = subprocess.run(argv, cwd=package, env=env, capture_output=True, timeout=120)
    (location/'command_buffer.stderr').write_bytes(run.stderr)
    if run.returncode or not output.exists():
        raise RuntimeError('ordinary command-buffer emission failed: '+str(location))
    return json.loads(output.read_text()), dict(command_buffer_path=str(output), command_buffer_sha256=digest(output.read_bytes()), command_buffer_argv=argv)


def consume_replay(package, source, location, expected, expected_cb, mode, mechanism='host_pointwise_fusion'):
    script = r'''
import sys,json,re
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package),str(package/'compiler_library')]
import gemmini_encode as encoder
effects=[]
def normalize(v):
 if v is None or type(v) in (str,int,float,bool):return v
 if isinstance(v,dict):return {str(k):normalize(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [normalize(x) for x in v]
 return re.sub(r'0x[0-9a-fA-F]+','<address>',str(v))
original=encoder.insn
def terminal(fn,funct,rs1,rs2):
 effects.append(dict(funct=funct,rs1=normalize(rs1),rs2=normalize(rs2)))
 return original(fn,funct,rs1,rs2)
encoder.insn=terminal
from gemmini_backend import compile_source
cb,llvm=compile_source(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_bytes((llvm+'\n').encode())
Path(sys.argv[4]).write_text(json.dumps(dict(command_buffer=cb,terminal_effect_sites=effects),sort_keys=True,indent=2)+'\n')
'''
    replay, evidence = location/'replay.mlir', location/'consumed.json'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package/'compiler_library'))
    run = subprocess.run([sys.executable, '-c', script, str(package), str(source), str(replay), str(evidence)],
                         cwd=package, env=env, capture_output=True, timeout=120)
    (location/'replay.stderr').write_bytes(run.stderr)
    if run.returncode:
        raise RuntimeError('consumption replay failed: '+str(location/'replay.stderr'))
    if replay.read_bytes() != expected:
        raise ValueError('observing host-fusion consumption changed ordinary emitter bytes')
    body = json.loads(evidence.read_text())
    if body['command_buffer'] != expected_cb:
        raise ValueError('ordinary command buffer differs from consumed emitter command buffer')
    reports = expected_cb['params'].get(mechanism, [])
    rewrites = sum(row['rewrites'] for row in reports) if mechanism == 'host_pointwise_fusion' else len(reports)
    if (mode == 'on') != bool(rewrites):
        raise ValueError('requested host optimization mode not actually consumed: '+mechanism)
    plan = expected_cb['params']['global_program_plan']
    return dict(consumption_path=str(evidence), replay_byte_identical=True,
                command_buffer_replay_identical=True, effect_sha256=digest(body['terminal_effect_sites']),
                effect_scope='static actual encoder sites; no dynamic multiplicity inference',
                consumed_fusions=reports if mechanism == 'host_pointwise_fusion' else [],
                consumed_optimizations=reports, consumed_rewrites=rewrites, source_sha256=plan['source_sha256'],
                source_plan_sha256=digest(plan))


def prepare(*, submission, output, variant_prefix, source, capsule, expected_off=None, expected_on=None,
            reuse_request=None, reuse_artifact=None, mechanism='host_pointwise_fusion'):
    if mechanism not in ('host_pointwise_fusion','host_tensor_views'):
        raise ValueError('unsupported host optimization comparison')
    submission, output, source = map(lambda p: Path(p).resolve(), (submission, output, source))
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('use a fresh experiment directory outside submission')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', variant_prefix):
        raise ValueError('invalid variant prefix')
    members = json.loads((WORKSPACE/'PUBLIC_CAPABILITIES.json').read_text())['members']
    member = members[capsule]
    raw = source.read_bytes()
    if member['selfcheck_suite'] != 'scalability' or member['source_analysis']['sha256'] != digest(raw):
        raise ValueError('source bytes/suite differ from public declaration')
    output.mkdir(parents=True)
    frozen_source = output/'source.mlir'; frozen_source.write_bytes(raw)
    parent = tree_identity(submission)
    frozen = output/'compiler'; copy_package(submission, frozen)
    if tree_identity(frozen) != parent or tree_identity(submission) != parent:
        raise ValueError('compiler changed during snapshot; retry a fresh experiment')
    variants, emissions = {}, {}
    for mode, expected_path in (('off', expected_off), ('on', expected_on)):
        package = WORKSPACE/'variants'/(variant_prefix+'-'+mode)
        if package.exists():
            raise ValueError('variant already exists')
        copy_package(frozen, package)
        declaration = json.loads((package/'TUNING_KNOBS.json').read_text())
        if declaration['knobs'][mechanism]['values'] != ['off', 'on']:
            raise ValueError('host optimization knob is not ordinarily declared')
        settings_path = package/declaration['settings_file']
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        settings[mechanism] = mode
        settings_path.write_text(json.dumps(settings, indent=2)+'\n')
        variants[mode] = dict(package=str(package), compiler_sha256=tree_identity(package), settings=settings)
        location = output/'products'/mode
        product, receipt = manifest_emit(package, frozen_source, location)
        native = None
        if expected_path:
            expected = Path(expected_path).read_bytes()
            if product not in (expected, expected+b'\n'):
                raise ValueError('ordinary emitter differs from native-checked expected '+mode+' bytes')
            native = dict(path=str(Path(expected_path).resolve()), sha256=digest(expected),
                          ordinary_stdout_added_newline=(product == expected+b'\n'))
        cb, cb_receipt = command_buffer(package, frozen_source, location)
        replay = consume_replay(package, frozen_source, location, product, cb, mode, mechanism)
        if replay['source_sha256'] != digest(raw):
            raise ValueError('emitter lost original source identity')
        emissions[mode] = dict(**receipt, **cb_receipt, **replay, native_expected=native)
    # Preserve all original source and task observations, allowing instruction
    # extents to change with the actual emitted implementation.
    plans = [json.loads(Path(emissions[m]['command_buffer_path']).read_text())['params']['global_program_plan'] for m in ('off','on')]
    for key in ('source_sha256','source_op_count','source_values','entry_bindings','output_bindings'):
        if plans[0][key] != plans[1][key]: raise ValueError('fusion changed source observation '+key)
    for off, on in zip(plans[0]['tasks'], plans[1]['tasks'], strict=True):
        for key in ('kind','source_op_indices','source_region_ids','reads','writes'):
            if off[key] != on[key]: raise ValueError('fusion changed task ownership '+key)
    report = dict(schema='merlin.gemmini.host_pointwise_evaluation.v1', mechanism=mechanism, created_unix_ns=time.time_ns(),
        source=dict(path=str(source), frozen_path=str(frozen_source), source_sha256=digest(raw), capsule=capsule, suite='scalability'),
        parent_compiler_sha256=parent, frozen_compiler_sha256=tree_identity(frozen),
        variants=variants, emissions=emissions, requests={}, reuse=None,
        pass_order=(['private_equal_domain_pointwise_chains','shared_live_float_integer_pointwise']
                    if mechanism == 'host_pointwise_fusion' else ['shared_yield_only_affine_projection_admission','readonly_descriptor'])
                    +['shared_compact_host_lowering','ordinary_manifest_emitter'],
        source_ownership_identical=True, comparison_scope='complete kernel call; identical original source and numerical outputs',
        facts_sha256=digest((frozen/'data/rtl_facts.json').read_bytes()),
        numerical_contract_sha256=digest((frozen/'data/target_contract.yaml').read_bytes()),
        selection='retain off default; matched target comparison pending', comparison={'cycles':'UNKNOWN','roofline':'UNKNOWN','utilization':'UNKNOWN'})
    if reuse_request and reuse_artifact:
        artifact = Path(reuse_artifact).resolve()
        # Require an artifact actually owned by that immutable broker request.
        if reuse_request not in artifact.parts or not artifact.is_relative_to(WORKSPACE/'selfcheck_out'):
            raise ValueError('reuse artifact is not owned by the selected request')
        if artifact.read_bytes() == Path(emissions['off']['product_path']).read_bytes():
            report['reuse'] = dict(request_id=reuse_request, artifact_path=str(artifact),
                                   program_sha256=emissions['off']['program_sha256'], equality='exact owner LLVM bytes')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def _result(body, capsule):
    rows = body.get('cases', body.get('per_capsule', []))
    if isinstance(rows, dict): rows = [dict(v, capsule=k) for k,v in rows.items()]
    row = next((r for r in rows if r.get('capsule',r.get('name')) == capsule), {})
    detail = row.get('detail', {}).get('path')
    if detail:
        path = WORKSPACE/detail
        raw = path.read_bytes()
        if digest(raw) != row['detail']['sha256']: raise ValueError('result detail identity mismatch')
        row = json.loads(raw)
    perf = row.get('performance', {})
    return dict(status=row.get('status','UNKNOWN'), tiers=row.get('tiers',{}),
        numeric=row.get('numeric','UNKNOWN'), failure_detail=row.get('failure_detail'),
        cycles=perf.get('estimated_whole_soc_cycles','UNKNOWN'), timing_status=perf.get('status','UNKNOWN'),
        provider='modeled: Spike timing model' if type(perf.get('estimated_whole_soc_cycles')) is int else 'UNKNOWN',
        pins={k:perf.get(k,'UNKNOWN') for k in ('scope','scope_sha256','model_sha256','facts_sha256',
            'comparison_domain_sha256','selection_sha256','timing_epoch','timing_mode','elf_sha256','complete_event_accounting')},
        roofline=perf.get('structural_roofline','UNKNOWN'), scoped_resources=perf.get('scoped_resource_feedback','UNKNOWN'),
        utilization=perf.get('array_utilization_pct','UNKNOWN'),
        owner_detail_path=str(WORKSPACE/detail) if detail else None,
        owner_artifacts=row.get('your_artifacts','UNKNOWN'),
        feature_counts=perf.get('feature_counts',{}))


def compact_report(report):
    """Keep artifact ownership and provisional timing qualifications together."""
    modes = {}
    for mode, variant in report['variants'].items():
        emission = report['emissions'][mode]
        request = report['requests'].get(mode, {})
        result = request.get('result', {})
        modes[mode] = dict(compiler_sha256=variant['compiler_sha256'],
            settings=variant['settings'], program_sha256=emission['program_sha256'],
            command_buffer_sha256=emission['command_buffer_sha256'],
            source_plan_sha256=emission['source_plan_sha256'],
            effect_sha256=emission['effect_sha256'],
            consumed_rewrites=emission.get('consumed_rewrites',sum(r['rewrites'] for r in emission['consumed_fusions'])),
            request_id=request.get('request_id', 'UNKNOWN'),
            complete=request.get('complete', False),
            status=result.get('status','UNKNOWN'), tiers=result.get('tiers',{}),
            failure_detail=result.get('failure_detail'),
            numeric=result.get('numeric','UNKNOWN'), cycles=result.get('cycles','UNKNOWN'),
            provider=result.get('provider','UNKNOWN'), pins=result.get('pins',{}),
            host_feature_counts={k:result.get('feature_counts',{}).get(k,'UNKNOWN')
                for k in ('cpu.branch','cpu.fp_load','cpu.fp_store')},
            scoped_resources=result.get('scoped_resources','UNKNOWN'),
            owner_detail_path=result.get('owner_detail_path'),
            owner_artifacts=result.get('owner_artifacts','UNKNOWN'),
            owner_products=request.get('owner_products','UNKNOWN'))
    return dict(source=report['source'], mechanism=report.get('mechanism','host_pointwise_fusion'), parent_compiler_sha256=report['parent_compiler_sha256'],
        frozen_compiler_sha256=report['frozen_compiler_sha256'],
        pass_order=report['pass_order'], variants=modes,
        source_ownership_identical=report['source_ownership_identical'],
        native_validation=report.get('native_validation','UNKNOWN'),
        object_validation=report.get('object_validation','UNKNOWN'),
        public_transport_limit=report.get('public_transport_limit','UNKNOWN'),
        comparison=report['comparison'], selection=report['selection'])


def owner_products(report, mode, request_id):
    """Inspect only the broker's documented public compiler-output directory."""
    root = WORKSPACE/'selfcheck_out'/'requests'/request_id/report['source']['capsule']
    emission = report['emissions'][mode]
    result = {}
    for name, local, semantic_json in (
        ('lowered.llvm.mlir', emission['product_path'], False),
        ('command_buffer.json', emission['command_buffer_path'], True)):
        artifact = root/name
        if not artifact.exists():
            result[name] = 'UNKNOWN: broker owner artifact not published'
            continue
        actual, expected = artifact.read_bytes(), Path(local).read_bytes()
        equal = json.loads(actual) == json.loads(expected) if semantic_json else actual == expected
        result[name] = dict(path=str(artifact), sha256=digest(actual),
            equality='semantic JSON' if semantic_json else 'exact bytes', matches_submission=equal)
        if not equal:
            raise ValueError('broker owner product differs from frozen ordinary emitter: '+str(artifact))
    return result


def evaluate(report_path, *, collect=False, wait_budget=5):
    if not 1 <= wait_budget <= 45:
        raise ValueError('wait budget must be between 1 and 45 seconds')
    path = Path(report_path).resolve(); report = json.loads(path.read_text())
    capsule = report['source']['capsule']
    for mode, variant in report['variants'].items():
        if tree_identity(Path(variant['package'])) != variant['compiler_sha256']:
            raise ValueError('immutable host variant changed')
        if mode == 'off' and report['reuse']:
            report['requests'].setdefault(mode, report['reuse'].copy())
        previous = report['requests'].get(mode)
        if not collect and previous: continue
        if collect and not previous: raise ValueError('evaluate before collecting')
        if collect and previous.get('complete'): continue
        response = path.parent/('selfcheck-'+mode+'.json')
        argv = [sys.executable,str(WORKSPACE/'agent_selfcheck.py')]
        argv += ['--attach',previous['request_id']] if collect or (mode == 'off' and report['reuse']) else [
            '--submission',variant['package'],'--capsules',capsule,'--suite','scalability','--sim','spike']
        argv += ['--wait-budget',str(wait_budget),'--out',str(response)]
        run = subprocess.run(argv,cwd=WORKSPACE,capture_output=True,timeout=60)
        (path.parent/('selfcheck-'+mode+'.stdout')).write_bytes(run.stdout)
        (path.parent/('selfcheck-'+mode+'.stderr')).write_bytes(run.stderr)
        body = _body(run.stdout.decode(errors='replace'))
        if not body and response.exists(): body = json.loads(response.read_text())
        request_id = body.get('request_id',body.get('selfcheck_request_id',previous.get('request_id') if previous else None))
        if not request_id: raise RuntimeError('broker returned no request identity')
        result = _result(body,capsule)
        # Scalability's `complete`/`all_pass` describe its entire required roster,
        # even for a finished single-case request. Keep those suite fields while
        # identifying request completion and the explicitly requested row.
        finished_request = body.get('selfcheck_request_id') == request_id and body.get('graded_at_unix_ns') is not None
        required_tiers = body.get('policy',{}).get('tiers',('L0','L1','L2'))
        selected_case_pass = result['status'] == 'pass' and all(
            result['tiers'].get(tier) == 'pass' for tier in required_tiers)
        report['requests'][mode] = dict(request_id=request_id,response_path=str(response),argv=argv,
            returncode=run.returncode,complete=finished_request,
            suite_complete=body.get('complete'),selected_case_pass=selected_case_pass,
            all_pass=body.get('all_pass'),result=result,evidence=body,
            owner_products=owner_products(report,mode,request_id))
        path.write_text(json.dumps(report,indent=2)+'\n')
        if run.returncode not in (0,1,3): raise RuntimeError('broker request failed')
    for mode, request in report['requests'].items():
        request['owner_products'] = owner_products(report,mode,request['request_id'])
    a,b = [report['requests'].get(m,{}).get('result',{}) for m in ('off','on')]
    pins = ('scope_sha256','model_sha256','facts_sha256','comparison_domain_sha256','selection_sha256','timing_epoch','timing_mode')
    matched = bool(a and b) and all(a.get('pins',{}).get(k) == b.get('pins',{}).get(k) not in (None,'UNKNOWN') for k in pins)
    finished = all(report['requests'].get(m,{}).get('complete')
                   and report['requests'].get(m,{}).get('selected_case_pass') is True for m in ('off','on'))
    implementation_differs = report['emissions']['off']['program_sha256'] != report['emissions']['on']['program_sha256']
    same_elf = a.get('pins',{}).get('elf_sha256') == b.get('pins',{}).get('elf_sha256') not in (None,'UNKNOWN')
    execution_identity_issue = bool(implementation_differs and same_elf)
    connections = {}
    for mode in ('off','on'):
        products = report['requests'].get(mode,{}).get('owner_products',{})
        connections[mode] = ('matched ordinary LLVM and command buffer' if all(
            isinstance(products.get(name),dict) and products[name].get('matches_submission') is True
            for name in ('lowered.llvm.mlir','command_buffer.json')) else 'UNKNOWN: owner compiler products not published')
    eligible = matched and finished and all(r.get('status') == 'pass' and type(r.get('cycles')) is int
                              and r.get('timing_status') in ('PROVISIONAL','QUALIFIED')
                              and r.get('pins',{}).get('complete_event_accounting') is True for r in (a,b))
    eligible = eligible and not execution_identity_issue and all(
        value == 'matched ordinary LLVM and command buffer' for value in connections.values())
    report['comparison'] = dict(matched_domain=matched, provisional_ranking_eligible=eligible,
        execution_connection=connections,
        execution_identity_issue='same ELF reported for different ordinary compiler products; owner artifacts needed'
            if execution_identity_issue else None,
        observed_cycle_ratio_off_over_on=a['cycles']/b['cycles'] if matched and finished
            and all(type(r.get('cycles')) is int for r in (a,b)) else 'UNKNOWN',
        cycle_ratio_off_over_on=a['cycles']/b['cycles'] if eligible else 'UNKNOWN',
        off_cycles=a.get('cycles','UNKNOWN'),on_cycles=b.get('cycles','UNKNOWN'),
        evidence='modeled: Spike timing model; not measured hardware cycles',
        roofline={m:report['requests'].get(m,{}).get('result',{}).get('roofline','UNKNOWN') for m in ('off','on')},
        utilization={m:report['requests'].get(m,{}).get('result',{}).get('utilization','UNKNOWN') for m in ('off','on')})
    report['selection'] = 'retain off default; provisional comparison requires independent shapes and measured qualification'
    path.write_text(json.dumps(report,indent=2)+'\n')
    path.with_name('comparison.json').write_text(json.dumps(compact_report(report),indent=2)+'\n')
    return report['comparison']


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--submission',default='submission');prep.add_argument('--output',required=True)
    prep.add_argument('--variant-prefix',required=True);prep.add_argument('--source',required=True);prep.add_argument('--capsule',required=True)
    prep.add_argument('--expected-off');prep.add_argument('--expected-on');prep.add_argument('--reuse-request');prep.add_argument('--reuse-artifact')
    prep.add_argument('--mechanism',choices=('host_pointwise_fusion','host_tensor_views'),default='host_pointwise_fusion')
    for command in ('evaluate','collect'):
        action=sub.add_parser(command);action.add_argument('--report',required=True)
        action.add_argument('--wait-budget',type=int,default=5)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='prepare':
        report=prepare(**{k.replace('-','_'):v for k,v in args.items()});print(json.dumps({'report':str(Path(args['output']).resolve()/'report.json'),'emissions':report['emissions']},indent=2))
    else:print(json.dumps(evaluate(args['report'],collect=command=='collect',wait_budget=args['wait_budget']),indent=2))


if __name__ == '__main__': main()
