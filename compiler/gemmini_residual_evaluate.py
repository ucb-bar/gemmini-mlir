#!/usr/bin/env python3
"""Freeze and evaluate matched serial versus checked loaded-sum epoch compilers.

All candidate payloads come from the ordinary manifest emitter. Consumption is
observed by replay and must reproduce those exact bytes. Broker jobs retain
complete source calls, exact original numeric policies and separate public suite scopes.
No absent complete-call cycles become a ranking or a default-setting change.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
sys.path[:0] = [str(PACKAGE / 'compiler_library'), str(PACKAGE)]
from gemmini_conv_evaluate import digest, tree_identity, copy_package, manifest_emit


def source_record(path, split, directory):
    from gemmini_frontend import normalize_source
    from merlin.transforms.pass_exploration import ExplorationGeometry
    path = Path(path).resolve()
    data = path.read_bytes()
    model = normalize_source(data.decode())
    if model['grammar'] != 'merlin_iface' or 'residual_add' not in data.decode():
        raise ValueError('residual evaluation requires an original complete interface residual source')
    family = 'residual_convolution' if 'conv2d' in data.decode() else 'residual_dense'
    shapes = tuple(tuple(map(int, s.split('x'))) for s in re.findall(r'tensor<([0-9x]+)xi(?:8|32)>', data.decode()))
    geometry = ExplorationGeometry(family, shapes)
    case = path.parent.name if path.name == 'capsule.interface.mlir' else path.stem
    members = json.loads((WORKSPACE / 'PUBLIC_CAPABILITIES.json').read_text())['members']
    if case not in members:
        raise ValueError('source does not identify a public capsule: ' + case)
    member = members[case]
    declared_sha = member.get('source_analysis', {}).get('sha256')
    if declared_sha and declared_sha != digest(data):
        raise ValueError('source bytes differ from public capability declaration for ' + case)
    saved = directory / (split + '-' + digest(data) + '.mlir')
    saved.write_bytes(data)
    return geometry, dict(path=str(path), frozen_path=str(saved), capsule=case,
        suite=member.get('selfcheck_suite', 'core'), source_sha256=digest(data),
        geometry_sha256=geometry.identity, shapes=shapes, split=split,
        public_cohort=member['cohort'])


def consume_replay(package, source, output, expected, route):
    script = r'''
import sys,json,re
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package),str(package/'compiler_library')]
import gemmini_encode as encoder
import gemmini_pointwise as pointwise
import gemmini_residual_schedule as scheduling
receipts=[];epochs=[];effects=[]
def normalize(v):
    if v is None or type(v) in (str,int,float,bool):return v
    if isinstance(v,dict):return {str(k):normalize(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [normalize(x) for x in v]
    return re.sub(r'0x[0-9a-fA-F]+','<address>',str(v))
original_residual=pointwise.emit_residual
def residual(*args,**kwargs):
    result=original_residual(*args,**kwargs)
    receipts.append(normalize(result));return result
pointwise.emit_residual=residual
original_epochs=scheduling.emit_loaded_sum_epochs
def epoch(plan,builder,lower):
    result=original_epochs(plan,builder,lower)
    epochs.append(dict(plan=normalize(plan.report()),receipt=normalize(result)))
    return result
scheduling.emit_loaded_sum_epochs=epoch
original=encoder.insn
def terminal(fn,funct,rs1,rs2):
    effects.append(dict(funct=funct,rs1=normalize(rs1),rs2=normalize(rs2)))
    return original(fn,funct,rs1,rs2)
encoder.insn=terminal
from gemmini_backend import compile_source
_,llvm=compile_source(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_bytes((llvm+'\n').encode())
Path(sys.argv[4]).write_text(json.dumps(dict(consumed_schedules=receipts,actual_epoch_consumption=epochs,terminal_effect_sites=effects),sort_keys=True,indent=2)+'\n')
'''
    replay, evidence = output / 'replay.mlir', output / 'consumed.json'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package / 'compiler_library'))
    run = subprocess.run([sys.executable, '-c', script, str(package), str(source), str(replay), str(evidence)],
                         cwd=package, env=env, capture_output=True, timeout=120)
    (output / 'replay.stderr').write_bytes(run.stderr)
    if run.returncode:
        raise RuntimeError('consumption replay failed: ' + str(output / 'replay.stderr'))
    if replay.read_bytes() != expected:
        raise ValueError('observing actual residual consumption changed ordinary emitter bytes')
    body = json.loads(evidence.read_text())
    if not body['consumed_schedules']:
        raise ValueError('source residual was not emitted')
    if route == 'epochs' and (not body['actual_epoch_consumption'] or not any(
            r.get('residual_schedule') == 'epochs' for r in body['consumed_schedules'])):
        raise ValueError('actual emitter did not consume checked loaded-sum epoch plan')
    if route == 'serial' and body['actual_epoch_consumption']:
        raise ValueError('serial incumbent unexpectedly consumed epoch schedule')
    return dict(consumption_path=str(evidence), effect_sha256=digest(body['terminal_effect_sites']),
        schedule_sha256=digest(body['consumed_schedules']), consumed_routes=[route],
        replay_byte_identical=True, actual_shared_plan_count=len(body['actual_epoch_consumption']),
        effect_scope='static actual encoder sites; no dynamic multiplicity inference')


def prepare(*, submission, output, variant_prefix, training, held_out, reuse_reports=(), modes=None,
            training_report=None):
    from merlin.transforms.pass_exploration import (PassFamily, EmittedProgram,
        ValidationPartition, generate_candidates, explore_candidates)
    from merlin.transforms.pipeline_policy import RegisteredPass, PolicyFacts
    from merlin.transforms.pipeline_state import PassSpec, PipelineContext
    submission, output = Path(submission).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('use a fresh experiment directory outside submission')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', variant_prefix):
        raise ValueError('use a bounded alphanumeric variant prefix')
    output.mkdir(parents=True)
    (output / 'sources').mkdir()
    groups, records = [], []
    for split, paths in (('training', training), ('held_out', held_out)):
        geometries = []
        for path in paths:
            geometry, record = source_record(path, split, output / 'sources')
            geometries.append(geometry)
            records.append(record)
        groups.append(tuple(geometries))
    prior_training = None
    if not groups[0]:
        if training_report is None:
            raise ValueError('fresh holdout needs a frozen prior training report')
        prior_path = Path(training_report).resolve()
        prior_raw = prior_path.read_bytes()
        prior = json.loads(prior_raw)
        from merlin.transforms.pass_exploration import ExplorationGeometry
        groups[0] = tuple(ExplorationGeometry(row['family'],
            tuple(tuple(shape) for shape in row['operand_shapes']))
            for row in prior['partition']['training'])
        prior_training = dict(path=str(prior_path), sha256=digest(prior_raw),
            policy='frozen prior training geometry only; no re-emission or target request')
    elif training_report is not None:
        raise ValueError('choose new training sources or prior training metadata, not both')
    partition = ValidationPartition(*groups)
    identity = tree_identity(submission)
    frozen = output / 'compiler'
    copy_package(submission, frozen)
    if tree_identity(frozen) != identity or tree_identity(submission) != identity:
        raise ValueError('compiler changed during snapshot; retry with a fresh output')
    modes = tuple(modes or ('serial', 'epochs'))
    if len(set(modes)) != len(modes) or len(modes) < 2 or set(modes) - {'serial', 'epochs'}:
        raise ValueError('select two distinct declared residual schedules')
    variants = {}
    for route in modes:
        destination = WORKSPACE / 'variants' / (variant_prefix + '-' + route)
        if destination.exists():
            raise ValueError('variant destination already exists')
        copy_package(frozen, destination)
        declaration = json.loads((destination / 'TUNING_KNOBS.json').read_text())
        if route not in declaration['knobs']['residual_schedule']['values']:
            raise ValueError('residual schedule is not an ordinary declared compiler knob')
        settings_file = destination / declaration['settings_file']
        settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
        settings['residual_schedule'] = route
        # Keep every non-residual setting byte-for-byte paired between the variants.
        settings_file.write_text(json.dumps(settings, indent=2) + '\n')
        variants[route] = dict(package=str(destination), compiler_sha256=tree_identity(destination), settings=settings)
    emissions = []
    def select(knobs):
        def apply(session):
            session.module['route'] = knobs['route']
            return {'residual_settings': knobs['route']}
        return RegisteredPass(PassSpec('residual-schedule', 0, produces=('residual_settings',), ir_level='authored_source'), apply)
    def emit(_):
        def apply(session):
            route, source = session.module['route'], session.module['source']
            package = Path(variants[route]['package'])
            location = output / 'products' / source['source_sha256'] / route
            product, receipt = manifest_emit(package, Path(source['frozen_path']), location)
            replay = consume_replay(package, Path(source['frozen_path']), location, product, route)
            schedule = digest({'source': source['source_sha256'], 'settings': variants[route]['settings'],
                               'consumption': replay['schedule_sha256']})
            emissions.append(dict(mode=route, split=source['split'], capsule=source['capsule'], suite=source['suite'],
                source_sha256=source['source_sha256'], geometry_sha256=source['geometry_sha256'],
                compiler_sha256=variants[route]['compiler_sha256'], pipeline_receipt=None,
                emitted_schedule_sha256=schedule, **receipt, **replay))
            return {'program': {'data': product, 'schedule_sha256': schedule}}
        return RegisteredPass(PassSpec('ordinary-manifest-emitter', 1, requires=('residual_settings',),
            produces=('program',), mutates_ir=False, ir_level='authored_source'), apply)
    candidates = generate_candidates((PassFamily('residual-schedule', select, (('route', modes),)),
                                     PassFamily('ordinary-manifest-emitter', emit)), output='program', max_candidates=len(modes))
    context = PipelineContext(hardware=digest((frozen / 'data/rtl_facts.json').read_bytes()),
        numerical_policy=digest((frozen / 'data/target_contract.yaml').read_bytes()),
        execution='ordinary-manifest-residual-epochs;complete-target-check-pending',
        library=tree_identity(frozen / 'compiler_library'))
    summaries = []
    for source in records:
        for row in explore_candidates({'source': source, 'route': 'serial'}, candidates,
            lambda s: PolicyFacts(s.stamp(), shapes=tuple(map(tuple, source['shapes']))), output='program',
            emission=lambda r: EmittedProgram(r.artifact['data'], 'llvm-mlir', r.artifact['schedule_sha256']),
            context=context, ir_level='authored_source', clone=copy.deepcopy, fingerprint=digest, verify_ir=lambda m: None):
            if row.emission is None:
                raise ValueError('ordinary residual candidate refused: ' + repr(row.summary()))
            for result in emissions:
                if result['source_sha256'] == source['source_sha256'] and result['program_sha256'] == digest(row.emission.data):
                    result['pipeline_receipt'] = row.evaluation.result.receipt
            summaries.append({**row.summary(), 'source_sha256': source['source_sha256']})
    changes = []
    for source in records:
        paired = {r['mode']: r for r in emissions if r['source_sha256'] == source['source_sha256']}
        for baseline, candidate in zip(modes, modes[1:]):
            changed_program = paired[baseline]['program_sha256'] != paired[candidate]['program_sha256']
            changed_consumption = paired[baseline]['schedule_sha256'] != paired[candidate]['schedule_sha256']
            if not changed_program or not changed_consumption:
                raise ValueError('candidate has no real emitted and consumed schedule change')
            changes.append(dict(capsule=source['capsule'], source_sha256=source['source_sha256'],
                baseline=baseline, candidate=candidate,
                actual_emitted_change=changed_program, consumed_schedule_change=changed_consumption))
    report = dict(schema='merlin.gemmini.residual_epochs_evaluation.v1', created_unix_ns=time.time_ns(),
        hypothesis='bounded disjoint loaded-sum epochs reduce complete-call completion cost',
        frozen_compiler_sha256=identity, partition=partition.manifest(), prior_training=prior_training,
        sources=records, variants=variants,
        emissions=emissions, candidates=summaries, changes=changes, requests={}, reuses={},
        reuse_reports=list(map(str, reuse_reports)), pass_order=['residual-schedule', 'ordinary-manifest-emitter'],
        cycles='UNKNOWN', selection='retain serial default; no measured cycle winner',
        comparison_scope='exact source numerics and matched complete-call cycles only')
    # Equality here is per-source actual LLVM, never whole-tree equality. A
    # matched old request may contain extra cases; only its equivalent row is reused.
    for previous_path in reuse_reports:
        previous = json.loads(Path(previous_path).read_text())
        for emitted in emissions:
            if emitted['mode'] != 'serial':
                continue
            for old in previous.get('emissions', []):
                if (old['source_sha256'], old['program_sha256']) != (emitted['source_sha256'], emitted['program_sha256']):
                    continue
                request = previous.get('requests', {}).get(old['mode'],
                    previous.get('requests', {}).get(old['mode'] + ':' + emitted['suite'], {}))
                if not request.get('request_id'):
                    reused = previous.get('reuses', {}).get(emitted['capsule'], {})
                    if (reused.get('source_sha256'), reused.get('program_sha256')) == (
                            emitted['source_sha256'], emitted['program_sha256']):
                        request = reused
                if request.get('request_id'):
                    report['reuses'][emitted['capsule']] = dict(request_id=request['request_id'],
                        response_path=request.get('response_path'), report=str(Path(previous_path).resolve()),
                        old_mode=old['mode'], source_sha256=emitted['source_sha256'],
                        program_sha256=emitted['program_sha256'], equality='actual relevant emitted LLVM bytes')
                    break
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _body(stdout):
    for index, character in enumerate(stdout):
        if character != '{':
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(stdout[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and 'request_id' in value:
            return value
    return {}


def compare_complete_calls(report):
    """Compare exact numeric rows only under matched complete model identities."""
    results = {}
    for request in report['requests'].values():
        evidence = request.get('evidence', {})
        rows = evidence.get('per_capsule', evidence.get('cases', []))
        for row in rows:
            performance = row.get('performance', {})
            detail = row.get('detail', {})
            detail_path = WORKSPACE / detail.get('path', '')
            if detail.get('sha256') and detail_path.is_file():
                raw = detail_path.read_bytes()
                if digest(raw) != detail['sha256']:
                    raise ValueError('content-addressed target performance detail changed')
                performance = json.loads(raw).get('performance', performance)
            keys = ('model_sha256', 'facts_sha256', 'comparison_domain_sha256', 'scope_sha256')
            results[(request['route'], row['capsule'])] = dict(
                numerical_execution_passed=row.get('numerical_execution_passed',
                    row.get('status') == 'pass' and row.get('tiers', {}).get('L2') == 'pass'),
                elf_sha256=performance.get('elf_sha256'),
                status=performance.get('status'),
                estimated_whole_soc_cycles=performance.get('estimated_whole_soc_cycles'),
                complete_event_accounting=performance.get('complete_event_accounting'),
                model_identity={key: performance.get(key) for key in keys},
                detail=detail)
    comparisons = []
    modes = tuple(report['variants'])
    for source in report['sources']:
        before = results.get((modes[0], source['capsule']), {})
        for mode in modes[1:]:
            after = results.get((mode, source['capsule']), {})
            usable = all(v.get('numerical_execution_passed') is True and
                v.get('complete_event_accounting') is True and
                v.get('status') in ('PROVISIONAL', 'QUALIFIED') and
                type(v.get('estimated_whole_soc_cycles')) is int and
                all(v.get('model_identity', {}).values()) for v in (before, after))
            usable = usable and before['model_identity'] == after['model_identity']
            delta = (after['estimated_whole_soc_cycles'] - before['estimated_whole_soc_cycles']) if usable else None
            comparisons.append(dict(capsule=source['capsule'], source_sha256=source['source_sha256'],
                baseline=modes[0], candidate=mode, before=before, after=after,
                comparable=usable, delta_cycles=delta,
                evidence='modeled: matched complete kernel-call Spike window' if usable else 'UNKNOWN or pending; unranked'))
    return comparisons


def owner_product_custody(report):
    """Bind ordinary LLVM to owner build/performance bytes and qualification.

    The broker retains build manifests rather than ELF binaries. Distinguish
    exact numerical execution from its independent owner qualification veto.
    """
    rows = []
    for request in report['requests'].values():
        rid = request.get('request_id')
        if not rid or request.get('suite') != 'scalability':
            continue
        evidence = request.get('evidence', {})
        for row in evidence.get('cases', evidence.get('per_capsule', [])):
            source = next(v for v in report['sources'] if v['capsule'] == row['capsule'])
            emission = next(v for v in report['emissions']
                if v['source_sha256'] == source['source_sha256'] and v['mode'] == request['route'])
            locations = list((WORKSPACE / '.qa_channel').glob(
                '.resp_' + rid + '.*.scalability/runs/runs/gemmini-capsule-bench/' + row['capsule']))
            if len(locations) != 1:
                rows.append(dict(route=request['route'], capsule=row['capsule'],
                    request_id=rid, verified=False, reason='unique owner run not yet available'))
                continue
            run = locations[0]
            paths = [run / 'generated' / name for name in (
                'lowered.llvm.mlir', 'readback_build.json', 'lowering_recipe.json', 'kernel.ll')]
            if not all(p.is_file() for p in paths) or not (run / 'capsule_result.json').is_file():
                rows.append(dict(route=request['route'], capsule=row['capsule'],
                    request_id=rid, verified=False, reason='owner build artifacts not yet available'))
                continue
            build, recipe = (json.loads(paths[i].read_text()) for i in (1, 2))
            owner = json.loads((run / 'capsule_result.json').read_text())
            detail = row.get('detail', {})
            raw = (WORKSPACE / detail['path']).read_bytes()
            if digest(raw) != detail['sha256']:
                raise ValueError('owner custody performance detail changed')
            performance = json.loads(raw)['performance']
            actual = digest(paths[0].read_bytes())
            checks = dict(ordinary_llvm=actual == emission['program_sha256'],
                returned_llvm=recipe['returned_llvm_ir']['sha256'] == digest(paths[3].read_bytes()),
                executed_elf=build['elf_sha256'] == performance['elf_sha256'])
            if not all(checks.values()):
                raise ValueError('owner did not consume intended residual product: ' + repr(checks))
            rows.append(dict(route=request['route'], capsule=row['capsule'], request_id=rid,
                verified=True, checks=checks, run_path=str(run), actual_llvm_sha256=actual,
                expected_llvm_sha256=emission['program_sha256'],
                owner_kernel_object_sha256=build['kernel_object_sha256'],
                owner_elf_sha256=build['elf_sha256'], performance_elf_sha256=performance['elf_sha256'],
                elf_binary_retained=False, numeric=owner.get('numeric'),
                owner_qualification_status=owner.get('status'), owner_failure=owner.get('failure'),
                public_row_status=row.get('status'), public_tiers=row.get('tiers'),
                performance_detail=detail))
    return rows


def evaluate(report_path, *, collect=False):
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    for route, variant in report['variants'].items():
        if tree_identity(Path(variant['package'])) != variant['compiler_sha256']:
            raise ValueError('immutable residual variant changed')
        suites = sorted({s['suite'] for s in report['sources']})
        for suite in suites:
            cases = [s['capsule'] for s in report['sources'] if s['suite'] == suite
                     and not (route == 'serial' and s['capsule'] in report['reuses'])]
            if not cases:
                continue
            key = route + ':' + suite
            previous = report['requests'].get(key)
            if collect and not previous:
                raise ValueError('evaluate before collecting')
            if not collect and previous:
                continue
            if collect and previous.get('all_pass') is not None:
                continue
            response = path.parent / ('selfcheck-' + route + '-' + suite + '.json')
            argv = [sys.executable, str(WORKSPACE / 'agent_selfcheck.py')]
            if collect:
                argv += ['--attach', previous['request_id']]
            else:
                argv += ['--submission', variant['package'], '--capsules', ','.join(cases), '--suite', suite, '--sim', 'spike']
            argv += ['--wait-budget', '5', '--out', str(response)]
            run = subprocess.run(argv, cwd=WORKSPACE, capture_output=True, timeout=60)
            log = path.parent / ('selfcheck-' + route + '-' + suite)
            log.with_suffix('.stdout').write_bytes(run.stdout)
            log.with_suffix('.stderr').write_bytes(run.stderr)
            body = _body(run.stdout.decode(errors='replace'))
            if not body and response.exists():
                body = json.loads(response.read_text())
            # The broker writes --out only when complete. Preserve independent
            # partial numerical decisions without treating them as final cycles.
            status_path = path.parent / ('status-' + route + '-' + suite + '.json')
            status_path.write_text(json.dumps(body, indent=2) + '\n')
            report['requests'][key] = dict(request_id=body.get('request_id', previous.get('request_id') if previous else None),
                response_path=str(response), capsules=cases, suite=suite, route=route,
                status=body.get('status', 'complete' if 'all_pass' in body else 'unknown'), all_pass=body.get('all_pass'),
                status_path=str(status_path), evidence=body, argv=argv, returncode=run.returncode)
            path.write_text(json.dumps(report, indent=2) + '\n')
            if run.returncode not in (0, 3) and not (run.returncode == 1 and body.get('all_pass') is False):
                raise RuntimeError('ordinary residual selfcheck failed: ' + str(log.with_suffix('.stderr')))
    if collect:
        for case, reuse in report['reuses'].items():
            if reuse.get('all_pass') is not None:
                continue
            response = path.parent / ('reused-' + case + '.json')
            argv = [sys.executable, str(WORKSPACE / 'agent_selfcheck.py'), '--attach', reuse['request_id'],
                    '--wait-budget', '5', '--out', str(response)]
            run = subprocess.run(argv, cwd=WORKSPACE, capture_output=True, timeout=60)
            body = _body(run.stdout.decode(errors='replace'))
            if not body and response.exists():
                body = json.loads(response.read_text())
            status_path = path.parent / ('status-reused-' + case + '.json')
            status_path.write_text(json.dumps(body, indent=2) + '\n')
            reuse.update(response_path=str(response), status_path=str(status_path), evidence=body,
                         status=body.get('status'), all_pass=body.get('all_pass'))
            (path.parent / ('reused-' + case + '.stdout')).write_bytes(run.stdout)
            (path.parent / ('reused-' + case + '.stderr')).write_bytes(run.stderr)
    report['comparisons'] = compare_complete_calls(report)
    report['owner_product_custody'] = owner_product_custody(report)
    report['selection'] = 'retain serial default; compare only matched complete-call evidence; no automatic promotion'
    path.write_text(json.dumps(report, indent=2) + '\n')
    return dict(requests={key: {k: row.get(k) for k in
        ('request_id', 'status', 'all_pass', 'response_path')} for key, row in report['requests'].items()},
        comparisons=report['comparisons'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--submission', default='submission')
    prepare_parser.add_argument('--output', required=True)
    prepare_parser.add_argument('--variant-prefix', required=True)
    prepare_parser.add_argument('--train', action='append', default=[])
    prepare_parser.add_argument('--training-report')
    prepare_parser.add_argument('--held-out', action='append', required=True)
    prepare_parser.add_argument('--reuse-report', action='append', default=[])
    prepare_parser.add_argument('--mode', action='append', choices=('serial', 'epochs'))
    for command in ('evaluate', 'collect'):
        sub.add_parser(command).add_argument('--report', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        report = prepare(submission=args.submission, output=args.output, variant_prefix=args.variant_prefix,
                         training=args.train, held_out=args.held_out, reuse_reports=args.reuse_report, modes=args.mode,
                         training_report=args.training_report)
        print(json.dumps(dict(report=str(Path(args.output).resolve() / 'report.json'), changes=report['changes'],
                              reuses=report['reuses']), indent=2))
    else:
        print(json.dumps(evaluate(args.report, collect=args.command == 'collect'), indent=2))

if __name__ == '__main__':
    main()
