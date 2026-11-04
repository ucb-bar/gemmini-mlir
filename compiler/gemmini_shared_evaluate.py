#!/usr/bin/env python3
"""Freeze paired ordinary shared-input compilers and request matched checks."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'compiler_library')]
from gemmini_conv_evaluate import (WORKSPACE,digest,tree_identity,copy_package,manifest_emit,broker_evaluate)


def consume(package,source,location,expected,mode):
    # Observe the very same source-to-target entry point used by the manifest.
    script=r'''
import sys,json,hashlib
from pathlib import Path
package=Path(sys.argv[1]);sys.path[:0]=[str(package),str(package/'compiler_library')]
import gemmini_graph_reuse as reuse
original=reuse.emit_shared_products
records=[]
def emit(fn,plan,pointers):
    numeric=reuse.verify_numerics(plan)
    receipt=original(fn,plan,pointers)
    records.append(dict(receipt=receipt,packets=plan.packets,products=plan.products,numerics=numeric))
    return receipt
reuse.emit_shared_products=emit
from gemmini_backend import compile_source
source=Path(sys.argv[2]).read_bytes()
cb,ll=compile_source(source.decode())
Path(sys.argv[3]).write_bytes((ll+'\n').encode())
Path(sys.argv[4]).write_text(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),outputs=cb['kernel_abi']['outputs'],consumed=records),sort_keys=True,indent=2)+'\n')
'''
    target=location/'replay.mlir';evidence=location/'consumed.json'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(package/'compiler_library'))
    result=subprocess.run([sys.executable,'-c',script,str(package),str(source),str(target),str(evidence)],
        cwd=package,env=env,capture_output=True,timeout=120)
    (location/'replay.stderr').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError('consumption failed: '+str(location/'replay.stderr'))
    if target.read_bytes()!=expected:raise ValueError('observation changed ordinary manifest product')
    body=json.loads(evidence.read_text())
    if body['source_sha256']!=digest(source.read_bytes()):raise ValueError('source identity changed')
    if bool(body['consumed'])!=(mode=='on'):raise ValueError('shared source schedule was not consumed as requested')
    for record in body['consumed']:
        receipt=record['receipt']
        if digest(record['packets'])!=receipt['emitted_effects_sha256']:raise ValueError('emitted effects identity changed')
        if digest(record['products'])!=receipt['source_bindings_sha256']:raise ValueError('original source bindings changed')
        if record['numerics']['emitted_effects_sha256']!=receipt['emitted_effects_sha256']:raise ValueError('numeric check used another effect stream')
        if receipt['independent_readouts']!=[p['dst'] for p in record['products']]:raise ValueError('original readouts changed')
    return dict(consumption_path=str(evidence),outputs=body['outputs'],
        consumed_sha256=digest(body['consumed']),shared_receipts=[r['receipt'] for r in body['consumed']],
        numeric_checks=[r['numerics'] for r in body['consumed']],replay_byte_identical=True)


def prepare(*,submission,output,variant_prefix,training,held_out,capsules,suite='core',sim='spike'):
    submission=Path(submission).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(submission):raise ValueError('use a fresh directory outside submission')
    if not variant_prefix or len(variant_prefix)>40 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in variant_prefix):raise ValueError('invalid variant prefix')
    output.mkdir(parents=True);(output/'sources').mkdir()
    records=[]
    for split,paths in [('training',training),('held_out',held_out)]:
        for path in paths:
            path=Path(path).resolve();payload=path.read_bytes();sha=digest(payload)
            frozen=output/'sources'/(split+'-'+sha+'.mlir');frozen.write_bytes(payload)
            records.append(dict(path=str(path),frozen_path=str(frozen),split=split,source_sha256=sha))
    if not training or not held_out:raise ValueError('independent training and held-out sources required')
    if set(r['source_sha256'] for r in records if r['split']=='training')&set(r['source_sha256'] for r in records if r['split']=='held_out'):raise ValueError('held-out source overlaps training')
    parent=tree_identity(submission);frozen=output/'compiler';copy_package(submission,frozen)
    if tree_identity(submission)!=parent or tree_identity(frozen)!=parent:raise ValueError('compiler changed while freezing; retry')
    variants={};emissions=[]
    variant_root=WORKSPACE/'variants';variant_root.mkdir(exist_ok=True)
    for mode in ('off','on'):
        variant=variant_root/(variant_prefix+'-'+mode)
        if variant.exists():raise ValueError('variant already exists')
        marker=variant_root/('.library-pass-pending-'+variant.name);marker.write_text('Shared-input preparation pending.\n')
        copy_package(frozen,variant)
        declaration=json.loads((variant/'TUNING_KNOBS.json').read_text())
        if mode not in declaration['knobs']['graph_shared_inputs']['values']:raise ValueError('undeclared shared-input choice')
        settings=variant/declaration['settings_file'];values=json.loads(settings.read_text());values['graph_shared_inputs']=mode
        settings.write_text(json.dumps(values,indent=2)+'\n')
        variants[mode]=dict(package=str(variant),compiler_sha256=tree_identity(variant),settings=values)
        for record in records:
            location=output/'products'/record['source_sha256']/mode
            product,receipt=manifest_emit(variant,Path(record['frozen_path']),location)
            consumed=consume(variant,Path(record['frozen_path']),location,product,mode)
            emissions.append(dict(mode=mode,source_sha256=record['source_sha256'],split=record['split'],**receipt,**consumed))
        marker.unlink()
    changes=[]
    for record in records:
        pair={r['mode']:r for r in emissions if r['source_sha256']==record['source_sha256']}
        if pair['off']['program_sha256']==pair['on']['program_sha256']:raise ValueError('shared setting did not change actual emitted LLVM')
        if pair['off']['outputs']!=pair['on']['outputs']:raise ValueError('published outputs changed')
        changes.append(dict(source_sha256=record['source_sha256'],baseline_program_sha256=pair['off']['program_sha256'],
            candidate_program_sha256=pair['on']['program_sha256'],outputs=pair['on']['outputs'],actual_emitted_change=True))
    report=dict(schema='merlin.gemmini.shared_input_experiment.v1',created_unix_ns=time.time_ns(),
        frozen_compiler_sha256=parent,sources=records,variants=variants,emissions=emissions,changes=changes,
        hypothesis='shared immutable LHS retention reduces complete-region traffic and elapsed cycles while retaining every original output',
        scope=dict(capsules=capsules,suite=suite,sim=sim),requests={},cycles='UNKNOWN',selection='retain default off; no promotion',
        evidence_scope='actual ordinary manifest LLVM and consumed shared schedule; exact local integer execution; target numerics and complete-call cycles owned by ordinary selfcheck')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def evaluate(report_path, *, collect=False):
    path=Path(report_path).resolve();report=json.loads(path.read_text())
    for source in report['sources']:
        if digest(Path(source['frozen_path']).read_bytes())!=source['source_sha256']:
            raise ValueError('frozen source changed')
    for emission in report['emissions']:
        if digest(Path(emission['product_path']).read_bytes())!=emission['program_sha256']:
            raise ValueError('frozen ordinary target product changed')
        consumption=json.loads(Path(emission['consumption_path']).read_text())
        if digest(consumption['consumed'])!=emission['consumed_sha256']:
            raise ValueError('consumed schedule evidence changed')
    return broker_evaluate(path,collect=collect)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--submission',default='submission');q.add_argument('--output',required=True)
    q.add_argument('--variant-prefix',required=True);q.add_argument('--train',action='append',required=True);q.add_argument('--held-out',action='append',required=True)
    q.add_argument('--capsules',required=True);q.add_argument('--suite',default='core',choices=['core','scalability','programs']);q.add_argument('--sim',default='spike',choices=['spike','gsim'])
    for command in ('evaluate','collect'):
        q=sub.add_parser(command);q.add_argument('--report',required=True)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='prepare':
        args['training']=args.pop('train');args['held_out']=args.pop('held_out');report=prepare(**args)
        print(json.dumps(dict(report=str(Path(args['output']).resolve()/'report.json'),parent_sha256=report['frozen_compiler_sha256'],changes=report['changes']),indent=2))
    else:print(json.dumps(evaluate(args['report'],collect=command=='collect'),indent=2))

if __name__=='__main__':main()
