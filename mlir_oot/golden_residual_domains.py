"""Source-proven residual domains and restricted-pair feasibility experiment.

No graph rewriting. Unknown producers retain the complete signed-i8 domain.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from xdsl.ir import Operation
from .golden_resadd_proof import match, op_name
from .golden_resadd_diagonal_search import CPP, source_values
from .captured_residual_bundle import inspect
from .frontend.parse import parse_module


def domain(value):
    if str(value.type.get_element_type()) != 'i8':
        raise ValueError('requires signed i8 tensor')
    owner=value.owner
    fallback=dict(minimum=-128,maximum=127,reason='complete signed-i8 type domain',path=[])
    if not isinstance(owner,Operation):
        return fallback
    name=op_name(owner)
    if name in ('tensor.collapse_shape','tensor.expand_shape','linalg.transpose'):
        try:
            owner.verify()
            source=owner.operands[0]
            a,b=source.type.get_shape(),value.type.get_shape()
            if any(d<=0 for d in (*a,*b)) or math.prod(a)!=math.prod(b) or source.type.get_element_type()!=value.type.get_element_type():
                return fallback
            result=domain(source)
            return dict(result,path=result['path']+[dict(operation=name,source_shape=list(a),result_shape=list(b))])
        except Exception:
            return fallback
    if name!='quant_ext.quantize_per_tensor':
        return fallback
    try:
        params=match(owner)
    except ValueError:
        return fallback
    if not params['relu']:
        return fallback
    # Complete signed-i8 inputs and finite source ordering: no sample-derived bounds.
    with np.errstate(over='ignore',invalid='ignore'):
        a=np.float32(128)*np.float32(params['lhs_scale'])
        b=np.float32(128)*np.float32(params['rhs_scale'])
        total=np.add(a,b,dtype=np.float32)
        reciprocal=np.float32(1.0/params['output_scale'])
        scaled=np.float32(total*reciprocal)
    if not all(np.isfinite(x) for x in (a,b,total,reciprocal,scaled)):
        return fallback
    return dict(minimum=0,maximum=127,reason='exact finite DQ+add+ReLU+positive symmetric Q source',
                path=[dict(operation=name,region=getattr(owner.attributes.get('prov.region_id'),'data',''),qparams=params,
                           maximum_abs_pre_relu=float(total),maximum_abs_scaled=float(scaled))])


def restricted_search(source,domains,engine,work):
    expected=source_values(source)
    a=np.arange(-128,128,dtype=np.int32)[:,None]
    b=np.arange(-128,128,dtype=np.int32)[None,:]
    mask=(a>=domains[0]['minimum'])&(a<=domains[0]['maximum'])&(b>=domains[1]['minimum'])&(b<=domains[1]['maximum'])
    encoded=np.where(mask,expected,-1).astype('<i2')
    path=work/'domain_pairs.bin';path.write_bytes(encoded.tobytes())
    result=json.loads(subprocess.check_output([str(engine),str(path),'32767',str(source['lhs_scale']/source['rhs_scale']),'2'],text=True))
    for candidate in result['accepted']:
        target=np.clip(np.rint((a*candidate['p']+b*candidate['q']).astype(np.float32)*np.float32(candidate['scale'])),0,127).astype('<i2')
        if not np.array_equal(target[mask],expected[mask]):
            raise AssertionError('independent admissible-pair verification failed')
        candidate['verified_target_sha256']=hashlib.sha256(target[mask].tobytes()).hexdigest()
        candidate['chunks']=(candidate['p']+126)//127+(candidate['q']+126)//127
    result.update(domains=domains,pairs=int(mask.sum()),source=source,
                  source_pairs_sha256=hashlib.sha256(expected[mask].tobytes()).hexdigest(),
                  scope='first exact candidate in explicit ratio neighborhood +/-2, coefficients1..32767; no optimality claim')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=False)
    # Sentinel only marks pairs excluded by independently established source domains.
    cpp=CPP.replace('pairs.push_back({a,b,expected[(a+128)*256+b+128]});',
                    'if(expected[(a+128)*256+b+128]>=0) pairs.push_back({a,b,expected[(a+128)*256+b+128]});')
    path=args.work/'search.cc';path.write_text(cpp);engine=args.work/'search'
    subprocess.run(['c++','-O3','-std=c++17','-ffp-contract=off',str(path),'-o',str(engine)],check=True)
    module=parse_module(args.source.read_text());results=[]
    for op in module.walk():
        if op_name(op)!='quant_ext.quantize_per_tensor':continue
        try: route=inspect(op,implementation='cpu_lut')
        except ValueError:continue
        domains=[domain(v) for v in route['inputs']]
        row=restricted_search(route['proof']['source'],domains,engine,args.work)
        row.update(region=getattr(op.attributes.get('prov.region_id'),'data',''),shape=route['shape'])
        if row['accepted']:
            row['issue_floor_cycles']=math.prod(route['shape'])//16*row['accepted'][0]['chunks']
        results.append(row);print(row['region'],row['pairs'],row['accepted'],flush=True)
    receipt=dict(schema='source_bound_residual_domain_experiment_v1',source=str(args.source),
                 source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                 proof_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 engine_sha256=hashlib.sha256(cpp.encode()).hexdigest(),results=results,
                 issue_floor_cycles=sum(r.get('issue_floor_cycles',0) for r in results),
                 promotion=False)
    (args.work/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
