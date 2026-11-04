#!/usr/bin/env python3
"""Run the requested shared parallel recipe on ordinary source-bound RHS plans.

The caller selects immutable ordinary packages. Actual checked effects are consumed
by their common terminal and compared with ordinary manifest stdout. This adapter
never requests timing; original paired feedback is joined separately, without
substituting declaration hashes for broker input/reference custody.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
        json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def tree_identity(path):
    return digest([(str(p.relative_to(path)),digest(p.read_bytes())) for p in sorted(path.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])


MAPPING={'original':'incumbent','batched':'rhs_ready','ready':'rhs_ready_wide','identical':'incumbent'}


class SourceBoundAdapter:
    def __init__(self,workspace,output,packages,sources):
        self.workspace=workspace;self.output=output;self.packages=packages;self.sources=sources
        self.programs={};self.domains={};self.bindings={}
        self.package_pins={name:tree_identity(path) for name,path in packages.items()}

    def binding(self,shape,caps):
        from gemmini_dense import plan_dense,geometry,FACTS_PATH
        from gemmini_dense_coalesced_policy import m1_coalesced_auto_schedule
        from gemmini_dense_rhs_ready_policy import rhs_ready_auto_schedule
        from gemmini_frontend import normalize_source
        from gemmini_readout import source_scales,_relu
        from merlin.transforms.pipeline_state import PipelineContext
        matches=[r for r in self.sources if tuple(r['shape'])==tuple(shape)]
        if len(matches)!=1:raise ValueError('requires exactly one original mathematical source per geometry')
        record=matches[0];source=Path(record['path']).read_text();model=normalize_source(source)
        product=[o for o in model['operations'] if o['category']=='product']
        readouts=[o for o in model['operations'] if o['category']=='readout']
        if len(product)!=1 or product[0]['opcode'] not in ('MATMUL','MATMUL_RESIDENT') or len(readouts)!=1:
            raise ValueError('source-bound adapter requires a single rank-two dense product/readout')
        attrs=readouts[0]['attributes'];dtype=attrs.get('output_dtype','i32')
        readout=json.dumps(dict(dtype=dtype,scale=float(source_scales(attrs)),relu=_relu(attrs)),sort_keys=True)
        base,adapter,problem=plan_dense(*shape,readout_identity=readout)
        arguments=dict(attrs=attrs,output_dtype=dtype,outer_order=('m','n'))
        original,mode,evidence=m1_coalesced_auto_schedule(base,adapter,problem,geometry(),FACTS_PATH,**arguments)
        alternatives={'incumbent':(original,mode,evidence)}
        result_owners={id(original):'incumbent'}
        def transform(result,branch):
            if result is not original:raise ValueError('branch lost its original checked source')
            name=MAPPING[branch]
            if name not in alternatives:
                candidate,terminal,proof=rhs_ready_auto_schedule(base,adapter,problem,geometry(),FACTS_PATH,
                    allow_scalar_readout=True,aggregate_transfers=name=='rhs_ready_wide',**arguments)
                candidate.revalidate();alternatives[name]=(candidate,terminal,proof);result_owners[id(candidate)]=name
            return alternatives[name][0]
        def emit(result):
            from gemmini_dense import consume_checked_dense
            from gemmini_backend import compile_source
            name=result_owners[id(result)];_,terminal,proof=alternatives[name]
            with consume_checked_dense(result,adapter,problem,mode=terminal,readout_identity=readout) as consumed:
                _,lowered=compile_source(source)
            payload=(lowered+'\n').encode()
            manifest=self.output/(record['capsule']+'-'+name+'.manifest.llvm.mlir')
            command=[sys.executable,str(self.packages[name]/'gemmini-opt'),
                '--convert-iface-to-gemmini','--emit-target-artifact',record['path']]
            done=subprocess.run(command,capture_output=True,check=True)
            manifest.write_bytes(done.stdout)
            if done.stdout!=payload:raise ValueError('actual consumed effects differ from ordinary manifest: '+name)
            row=dict(capsule=record['capsule'],shape=shape,source_path=record['path'],
                source_sha256=record['source_sha256'],branch=name,terminal=terminal,
                readout_identity=readout,adapter_numerical_identity=adapter.numerical_policy_identity,
                original_effect_sha256=original.kernel.digest(),effect_sha256=result.kernel.digest(),
                program_sha256=digest(payload),manifest_path=str(manifest),manifest_command=command,
                manifest_byte_equal=True,consumed=consumed['consumed'],admission=proof,
                compiler_tree_sha256=self.package_pins[name])
            self.programs[(record['source_sha256'],name)]=row
            return payload.decode(),row
        binding=dict(source=original,transform=transform,emit=emit,shapes=((shape[0],shape[1]),(shape[1],shape[2])),
            context=PipelineContext(hardware=digest(FACTS_PATH.read_bytes()),resources=digest(asdict(geometry()[1])),
                numerical_policy=adapter.numerical_policy_identity,execution='ordinary-selfcheck:spike:scalability',
                library=tree_identity(self.packages['rhs_ready_wide']/'compiler_library')),
            record=record,model=model)
        self.bindings[tuple(shape)]=binding
        return binding

    def identity(self,shape,binding):
        from gemmini_dense import FACTS_PATH
        from merlin.transforms.evaluated_exploration import ExplorationIdentity,exploration_context_identity
        record=binding['record'];model=binding['model']
        declaration=dict(source=record['source_sha256'],abi=model['logical_args'],custody='source declarations only')
        identity=ExplorationIdentity(record['source_sha256'],digest(model['logical_args']),digest(declaration),
            digest(dict(reference='original mathematical source',**declaration)),digest(FACTS_PATH.read_bytes()),
            tree_identity(self.packages['rhs_ready_wide']/'compiler_library'),
            digest((self.workspace/'agent_selfcheck.py').read_bytes()),
            digest('complete kernel-call window; host, device and completion inside that window'),
            digest(dict(source=record['source_sha256'],roi='complete kernel-call window')),
            exploration_context_identity(binding['context']),'llvm-mlir','modeled_roi')
        self.domains[identity.digest()]=binding
        return identity

    def hold_inputs(self,request):return nullcontext()

    def evaluate(self,program,receipt,identity):
        from merlin.transforms.evaluated_exploration import CompletedObservation
        matches=[r for r in self.programs.values() if r['source_sha256']==identity.source_sha256
            and r['program_sha256']==digest(program.data)]
        if not matches or any(Path(r['manifest_path']).read_bytes()!=program.data for r in matches):
            raise ValueError('evaluation does not own exact ordinary product')
        feedback=dict(products=matches,numerics='UNKNOWN: declaration custody differs from broker inputs/reference',
            cycles='UNKNOWN: use separate exact paired evidence join',timing_requests=0)
        path=self.output/('observation-'+identity.digest()+'-'+digest(program.data)+'.json')
        path.write_text(json.dumps(feedback,indent=2)+'\n')
        return CompletedObservation(identity,digest(program.data),digest(path.read_bytes()),'unknown','unknown',False,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--incumbent',type=Path,required=True)
    parser.add_argument('--rhs-ready',type=Path,required=True)
    parser.add_argument('--rhs-ready-wide',type=Path,required=True)
    parser.add_argument('--paired-evidence',type=Path)
    args=parser.parse_args();workspace=args.workspace.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    packages={name:getattr(args,option).resolve() for name,option in (
        ('incumbent','incumbent'),('rhs_ready','rhs_ready'),('rhs_ready_wide','rhs_ready_wide'))}
    # All plans/terminal imports come from the same frozen ordinary package.
    package=packages['rhs_ready_wide'];sys.path[:0]=[str(package),str(package/'compiler_library')]
    from gemmini_frontend import normalize_source
    sources=[]
    for path in args.source:
        path=path.resolve();source=path.read_text();model=normalize_source(source)
        products=[o for o in model['operations'] if o['category']=='product']
        if len(products)!=1:raise ValueError('expected one original dense product')
        extents=products[0]['extents']
        sources.append(dict(path=str(path),capsule=path.stem,source_sha256=digest(source.encode()),
            shape=[extents[d] for d in ('m','k','n')]))
    recipe_path=workspace/'.agents/skills/heavy-pass-recipes/scripts/parallel_schedule_recipe.py'
    spec=importlib.util.spec_from_file_location('requested_rhs_parallel_recipe',recipe_path)
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    owner=SourceBoundAdapter(workspace,output,packages,sources)
    with ThreadPoolExecutor(max_workers=2) as executor:
        result=recipe.run([tuple(r['shape']) for r in sources],{},executor=executor,max_in_flight=2,
            directory=output/'parallel',owner=owner)
    if any(tree_identity(packages[name])!=pin for name,pin in owner.package_pins.items()):
        raise ValueError('ordinary package custody changed during recipe.run')
    if any(digest(Path(r['path']).read_bytes())!=r['source_sha256'] for r in sources):
        raise ValueError('original source custody changed during recipe.run')
    evidence=dict(schema='gemmini.rhs_parallel_owner.v1',recipe_path=str(recipe_path),
        recipe_sha256=digest(recipe_path.read_bytes()),adapter_sha256=digest(Path(__file__).read_bytes()),
        mapping=MAPPING,sources=sources,package_pins=owner.package_pins,
        products=list(owner.programs.values()),completion_count=len(result['completions']),
        timing_requests=0,selection='UNKNOWN recipe observations; original paired feedback retained separately')
    if args.paired_evidence:
        paired=json.loads(args.paired_evidence.read_text());joined=[]
        variant_names={'incumbent':'dense-baseline-fresh10a','rhs_ready':'dense-rhs-ready-scalar-fresh10b',
            'rhs_ready_wide':'dense-rhs-ready-wide-scalar-fresh10b'}
        for product in owner.programs.values():
            rows=[r for r in paired['rows'] if r['capsule']==product['capsule'] and
                Path(r['submission']).name==variant_names[product['branch']]]
            # This is a source/producer evidence link, not a recipe input-custody upgrade.
            joined.append(dict(product_sha256=product['program_sha256'],source_sha256=product['source_sha256'],
                branch=product['branch'],paired_rows=rows,recipe_observation_upgrade=False,
                qualification='exact manifest bytes; ordinary variant provenance; actual broker input/reference bytes not exported'))
        evidence['paired_evidence']=dict(path=str(args.paired_evidence.resolve()),sha256=digest(args.paired_evidence.read_bytes()),
            join=joined,comparisons=paired['comparisons'])
    (output/'owner.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(report=str(output/'owner.json'),products=len(owner.programs),
        completions=len(result['completions']),timing_requests=0),indent=2))

if __name__=='__main__':main()
