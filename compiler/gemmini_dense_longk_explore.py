#!/usr/bin/env python3
"""Source-bound PassFamily exploration of inactive bounded long-K alternatives.

Uses the requested shared recipe to generate, apply and deduplicate alternatives.
Every emitted artifact goes through the ordinary checked consumer and must equal
its independent frozen package's ordinary manifest product. No timing is requested.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from gemmini_dense_rhs_parallel import SourceBoundAdapter,digest,tree_identity

MAPPING={'original':'incumbent','batched':'n16','ready':'n32','identical':'incumbent'}


class LongKAdapter(SourceBoundAdapter):
    def binding(self,shape,caps):
        from gemmini_dense import plan_dense,geometry,FACTS_PATH
        from gemmini_dense_coalesced_policy import m1_coalesced_auto_schedule
        from gemmini_dense_longk_experiment import plan_longk_output
        from gemmini_frontend import normalize_source
        from gemmini_readout import source_scales,_relu
        from merlin.transforms.pipeline_state import PipelineContext
        record=next(r for r in self.sources if tuple(r['shape'])==tuple(shape))
        source=Path(record['path']).read_text();model=normalize_source(source)
        readouts=[o for o in model['operations'] if o['category']=='readout']
        if len(readouts)!=1:raise ValueError('requires one original source readout')
        attrs=readouts[0]['attributes'];dtype=attrs.get('output_dtype','i32')
        readout=json.dumps(dict(dtype=dtype,scale=float(source_scales(attrs)),relu=_relu(attrs)),sort_keys=True)
        base,adapter,problem=plan_dense(*shape,readout_identity=readout)
        arguments=dict(attrs=attrs,output_dtype=dtype,outer_order=('m','n'))
        original,mode,evidence=m1_coalesced_auto_schedule(base,adapter,problem,geometry(),FACTS_PATH,**arguments)
        alternatives={'incumbent':(original,mode,evidence)};result_owners={id(original):'incumbent'}
        def transform(result,branch):
            if result is not original:raise ValueError('branch lost original source owner')
            name=MAPPING[branch]
            if name not in alternatives:
                candidate,proof=plan_longk_output(base,adapter,problem,geometry(),FACTS_PATH,
                    column_blocks=int(name[1:]),**arguments)
                candidate.revalidate();alternatives[name]=(candidate,'panel_epochs',proof)
                result_owners[id(candidate)]=name
            return alternatives[name][0]
        def emit(result):
            from gemmini_dense import consume_checked_dense
            from gemmini_backend import compile_source
            name=result_owners[id(result)];_,terminal,proof=alternatives[name]
            with consume_checked_dense(result,adapter,problem,mode=terminal,readout_identity=readout) as consumed:
                _,lowered=compile_source(source)
            payload=(lowered+'\n').encode();manifest=self.output/(record['capsule']+'-'+name+'.manifest.llvm.mlir')
            command=[sys.executable,str(self.packages[name]/'gemmini-opt'),'--convert-iface-to-gemmini',
                '--emit-target-artifact',record['path']]
            done=subprocess.run(command,capture_output=True,check=True);manifest.write_bytes(done.stdout)
            if done.stdout!=payload:raise ValueError('consumed checked effects differ from ordinary manifest: '+name)
            row=dict(capsule=record['capsule'],source_sha256=record['source_sha256'],branch=name,
                original_effect_sha256=original.kernel.digest(),effect_sha256=result.kernel.digest(),
                readout_identity=readout,numerical_identity=adapter.numerical_policy_identity,
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path,action='append')
    parser.add_argument('--incumbent',required=True,type=Path)
    parser.add_argument('--n16',required=True,type=Path);parser.add_argument('--n32',required=True,type=Path)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    packages={k:getattr(args,k).resolve() for k in ('incumbent','n16','n32')}
    # The inherited identity owner names its shared library package via this alias.
    packages['rhs_ready_wide']=packages['n32']
    sys.path[:0]=[str(packages['n32']),str(packages['n32']/'compiler_library')]
    from gemmini_frontend import normalize_source
    from merlin.transforms.pass_exploration import ExplorationGeometry,ValidationPartition
    sources=[]
    for path in args.source:
        path=path.resolve();model=normalize_source(path.read_text());products=[o for o in model['operations'] if o['category']=='product']
        if len(products)!=1:raise ValueError('requires one original rank-two dense product')
        extents=products[0]['extents'];shape=[extents[d] for d in ('m','k','n')]
        sources.append(dict(path=str(path),capsule=path.stem,source_sha256=digest(path.read_bytes()),shape=shape))
    workspace=Path.cwd();owner=LongKAdapter(workspace,output,packages,sources)
    recipe_path=workspace/'.agents/skills/heavy-pass-recipes/scripts/parallel_schedule_recipe.py'
    spec=importlib.util.spec_from_file_location('longk_shared_parallel_recipe',recipe_path)
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        result=recipe.run([tuple(r['shape']) for r in sources],{},executor=executor,max_in_flight=2,
            directory=output/'parallel',owner=owner)
    assert all(tree_identity(packages[k])==pin for k,pin in owner.package_pins.items())
    assert all(digest(Path(r['path']).read_bytes())==r['source_sha256'] for r in sources)
    report=dict(mapping=MAPPING,sources=sources,products=list(owner.programs.values()),package_pins=owner.package_pins,
        recipe_sha256=digest(recipe_path.read_bytes()),adapter_sha256=digest(Path(__file__).read_bytes()),
        completions=len(result['completions']),timing_requests=0,
        status='UNKNOWN recipe numerical/timing observations; use separately pinned ordinary matched feedback')
    (output/'owner.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(output/'owner.json'),products=len(owner.programs),completions=report['completions'])))

if __name__=='__main__':main()
