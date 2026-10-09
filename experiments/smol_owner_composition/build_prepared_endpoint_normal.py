"""Fresh source-normal endpoint composition; no target execution."""
from pathlib import Path
import json,os,subprocess,shutil,hashlib
B=Path(__file__).resolve().parents[2];OLD=B/'out/artifacts/probes/prepared-polynomial-constants/normal';W=B/'out/artifacts/probes/prepared-endpoint-normal';W.mkdir(exist_ok=False);D=W/'drivers';D.mkdir()
records={}
for oldname,name in [('build_provider.py','build_provider.py'),('prepare_source.py','prepare_source.py'),('build_normal_toolchain.py','build_normal.py'),('validate_native_toolchain.py','validate_native.py')]:
 p=OLD/'drivers'/oldname;s=p.read_text().replace('out/artifacts/probes/prepared-polynomial-constants/normal','out/artifacts/probes/prepared-endpoint-normal')
 if name=='build_provider.py':
  s='from experiments.attention_projection_frontier.bind_prepared_endpoint import bind_source as endpoint_source,bind_headers as endpoint_headers\n'+s
  a="        (dest / 'provider.c').write_text(selected)";assert s.count(a)==1;s=s.replace(a,"        selected=endpoint_source(selected)\n        endpoint_headers(dest)\n"+a)
 if name=='prepare_source.py':
  s='from experiments.attention_projection_frontier.bind_prepared_endpoint import bind_source as endpoint_source\n'+s
  a=' expected=bind_source(prepare_exact_row_radix(expected))';assert s.count(a)==1;s=s.replace(a,a+'\n expected=endpoint_source(expected)')
 (D/name).write_text(s);compile(s,str(D/name),'exec');records[name]={'original':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
env=os.environ.copy();env.update(MERLIN_CHIPYARD='/scratch2/agustin/chipyard',MERLIN_RTL_FACTS=str(B/'out/normal_composition/capability_snapshot/facts.json'),MERLIN_TARGET_CONTRACT=str(B/'out/normal_composition/capability_snapshot/target_contract.yaml'),MERLIN_COMPILER_PYTHON='/scratch/agustin/projects/model2MLIR/.venv/bin/python',MERLIN_MLIR_INSTALL='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install')
(W/'recipe_environment.json').write_text(json.dumps({k:v for k,v in env.items()if k.startswith('MERLIN_')or k=='PYTHONPATH'},indent=2));(W/'driver_binding.json').write_text(json.dumps(records,indent=2))
def run(name,*args):
 with(W/(name+'.log')).open('w')as log:subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(D/name),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
run('build_provider.py','--baseline','/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate','--output',W/'provider')
run('prepare_source.py');shutil.copyfile(W/'source/normal_prepared.mlir',W/'source/normal_prepared_original.mlir')
run('build_normal.py');run('validate_native.py');print('ENDPOINT_NORMAL_BUILD_NATIVE_PASS',flush=True)
