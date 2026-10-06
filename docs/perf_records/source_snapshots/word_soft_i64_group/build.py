from pathlib import Path
import json,subprocess,shutil,hashlib,ast
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parent;old=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_i64');core=Path('/scratch/agustin/tmp/merlin-word-soft-domain-20261006');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
header=(core/'merlin/runtime/c/monotone_bit_polynomial.h').read_text();words=header[header.index('/* Explicit alternative enclosure'):header.index('static inline merlin_f32_interval merlin_monotone_bit_polynomial_apply(')]
for name in ['numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir(exist_ok=True)
 for p in (old/name).glob('*.h'):shutil.copyfile(p,d/p.name)
 p=d/'monotone_bit_polynomial.h';s=p.read_text();assert 'merlin_monotone_bit_polynomial_apply_words('not in s;p.write_text(s.replace('static inline merlin_f32_interval merlin_monotone_bit_polynomial_apply(',words+'static inline merlin_f32_interval merlin_monotone_bit_polynomial_apply('));shutil.copyfile(core/'merlin/runtime/c/prepared_softmax_interval.h',d/'prepared_softmax_interval.h')
 s=(old/name/'provider.c').read_text();assert s.count('merlin_monotone_bit_polynomial_apply(x,&root_prepared)')==2 and s.count('merlin_softmax_domain_prepare(')==1;s=s.replace('merlin_monotone_bit_polynomial_apply(x,&root_prepared)','merlin_monotone_bit_polynomial_apply_words(x,&root_prepared)').replace('merlin_softmax_domain_prepare(','merlin_softmax_word_domain_prepare(');(d/'provider.c').write_text(s)
 if name=='numeric_frozen':
  recipe=json.loads((old/name/'reclosure.json').read_text());commands=[]
  for c in recipe['compile_commands']:
   c=[x.replace(str(old/name),str(d))for x in c];subprocess.run(c,check=True);commands.append(c)
  (d/'compile.json').write_text(json.dumps({'commands':commands,'pins':{str(p):h(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
 else:
  m=json.loads((old/name/'manifest.json').read_text());cmd=[x.replace(str(old/name),str(d))for x in m['compile']];subprocess.run(cmd,check=True);shutil.copyfile(old/name/'workspace_frontier_adapter.py',d/'workspace_frontier_adapter.py');m.update(compile=cmd,word_interval_enclosure=True,prepare_softmax_domain=True);m['local_pins']={p.name:h(p)for p in d.iterdir()if p.is_file()and p.suffix in('.h','.c','.so','.py')};(d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
prior=w.parent/'frontier_i64_allocated_pair_v2/candidate';b=json.loads((prior/'build.json').read_text());control=w/'control';candidate=w/'candidate';control.mkdir(exist_ok=True);candidate.mkdir(exist_ok=True)
for arm,d in [('control',control),('candidate',candidate)]:
 link=[str(d/'model.elf')if x==str(prior/'model.elf')else str(w/'numeric_frozen/provider.o')if arm=='candidate'and x.endswith('/frontier_i64_complete_group/candidate/provider.o')else x for x in b['link']];subprocess.run(link,check=True)
 if arm=='control':assert h(d/'model.elf')==h(prior/'model.elf')
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':h(d/'model.elf'),'control_elf_sha256':h(prior/'model.elf'),'same_allocation_aware_driver':True},indent=2)+'\n');print(arm,h(d/'model.elf'),flush=True)
 if arm=='candidate':
  shutil.copyfile(prior/'watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as log:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python','-u',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid,'elf_sha256':h(d/'model.elf')},indent=2)+'\n');print(p.pid)
# Reuse exact current normal i64 host LLVM/object; only separately pinned native numeric library changes.
s=(old/'validate_native.py').read_text();s=s.replace("w=Path(__file__).resolve().parent;h=w/'native';h.mkdir(exist_ok=True)","w=Path('"+str(old)+"');h=Path('"+str(w/'native')+"');h.mkdir(exist_ok=True)");s=s.replace("frozen=w/'native_numeric_frozen'","frozen=Path('"+str(w/'native_numeric_frozen')+"')")
start=s.index("prior=w.parent/'normal_attention_provider/native'");end=s.index("(h/'compiled_model_reuse.json').write_text",start)
s=s[:start]+"prior=w/'native'\nreuse=json.loads((prior/'compiled_model_reuse.json').read_text())\nassert sha(w/'build/lower/model.ll')==reuse['llvm'] and sha(prior/'model.o')==reuse['object']\nshutil.copyfile(prior/'model.o',h/'model.o')\nreuse['scope']='Exact accepted current i64 ordinary host object reused; only native numeric library changes'\n"+s[end:];s=s.replace("'New normal prepared48 group physical descriptor/workspace/fallback implementation; native exact integer product stand-ins, not hardware'","'Frozen accepted current i64 normal host plus explicit word/private-soft numeric composition; native exact integer product stand-ins, not new target or normal binding qualification'");(w/'validate_native.py').write_text(s)
