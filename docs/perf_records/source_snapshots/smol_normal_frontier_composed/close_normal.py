from pathlib import Path
import json,hashlib,time,subprocess,os
w=Path(__file__).resolve().parent;r=w.parents[1]
while True:
 p=w/'normal_watch_state.json'
 if p.exists():
  state=json.loads(p.read_text())
  if state['state'] in ('pass','failed','build_terminated_without_result'):
   assert state['state']=='pass',state;break
 time.sleep(15)
p=r/'out/normal_attention_provider_encoded_rows/build/lower/model.ll';n=w/'build/lower/model.ll';a=n.read_text();b=p.read_text();clean=lambda x:'\n'.join(l for l in x.splitlines() if not l.startswith('; ModuleID'))
assert clean(a)==clean(b)
alloc=[l.strip() for l in a.splitlines() if '@malloc(i64 123012992)' in l];calls=[l for l in a.splitlines() if 'call void @ordered_bf16_source_' in l and 'i64 123012928' in l];assert len(alloc)==1 and len(calls)==48
assert all('ptr %5332, ptr %5337, i64 0, i64 123012928, i64 1' in l for l in calls)
release=[l.strip() for l in a.splitlines() if 'call void @free(ptr %5332)' in l];assert len(release)==1
v={'status':'pass','required_bytes':123012928,'alignment':64,'allocated_bytes_including_alignment_padding':123012992,'allocation_count':1,'call_count':48,'same_workspace_ssa':['%5332','%5337'],'release':release,'public_forward_signature_equal_prior_normal':True,'all_other_llvm_bytes_equal_prior':'Exact except ModuleID','files':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in (n,p)]};(w/'workspace_llvm_closure.json').write_text(json.dumps(v,indent=2)+'\n')
subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(w/'archive.py')],check=True)
