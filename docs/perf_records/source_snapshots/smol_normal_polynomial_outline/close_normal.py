from pathlib import Path
import json,hashlib,time,subprocess
w=Path(__file__).resolve().parent
while True:
 p=w/'normal_watch_state.json'
 if p.exists():
  state=json.loads(p.read_text())
  if state['state'] in ('pass','failed','build_terminated_without_result'):
   assert state['state']=='pass',state;break
 time.sleep(10)
p=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_frontier_composed/build/lower/model.ll');n=w/'build/lower/model.ll';a=n.read_text();b=p.read_text();lines=a.splitlines()
alloc=[(i,l.strip())for i,l in enumerate(lines)if '@malloc(i64 123012992)' in l];assert len(alloc)==1
index,line=alloc[0];owner=line.split(' = ')[0];chain=[l.strip() for l in lines[index+1:index+6]];names=[l.split(' = ')[0] for l in chain]
expected=[f'{names[0]} = ptrtoint ptr {owner} to i64',f'{names[1]} = add i64 {names[0]}, 63',f'{names[2]} = urem i64 {names[1]}, 64',f'{names[3]} = sub i64 {names[1]}, {names[2]}',f'{names[4]} = inttoptr i64 {names[3]} to ptr'];assert chain==expected
aligned=names[4];span=f'ptr {owner}, ptr {aligned}, i64 0, i64 123012928, i64 1'
calls=[(i,l)for i,l in enumerate(lines)if 'call void @ordered_bf16_source_' in l and 'i64 123012928' in l];assert len(calls)==48 and all(span in l for _,l in calls)
release=[(i,l.strip())for i,l in enumerate(lines)if f'call void @free(ptr {owner})' in l];assert len(release)==1 and index<min(i for i,_ in calls) and release[0][0]>max(i for i,_ in calls)
for name in ['forward','_mlir_ciface_forward']:
 signature=lambda text:[l for l in text.splitlines()if l.startswith('define ')and '@'+name+'(' in l]
 assert len(signature(a))==1 and signature(a)==signature(b)
clean=lambda x:'\n'.join(l for l in x.splitlines()if not l.startswith('; ModuleID'))
v={'status':'pass','required_bytes':123012928,'alignment':64,'allocated_bytes_including_alignment_padding':123012992,'allocation_count':1,'call_count':48,'same_workspace_ssa':[owner,aligned],'allocation':line,'alignment_chain':chain,'release':[l for _,l in release],'public_forward_signature_equal_prior_normal':True,'host_llvm_equal_except_module_id':clean(a)==clean(b),'host_compilation_scope':'Fresh published-main normal route; LLVM differs from earlier isolated-core normal build. Fresh original full1600 native and retained-source fallback qualify this route; no whole target performance attribution.','files':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}for x in(n,p)]}
(w/'workspace_llvm_closure.json').write_text(json.dumps(v,indent=2)+'\n')
subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(w/'archive.py')],check=True)
