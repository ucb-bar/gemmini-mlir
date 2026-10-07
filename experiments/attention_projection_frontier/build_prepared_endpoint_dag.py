"""Source-bound producer/endpoint validation hoist, with checked fallback."""
from pathlib import Path
import json,shutil,subprocess
from merlin.llvmlower.prepared_endpoint_dag import PreparedEndpointContract,c_header
B=Path(__file__).resolve().parents[2];C=B/'out/artifacts/probes/prepared-polynomial-constants/candidate';W=B/'out/artifacts/probes/prepared-endpoint-dag-v3';W.mkdir(parents=True,exist_ok=False)
roles={}
for role in ('native_numeric','target_numeric'):
 D=W/role;D.mkdir()
 for p in (C/role).iterdir():
  if p.suffix in ('.c','.h'):shutil.copyfile(p,D/p.name)
 (D/'prepared_endpoint_dag.h').write_text(c_header(PreparedEndpointContract(*([True]*9))))
 s=(D/'provider.c').read_text()
 anchor='static int endpoint_intervals('
 assert s.count(anchor)==1;s=s.replace(anchor,'#include "prepared_endpoint_dag.h"\n'+anchor)
 old='int rows,const uint8_t*certified){';new='int rows,const uint8_t*certified,const merlin_endpoint_span_owner *owner,const void *owner_epoch){'
 assert s.count(old)==1;s=s.replace(old,new)
 old='  if(certified[r])continue;'
 assert s.count(old)==1;s=s.replace(old,old+'\n  if(merlin_endpoint_prepared_row(owner,owner_epoch,r,alpha+r*2,2,PARTS,dl[r],dh[r],lower+r*DEPTH,upper+r*DEPTH,estimate+r*DEPTH))continue;')
 old='static int certify_frontier(struct attention_workspace *w){'
 assert s.count(old)==1;s=s.replace(old,'static int certify_frontier(struct attention_workspace *w,merlin_endpoint_span_owner *owners,const void *owner_epoch){')
 old='h->out,ROWS,w->row_certified))return 0;'
 assert s.count(old)==1;s=s.replace(old,'h->out,ROWS,w->row_certified,owners+head,owner_epoch))return 0;')
 for old in ['    if(pass&&!h->den_exact[row]){','    if(!h->cell_exact[row*DEPTH+d]){']:
  assert s.count(old)==1;s=s.replace(old,old+'\n     if(!merlin_endpoint_span_invalidate_row(owners+head,owner_epoch,row))return 0;')
 old=' merlin_fma_bound environment=merlin_fma_bound_begin();if(!environment.valid)return 0;\n struct attention_workspace *w=workspace;'
 assert s.count(old)==1
 s=s.replace(old,old+'\n merlin_endpoint_span_owner endpoint_owners[HEADS];\n uint32_t endpoint_maximum[HEADS][ROWS];unsigned char endpoint_dirty[HEADS][ROWS],endpoint_epoch;')
 old='  struct attention_head *h=&w->heads[head];if(!gather_head(h,inputs,head))return 0;'
 assert s.count(old)==1;s=s.replace(old,old+'\n  endpoint_owners[head]=merlin_endpoint_span_begin(&environment,h->partlo,h->parthi,h->centers,endpoint_maximum[head],endpoint_dirty[head],ROWS,ROWS,DEPTH,2*PARTS,&endpoint_epoch);\n  if(!endpoint_owners[head].valid)return 0;')
 old='''   for(int i=0;i<ROWS*DEPTH;i++){
    int ix=(tile*PARTS+part)*ROWS*DEPTH+i;h->partlo[ix]=w->lower[i];h->parthi[ix]=w->upper[i];h->centers[ix]=(float)w->center[i];
   }'''
 assert s.count(old)==1;s=s.replace(old,'   if(!merlin_endpoint_span_record(endpoint_owners+head,&endpoint_epoch,tile*PARTS+part,w->lower,w->upper,w->center))return 0;')
 old=' if(!certify_frontier(w))return 0;';assert s.count(old)==1;s=s.replace(old,' if(!certify_frontier(w,endpoint_owners,&endpoint_epoch))return 0;')
 (D/'provider.c').write_text(s);commands=[]
 for oldcmd in json.loads((C/'build.json').read_text())['roles'][role]['commands']:
  cmd=[x.replace(str(C/role),str(D)) for x in oldcmd];subprocess.run(cmd,check=True);commands.append(cmd)
 roles[role]={'commands':commands}
(W/'build.json').write_text(json.dumps({'roles':roles,'scope':'same numeric policy;producer-issued finite spans and row endpoint range admission;invalidate before refinement;checked quantizer unchanged'},indent=2))
h=W/'native';h.mkdir()
s=(C.parent/'normal_native/run.py').read_text().replace(str(C.parent/'normal_native'),str(h)).replace(str(C/'native_numeric'),str(W/'native_numeric'))
(h/'run.py').write_text(s)
print(h/'run.py',flush=True)
