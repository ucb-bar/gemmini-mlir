from pathlib import Path
import json,hashlib,subprocess,shutil
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parent;control=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/smol-minmax-20261006/core_selected');base=w.parent/'closed_group_endpoint';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=(control/'provider.c').read_text();original=s;changes=[]
def replace(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new);changes.append((old,new))
header='''#include <stdint.h>
static uint64_t diagnostic_stage[10],diagnostic_calls[10];
static uint64_t diagnostic_tick(void){uint64_t v;__asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
static void diagnostic_add(int i,uint64_t begin){diagnostic_stage[i]+=diagnostic_tick()-begin;diagnostic_calls[i]++;}
void group_provider_profile(uint64_t *out){for(int i=0;i<10;i++){out[i]=diagnostic_stage[i];out[10+i]=diagnostic_calls[i];}}
'''
replace(' if(!encode_operand(w->a,m,k,0,w->arf,w->ar,w->ap,w->astep)||',' uint64_t diagnostic_at=diagnostic_tick();\n if(!encode_operand(w->a,m,k,0,w->arf,w->ar,w->ap,w->astep)||')
replace(' for(int i=0;i<m*n;i++)w->center[i]=0;',' diagnostic_add(1,diagnostic_at);diagnostic_at=diagnostic_tick();\n for(int i=0;i<m*n;i++)w->center[i]=0;\n diagnostic_add(3,diagnostic_at);')
replace('  if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,degree))return 0;','  diagnostic_at=diagnostic_tick();\n  if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,degree))return 0;\n  diagnostic_add(2,diagnostic_at);diagnostic_at=diagnostic_tick();')
replace('  for(int i=0;i<m*n;i++)w->center[i]+=(double)w->readout[i]*weight;','  for(int i=0;i<m*n;i++)w->center[i]+=(double)w->readout[i]*weight;\n  diagnostic_add(3,diagnostic_at);')
replace(' for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->center[r*n+c]*=w->astep[r]*w->bstep[c];',' diagnostic_at=diagnostic_tick();\n for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->center[r*n+c]*=w->astep[r]*w->bstep[c];\n diagnostic_add(3,diagnostic_at);diagnostic_at=diagnostic_tick();')
replace(' return dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms);',' int diagnostic_result=dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms);\n diagnostic_add(4,diagnostic_at);return diagnostic_result;')
replace('struct attention_head *h=&w->heads[head];if(!gather_head(h,inputs,head))return 0;','struct attention_head *h=&w->heads[head];uint64_t diagnostic_at=diagnostic_tick();if(!gather_head(h,inputs,head))return 0;diagnostic_add(0,diagnostic_at);')
line='  if(!soft_details(h->q,h->k,h->mask,h->qlo,h->qhi,h->p,h->plo,h->phi,h->denlo,h->denhi,h->alpha,ROWS,w->softcounts,h->ylo,h->yhi,h->maxima))return 0;'
replace(line,'  diagnostic_at=diagnostic_tick();\n'+line+'\n  diagnostic_add(5,diagnostic_at);')
line=' if(!certify_frontier(w))return 0;'
replace(line,' uint64_t diagnostic_frontier=diagnostic_tick();\n'+line+'\n diagnostic_add(6,diagnostic_frontier);')
replace(' merlin_dot_norms *bn=scratch->bn,*en=scratch->en;', ' uint64_t rhs_begin=diagnostic_tick();\n merlin_dot_norms *bn=scratch->bn,*en=scratch->en;')
replace(' for(int r=0;r<m;r++){\n  merlin_dot_norms an=', ' diagnostic_add(7,rhs_begin);\n for(int r=0;r<m;r++){\n  uint64_t lhs_begin=diagnostic_tick();\n  merlin_dot_norms an=')
replace('  for(int j=0;j<n;j++){\n   double e=', '  diagnostic_add(8,lhs_begin);\n  uint64_t cells_begin=diagnostic_tick();\n  for(int j=0;j<n;j++){\n   double e=')
replace('  }\n }return 1;\n}\n\nstatic int soft_details', '  }\n  diagnostic_add(9,cells_begin);\n }return 1;\n}\n\nstatic int soft_details')
check=s
for old,new in reversed(changes):assert check.count(new)==1;check=check.replace(new,old)
assert check==original
(w/'provider.c').write_text(header+s)
commands=[]
for cmd in json.loads((control/'compile.json').read_text())['commands']:
 # Only the translation unit/output location changes; all actual headers stay frozen.
 cmd=[str(w/Path(x).name)if x in [str(control/n)for n in ['provider.c','provider.o','provider.ll','provider.d']]else x for x in cmd]
 subprocess.run(cmd,check=True);commands.append(cmd)
driver_control=base/'descriptor_bridge_target_marked';driver=(driver_control/'driver.c').read_text();driver='extern void group_provider_profile(unsigned long long*);\n'+driver
old=' printf("WORKSPACE_GROUP ALL%d AND GUARDS PASS\\n",196608);return 0;'
new=' unsigned long long stages[20];group_provider_profile(stages);for(int i=0;i<10;i++)printf("PROVIDER_STAGE %d %llu %llu\\n",i,stages[i],stages[i+10]);\n'+old
assert driver.count(old)==1;driver=driver.replace(old,new);(w/'driver.c').write_text(driver)
cmd=json.loads((driver_control/'capsule_build.json').read_text())['compile'];cmd=[x.replace(str(driver_control),str(w))for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd)
link=json.loads((control/'compile.json').read_text())['link'];link=[str(w/'model.elf')if x==str(control/'model.elf')else str(w/'provider.o')if x==str(control/'provider.o')else str(w/'driver.o')if x.endswith('/driver.o')else x for x in link];subprocess.run(link,check=True)
audit=audit_elf((w/'model.elf').read_bytes());assert audit['status']=='pass';(w/'model.nofsm.json').write_text(json.dumps(audit,indent=2)+'\n')
record={'scope':'Diagnostic stage instrumentation of qualified complete original12-head minmax provider. Counter removal exactly restores original C; driver adds postROI prints. Counters are Spike retired instructions until actual hardware.','control_elf_sha256':sha(control/'model.elf'),'control_provider_c_sha256':sha(control/'provider.c'),'counter_removal_source_exact':True,'elf_sha256':sha(w/'model.elf'),'compile':commands,'link':link,'stage_names':['input_gather','radix_preparation','product_callbacks','integer_degree_reconstruction','dot_bounds','source_softmax_intervals','consumer_certificate_and_refinement'],'pins':{str(Path(x).resolve()):sha(x)for x in link if Path(x).is_file()},'token_usage_available':False};(w/'build.json').write_text(json.dumps(record,indent=2)+'\n')
shutil.copyfile(base/'numeric_capability_both_target/watch.py',w/'watch.py');print('PROFILE_BUILT',record['elf_sha256'])
