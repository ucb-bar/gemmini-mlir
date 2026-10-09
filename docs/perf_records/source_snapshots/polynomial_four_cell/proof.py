from pathlib import Path
import json,subprocess,hashlib,sys
from mlir_oot.host_outward_fp import emit_fixed_outward_f64_header
from mlir_oot.no_fsm_audit import audit_elf
sys.path.insert(0,'/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006/merlin/tests/runtime')
from test_prepared_polynomial_batch import C
w=Path(__file__).parent.resolve();d=w/'independent_target';d.mkdir(exist_ok=True);old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(d/'exact_provider.h').write_text(emit_fixed_outward_f64_header(name='certificate_f64',host_isa='rv64gc'))
source=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/frontier_i64_allocated_pair_v2/driver.c').read_text().replace('int main(void){','int unused_group_main(void){')
C=C.replace('#include <fenv.h>','static int getfrm(void){unsigned x;__asm__ volatile("frrm %0":"=r"(x));return x;}\nstatic void setfrm(int x){__asm__ volatile("fsrm %0"::"r"(x):"memory");}').replace('int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};','int modes[]={0,1,2,3,4};').replace('fegetround()','getfrm()').replace('fesetround(','setfrm(')
import random
rng=random.Random(921);values=[[-88,-87.3365478515625,-87.3365478515625,-87.,-0.,0.,0.,0.],[-1,-.999999,-2,-1.999999,-10,-9.999,-70,-69.9],[float('nan'),0,float('-inf'),0,0,float('inf'),1,-1]]
for _ in range(1000):
 v=[]
 for _ in range(4):
  lo=rng.uniform(-89,0);v.extend((lo,min(0,lo+rng.choice([0,1e-6,.01,.5]))))
 values.append(v)
def cf(x):
 import math
 return 'NAN'if math.isnan(x)else '-INFINITY'if x==-math.inf else 'INFINITY'if x==math.inf else float(x).hex()+'f'
table='\nstatic const float cases[][8]={\n'+',\n'.join('{'+','.join(cf(x)for x in v)+'}'for v in values)+'\n};\n'
source+='\n'+C+table+'int main(void){const float plan[]={-87.3365478515625f,1.4426950216293335f,-.079204238951206207f,-.22433836758136749f,.30354261398315430f,.00010703434963943437f,8388608.f,1065353216.f};unsigned failures=0;for(unsigned m=0;m<5;m++)for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++)for(int invalid=0;invalid<2;invalid++)if(probe(plan,cases[i],m,invalid)!=1)failures++;printf("POLYNOMIAL_FOUR %u\\n",failures);return failures!=0;}\n';(d/'proof.c').write_text(source)
c=json.loads((old/'numeric_frozen/compile.json').read_text())['commands'][0];c=[str(d/'proof.c')if x==str(old/'numeric_frozen/provider.c')else str(d/'proof.o')if x==str(old/'numeric_frozen/provider.o')else str(d/'proof.d')if x==str(old/'numeric_frozen/provider.d')else str(d/'exact_provider.h')if x==str(old/'numeric_frozen/fixed_outward_f64.h')else x for x in c];c.insert(1,'-I/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006/merlin/runtime/c');c.insert(1,'-frounding-math');c.insert(1,'-I/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/frontier_i64_allocated_pair_v2');subprocess.run(c,check=True)
link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(d/'proof.o')if x.endswith('/frontier_i64_allocated_pair_v2/driver.o')else x for x in link];subprocess.run(link,check=True);a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n')
with(d/'spike.log').open('w')as f:p=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(d/'model.elf')],stdout=f,stderr=subprocess.STDOUT,timeout=120)
assert p.returncode==0 and 'POLYNOMIAL_FOUR 0'in(d/'spike.log').read_text()
(d/'receipt.json').write_text(json.dumps({'status':'pass','rc':p.returncode,'independent_interval_batches':len(values),'actual_frm_modes':5,'endpoint_comparisons':len(values)*5*2*4,'compile':c,'link':link,'elf_sha256':h(d/'model.elf'),'log_sha256':h(d/'spike.log'),'source_rounding_unchanged':True,'signed_zero_exact':True,'invalid_nonfinite_and_non_RNE':'Checked scalar fallback matches; non-RNE preparation refuses fast admission'},indent=2)+'\n');print('PASS',len(values)*5*2*4)
