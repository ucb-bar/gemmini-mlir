from pathlib import Path
import ast,hashlib,json,shutil,subprocess
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/benchmark-buffer-20261006'
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
shutil.copy2(core/'merlin/runtime/c/benchmark_buffer.h',work/'benchmark_buffer.h')
shutil.copy2(core/'merlin/tests/runtime/test_benchmark_buffer.py',work/'native_fixture.py')
module=ast.parse((work/'native_fixture.py').read_text())
source=next(ast.literal_eval(item.value)for item in module.body if isinstance(item,ast.Assign)and any(isinstance(target,ast.Name)and target.id=='SOURCE'for target in item.targets))
cost=r'''
#include <stdio.h>
static unsigned char cost_a[8192] __attribute__((aligned(__alignof__(uint64_t))));
static unsigned char cost_b[8192] __attribute__((aligned(__alignof__(uint64_t))));
static __attribute__((noinline)) size_t byte_difference(const void*a,const void*b,size_t n){
 const unsigned char*x=a,*y=b;
 for(size_t i=0;i<n;i++)if(x[i]!=y[i])return i;
 return n;
}
static __attribute__((noinline)) size_t word_difference(const void*a,const void*b,size_t n){
 return merlin_benchmark_first_difference(a,b,n);
}
static void measure(void){
 for(size_t i=0;i<8192;i++)cost_a[i]=cost_b[i]=(unsigned char)(i*73+19);
 for(size_t offset=0;offset<2;offset++){
  unsigned long start,end;
  __asm__ volatile("rdinstret %0":"=r"(start)::"memory");
  for(int i=0;i<16;i++)assert(byte_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);
  __asm__ volatile("rdinstret %0":"=r"(end)::"memory");
  printf("BENCH_BYTE offset%lu instructions%lu\n",(unsigned long)offset,end-start);
  __asm__ volatile("rdinstret %0":"=r"(start)::"memory");
  for(int i=0;i<16;i++)assert(word_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);
  __asm__ volatile("rdinstret %0":"=r"(end)::"memory");
  printf("BENCH_WORD offset%lu instructions%lu\n",(unsigned long)offset,end-start);
 }
 printf("BENCHMARK_BUFFER EXACT PASS\n");
}
'''
source=source.replace('int main(void){',cost+'\nint main(void){').replace(' return 0;\n}', ' measure();\n return 0;\n}')
assert source.count('measure();')==1
(work/'check.c').write_text(source)
previous=root/'out/artifacts/probes/smol-absolute-values-20261006/representation_proof'
receipt=json.loads((previous/'build.json').read_text())
command=receipt['compile'];command=command[:command.index('-I')]+['-I',str(work),'-c','-MD','-MF',str(work/'check.d'),str(work/'check.c'),'-o',str(work/'check.o')]
subprocess.run(command,check=True)
link=[arg.replace(str(previous/'absolute.elf'),str(work/'check.elf')).replace(str(previous/'absolute.o'),str(work/'check.o'))for arg in receipt['link']]
subprocess.run(link,check=True)
audit=audit_elf((work/'check.elf').read_bytes());assert audit['status']=='pass'
deps=(work/'check.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
(work/'build.json').write_text(json.dumps({'scope':'Exactportablebenchmarkchecker, allnativefixturealignments/tails/mismatch/dirtyguards onactualRV64GC; isolatedreadcheck retiredinstructions, notkernel/hardwarecycles.','compile':command,'link':link,'pins':{str(Path(path).resolve()):sha(path)for path in[*command,*link,*deps,str(work/'native_fixture.py'),str(work/'prepare.py')]if Path(path).is_file()},'nofsm':audit},indent=2)+'\n')
print(json.dumps({'elf':sha(work/'check.elf'),'nofsm':audit['status']}))
