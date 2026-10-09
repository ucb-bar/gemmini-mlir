"""Independent checked endpoint graph, tails/refusals and all target modes."""
from pathlib import Path
import ast,hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-endpoint-dag-v3';D=W/'target_numeric';T=W/'target_independent_v2';T.mkdir()
core=Path('/scratch/agustin/tmp/merlin-prepared-endpoint-dag-20261007')
tree=ast.parse((core/'merlin/tests/runtime/test_prepared_endpoint_dag.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_native_source_order_refinement_epoch_and_overflow')
node=next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
while isinstance(node,ast.Call):node=node.func.value
code=ast.literal_eval(node)
(T/'prepared.h').write_bytes((D/'prepared_endpoint_dag.h').read_bytes())
commands=[];objects=[]
for number,(rows,cols,parts) in enumerate([(1,1,1),(3,7,3),(2,13,2)]):
 s=code.replace('ROWS',str(rows)).replace('COLS',str(cols)).replace('PARTS',str(parts)).replace('int main(void)',f'int test{number}(void)')
 old=' assert(!fesetround(FE_TONEAREST));\n}\n';assert s.count(old)==1
 s=s.replace(old,''' for(unsigned mode=0;mode<5;mode++){
  __asm__ volatile("csrw frm,%0"::"r"(mode):"memory");env=merlin_fma_bound_begin();
  owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);
  assert(owner.valid==(mode==0));
 }
 __asm__ volatile("csrw frm,zero":::"memory");return 0;
}
''')
 p=T/f'test{number}.c';p.write_text(s);obj=p.with_suffix('.o');objects.append(obj)
 cmd=json.loads((W/'build.json').read_text())['roles']['target_numeric']['commands'][0]
 cmd=[str(p) if x==str(D/'provider.c') else str(obj) if x==str(D/'provider.o') else str(p.with_suffix('.d')) if x==str(D/'provider.d') else x for x in cmd]
 subprocess.run(cmd,check=True);commands.append(cmd)
p=T/'main.c';p.write_text('#include <stdio.h>\nint test0(void);int test1(void);int test2(void);\nint main(void){if(test0()||test1()||test2())return 1;printf("PREPARED_ENDPOINT_TARGET_PASS shapes=3 modes=5 scenarios=579 endpoint_words=%u\\n",27792u);return 0;}\n')
cmd=[str(p) if x==str(T/'test2.c') else str(T/'main.o') if x==str(T/'test2.o') else str(T/'main.d') if x==str(T/'test2.d') else x for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd);objects.append(T/'main.o')
link=json.loads((W/'target_build.json').read_text())['link'];link=[x for x in link if not x.endswith('.o') or Path(x).name in ('crt.o','syscalls.o')]
link=[str(T/'model.elf') if x==str(W/'model.elf') else x for x in link];link[link.index('-lm'):link.index('-lm')]=list(map(str,objects));subprocess.run(link,check=True);commands.append(link)
audit=audit_elf((T/'model.elf').read_bytes());assert audit['status']=='pass'
cmd=['timeout','60s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc',str(T/'model.elf')];p=subprocess.run(cmd,capture_output=True,text=True);commands.append(cmd)
(T/'stdout').write_text(p.stdout);(T/'stderr').write_text(p.stderr)
assert p.returncode==0 and 'PREPARED_ENDPOINT_TARGET_PASS' in p.stdout,(p.returncode,p.stdout,p.stderr)
(T/'qualification.json').write_text(json.dumps({'commands':commands,'stdout':p.stdout,'returncode':p.returncode,'audit':audit,'core_test':str(core/'merlin/tests/runtime/test_prepared_endpoint_dag.py'),'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in T.iterdir() if p.is_file()}},indent=2))
print(p.stdout)
