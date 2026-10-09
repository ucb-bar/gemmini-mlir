from pathlib import Path
import hashlib,json,subprocess,ctypes as C
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
from dataclasses import asdict
W=Path(__file__).resolve().parent;R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cap=json.loads((R/'capture/receipt.json').read_text());q=json.loads((R/'capsule/qualification.json').read_text());assert cap['status']=='pass'
assert sha(R/'capsule/baseline.target.ll')==q['cases'][0]['target_sha256']
assert sha(R/'capsule/baseline.native.ll')==q['cases'][0]['native_sha256']
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off'];names=['a','scale_a','b','scale_b'];args=[np.load(R/'capture'/(n+'.npy'))for n in names];expected=np.load(R/'capture/expected.npy')
helper=(W/'observer.c').read_text().split('float exported_original')[0].replace('#include <string.h>\n','').replace('#include <math.h>\n','');(W/'helper.c').write_text(helper)
objects=[];receipts=[]
for name in ['control','candidate']:
 for mode in ['target','native']:
  source=(R/f'capsule/baseline.{mode}.ll').read_text().replace('baseline',name)
  if name=='candidate':
   for i,d,a,b in [(151,143,103,97),(152,144,104,98)]:
    old=f'  %{i} = fmul float %{i-2}, 0x4072590F80000000';assert source.count(old)==1
    source=source.replace(old,f'  %{i} = call float @exported_observer(float %{d}, float %{a}, float %{b}, float 0x4072590F80000000)')
   source+='\ndeclare float @exported_observer(float,float,float,float)\n'
  raw=W/f'{name}.{mode}.raw.ll';raw.write_text(source);final=W/f'{name}.{mode}.ll'
  if name=='candidate':
   helperll=W/f'helper.{mode}.ll';hf=flags if mode=='target' else ['-O3','-fPIC','-ffp-contract=off','-fno-fast-math']
   subprocess.run([str(L/'clang'),*hf,'-S','-emit-llvm',str(W/'helper.c'),'-o',str(helperll)],check=True,capture_output=True)
   subprocess.run([str(L/'llvm-link'),'-S',str(raw),str(helperll),'-o',str(final)],check=True,capture_output=True)
  else:final.write_text(source)
 so=W/(name+'.so');nargv=[str(L/'clang'),'-O3','-fPIC','-shared',str(W/f'{name}.native.ll'),'-lm','-o',str(so)]
 subprocess.run(nargv,check=True,capture_output=True)
 actual=np.zeros_like(expected);HostModel.load(str(so),name=name)([(a.ctypes.data,a.shape)for a in args]+[(actual.ctypes.data,actual.shape)])
 assert np.array_equal(actual,expected),(name,int(np.count_nonzero(actual!=expected)));np.save(W/(name+'.npy'),actual)
 obj=W/(name+'.o');argv=[str(L/'clang'),*flags,'-c',str(W/f'{name}.target.ll'),'-o',str(obj)]
 subprocess.run(argv,check=True,capture_output=True);objects.append(obj);receipts.append(dict(name=name,compile_argv=argv,native_argv=nargv,object_sha256=sha(obj),target_ir_sha256=sha(W/f'{name}.target.ll'),native_ir_sha256=sha(W/f'{name}.native.ll'),original_45056_i8_exact=True));print(name,'ORIGINAL45056_EXACT',flush=True)
# Reuse exact same pinned source companion data bytes.
assert sha(R/'capsule/data.o');objects.append(R/'capsule/data.o')
main=(R/'capsule/main.c').read_text().replace('baseline','control').replace('broadcast','candidate').replace('GATE_PACKET_CYCLES','RECIP_OBSERVER_CYCLES').replace('ORIGINAL_GATE_PACKET','ORIGINAL_RECIP_OBSERVER')
# The existing original host numerical contract is RNE. Refuse rather than run
# this experiment under a different frm; the future compiler pass remains off.
main=main.replace('int main(void){','int main(void){unsigned frm;asm volatile("csrr %0,frm":"=r"(frm));if(frm!=0){printf("RNE_REFUSED\\n");return 9;}')
(W/'main.c').write_text(main);subprocess.run([str(L/'clang'),*flags,'-c',str(W/'main.c'),'-o',str(W/'main.o')],check=True,capture_output=True);objects.append(W/'main.o')
build=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass';(W/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
sp=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(build.elf)],capture_output=True,text=True,timeout=600)
(W/'spike.stdout').write_text(sp.stdout);(W/'spike.stderr').write_text(sp.stderr);assert sp.returncode==0 and 'ORIGINAL_RECIP_OBSERVER PASS' in sp.stdout+sp.stderr
record=dict(schema='experimental_exact_reciprocal_observer_capsule_v1',actual_original_tap_words=45056,dirty_guards=128,immutable_inputs_bytes=405504,source_sha256=sha(R/'source_capsule.mlir'),capture_sha256=sha(R/'capture/receipt.json'),source_scope='Original first preDown stage: only source reciprocal+3 separately roundedmul closed int8RNE observer; same full45056source body including polynomial,loads/stores,allocation/finalcopy.',source_closure_scope='Experimental exact frozen scalar body proof only; no generalized compiler matcher or route installed.',policy='RNE/gradualunderflow/nontrapping/flags+errno unobserved explicit experimental premise; othermodes refused before ROI.',cases=receipts,elf_sha256=sha(build.elf),zero_FSM=True,strict_original_pass=True,rational_native_screen_sha256=sha(W/'rational_native_screen.json'),performance_promotion=False,token_usage_available=False)
(W/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print('STRICT_RECIP_OBSERVER_PASS',flush=True)
raise SystemExit(0)
res=run_on_gsim(build.elf,target='gemmini',timeout_s=1800,max_cycles=100000000,stdout_path=W/'gsim.stdout');(W/'gsim_receipt.json').write_text(json.dumps(asdict(res),indent=2,default=str)+'\n');print(res,flush=True)
