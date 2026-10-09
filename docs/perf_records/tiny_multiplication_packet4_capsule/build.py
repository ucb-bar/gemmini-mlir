from pathlib import Path
import os,json,hashlib,subprocess
from dataclasses import asdict
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.broadcast_math_hoist import FEATURE as HOIST
from merlin.llvmlower.scalar_pointwise_packet import MULTIPLY_FOUR_FEATURE
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent;N=W.parent/'tiny-broadcast-math-20261005'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
F=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
R=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/abi/mlir_runtime.c')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
llvm=lower_to_llvm_ir((N/'source.mlir').read_text(),workdir=W/'lower',features={HOIST,MULTIPLY_FOUR_FEATURE}).replace('forward','multiply_packet').replace('dealloc_helper','multiply_packet_dealloc_helper')
native,nproof=rewrite(llvm,host_isa='portable');target,tproof=rewrite(llvm,host_isa='rv64gc',combine_clamp=True)
(W/'native.ll').write_text(native);(W/'target.ll').write_text(target)
subprocess.run([str(LLVM/'clang'),'-O3','-shared','-fPIC','-ffp-contract=off',str(W/'native.ll'),str(R),'-lm','-o',str(W/'model.so')],check=True,capture_output=True)
a=np.fromfile(N/'activation.bin',np.float32).reshape(1,8,2048);w=np.fromfile(N/'weight.bin',np.float32);expected=np.fromfile(N/'expected.bin',np.int8).reshape(a.shape)
outs=[]
for path,name in [(N/'hoisted/hoisted.so','hoisted'),(W/'model.so','multiply_packet')]:
 out=np.full(a.shape,19,np.int8);HostModel.load(str(path),name=name)([(v.ctypes.data,v.shape)for v in[w,a,out]])
 assert np.array_equal(out,expected);outs.append(out)
assert np.array_equal(*outs);print('MULTIPLY_NATIVE_16384_EXACT',flush=True)
subprocess.run([str(LLVM/'clang'),*flags,'-c',str(W/'target.ll'),'-o',str(W/'model.o')],check=True,capture_output=True)
base=(N/'main.c').read_text().replace('_mlir_ciface_hoisted','_mlir_ciface_CANDIDATE').replace('_mlir_ciface_control','_mlir_ciface_hoisted').replace('_mlir_ciface_CANDIDATE','_mlir_ciface_multiply_packet')
# Add immutable input identity check before and after every complete call; outside ROI.
base=base.replace('int main(void){','static uint64_t input_hash(void){uint64_t h=1469598103934665603ull;const uint8_t*p=(const uint8_t*)weight;for(unsigned i=0;i<8192;i++){h^=p[i];h*=1099511628211ull;}p=(const uint8_t*)activation;for(unsigned i=0;i<65536;i++){h^=p[i];h*=1099511628211ull;}return h;}\nint main(void){const uint64_t original_hash=input_hash();')
base=base.replace('printf("SOURCE_NORM_COMPLETE PASS\\n");','if(input_hash()!=original_hash){printf("INPUT_FAIL\\n");return 6;}printf("SOURCE_NORM_COMPLETE PASS\\n");')
(W/'strict_main.c').write_text(base)
subprocess.run([str(LLVM/'clang'),*flags,'-ffp-contract=off','-c',str(W/'strict_main.c'),'-o',str(W/'strict_main.o')],check=True,capture_output=True)
shared=[N/'hoisted/model.o',W/'model.o',N/'data.o',F/'mlir_rt.o']
b=build_program([*shared,W/'strict_main.o'],W/'strict_build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass'
sp=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(b.elf)],capture_output=True,text=True,timeout=300)
console=sp.stdout+sp.stderr;(W/'strict.log').write_text(console);assert sp.returncode==0 and 'SOURCE_NORM_COMPLETE PASS' in console,console[-3000:]
q=dict(schema='source_exact_hoisted_multiplication_packet4_v1',source_shape=[1,8,2048],native_words=16384,native_bitexact=True,strict_words=16384,strict_all5_rounding_modes_and_sticky_flags_match=True,dirty_output_guard_bytes=128,immutable_original_input_bytes=73728,scope='Complete actual source hoisted normalization; original1880runtime/data/control object. Only tensor pure f32 multiplication consumer scheduling changes. No new rsqrt or arithmetic permission.',strict_elf_sha256=b.elf_sha256,audit=audit,source_pins={str(p):sha(p)for p in [N/'source.mlir',N/'weight.bin',N/'activation.bin',N/'expected.bin',W/'target.ll',W/'native.ll',W/'strict_main.c']},object_pins={str(p):sha(p)for p in shared},native_rne=nproof,target_rne=tproof,token_usage_available=False)
(W/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print('MULTIPLY_STRICT5_EXACT',flush=True)
start=base.index('for(unsigned m=0;m<5;m++)');end=base.index('mode(0);for(unsigned r=0;r<2;r++)',start);short=base[:start]+base[end:]
(W/'timing_main.c').write_text(short);subprocess.run([str(LLVM/'clang'),*flags,'-ffp-contract=off','-c',str(W/'timing_main.c'),'-o',str(W/'timing_main.o')],check=True,capture_output=True)
b=build_program([*shared,W/'timing_main.o'],W/'timing_build',target='gemmini',max_loaded_bytes=None);assert audit_elf(b.elf.read_bytes())['status']=='pass'
s=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(b.elf)],capture_output=True,text=True,timeout=120);(W/'timing_spike.log').write_text(s.stdout+s.stderr);assert s.returncode==0 and 'SOURCE_NORM_COMPLETE PASS' in s.stdout+s.stderr
print('MULTIPLY_TIMING_STRICT_PASS',flush=True)
r=run_on_gsim(b.elf,target='gemmini',timeout_s=1200,max_cycles=12000000,stdout_path=W/'gsim.stdout')
(W/'gsim_receipt.json').write_text(json.dumps(asdict(r),indent=2,default=str)+'\n');text=(W/'gsim.stdout').read_text();assert 'SOURCE_NORM_COMPLETE PASS' in text,text[-3000:]
rows=[[int(x)for x in line.split()[1:]]for line in text.splitlines()if line.startswith('NORM_COMPLETE_CYCLES ')]
assert [(round_,arm)for round_,arm,c in rows]==[(0,0),(0,1),(1,1),(1,0)]
old=[c for _,a,c in rows if a==0];new=[c for _,a,c in rows if a==1]
res={**q,'timing_elf_sha256':b.elf_sha256,'paired_complete_gsim_cycles':rows,'before_mean_cycles':sum(old)/2,'after_mean_cycles':sum(new)/2,'saved_percent':100*(sum(old)-sum(new))/sum(old),'gsim_receipt':asdict(r),'gsim_console_sha256':sha(W/'gsim.stdout'),'decision':'Positive local screen; whole-model gates still required.'if sum(new)<sum(old)else'Reject performance promotion.','whole_cycle_prediction':None}
(W/'result.json').write_text(json.dumps(res,indent=2,default=str)+'\n');print('MULTIPLY_COMPLETE_GSIM_RESULT',rows,res['saved_percent'],flush=True)
