"""Matched complete source pointwise capsule for existing generic loop outlining."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.llvm_loop_outline import outline_loops
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf

W = Path(__file__).resolve().parent
O = W.parents[3]
OLD = Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006/cost_m2')
R = Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006')
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
FLAGS = ['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
save = lambda p, x: Path(p).write_text(json.dumps(x, indent=2, default=str)+'\n')
run = lambda argv: subprocess.run([str(x) for x in argv], check=True, capture_output=True)
objects = []
cases = []
args = [np.load(R/'capture'/(name+'.npy')) for name in ['a','scale_a','b','scale_b']]
args[0] = args[0][:,:2,:].copy()
args[2] = args[2][:,:2,:].copy()
expected = np.load(R/'capture/expected.npy')[:,:2,:].copy()
original_inputs = [x.copy() for x in args]

# The original packet2 source body is retained. Apply exactly the existing
# bounded-RNE policy to both arms before the source-exact outlining alternative.
for mode in ['native','target']:
    raw = (OLD/f'control.{mode}.ll').read_text()
    control, proof = rewrite(raw, host_isa='portable' if mode=='native' else 'rv64gc', combine_clamp=mode=='target')
    (W/f'control.{mode}.ll').write_text(control)
    candidate = outline_loops(control.replace('control','outlined'), W/f'{mode}_outline')
    source = W/f'outlined.{mode}.premerge.ll'
    source.write_text(candidate)
    out = W/f'outlined.{mode}.ll'
    run([L/'opt','-S','-passes=mergefunc,verify',source,'-o',out])
    save(W/f'{mode}_rne.json',proof)

for name in ['control','outlined']:
    so = W/(name+'.so')
    run([L/'clang','-O3','-shared','-fPIC','-ffp-contract=off',W/f'{name}.native.ll','-lm','-o',so])
    result = np.full_like(expected, 73)
    HostModel.load(str(so), name=name)([(x.ctypes.data,x.shape) for x in args]+[(result.ctypes.data,result.shape)])
    assert np.array_equal(result,expected)
    assert all(np.array_equal(a,b) for a,b in zip(args,original_inputs))
    np.save(W/(name+'.npy'),result)
    obj = W/(name+'.o')
    argv = [L/'clang',*FLAGS,'-c',W/f'{name}.target.ll','-o',obj]
    run(argv)
    objects.append(obj)
    cases.append(dict(name=name,native_words_exact=11264,native_sha256=sha(so),target_object_sha256=sha(obj),target_llvm_sha256=sha(W/f'{name}.target.ll'),compile_argv=argv))
    print(name,'NATIVE11264_EXACT',flush=True)

main = (O/'out/artifacts/probes/fma-constant-lifetime-20261006/efficient_timing/main.c').read_text()
start = main.index('extern void source_probe0')
end = main.index('int main(void)',start)
main = (main[:start]+main[end:]).replace('materialized','outlined').replace('FMA_LIFETIME_CYCLES','OUTLINE_COMPLETE_CYCLES').replace('ORIGINAL_FMA_LIFETIME','ORIGINAL_OUTLINE_COMPLETE')
strict = '''
static int8_t reference[11264];
static int check_rounding(struct d3 *da,struct d1 *sa,struct d3 *db,struct d1 *sb,struct d3 *out){
 for(unsigned frm=0;frm<5;frm++){
  unsigned a_flags,b_flags;
  poison();cursor=0;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");
  _mlir_ciface_control(da,sa,db,sb,out);asm volatile("csrr %0,fflags":"=r"(a_flags)::"memory");
  for(unsigned i=0;i<11264;i++)reference[i]=guarded[64+i];
  poison();cursor=0;asm volatile("csrw fflags,%0"::"r"(8):"memory");
  _mlir_ciface_outlined(da,sa,db,sb,out);asm volatile("csrr %0,fflags":"=r"(b_flags)::"memory");
  for(unsigned i=0;i<11264;i++)if(reference[i]!=guarded[64+i]){printf("OUTLINE_FRM_BITS_FAIL %u %u\\n",frm,i);return 1;}
  for(unsigned i=0;i<64;i++)if(guarded[i]!=73||guarded[11264+64+i]!=73){printf("OUTLINE_FRM_GUARD_FAIL\\n");return 2;}
  if(a_flags!=b_flags){printf("OUTLINE_FRM_FLAGS_FAIL %u %u %u\\n",frm,a_flags,b_flags);return 3;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 printf("OUTLINE_FIVE_FRM_FLAGS PASS 56320\\n");return 0;
}
'''
strict_main = main.replace('int main(void){',strict+'\nint main(void){')
needle='uint64_t input_hash[4]='
strict_main = strict_main.replace(needle,'if(check_rounding(&da,&sa,&db,&sb,&out))return 12;\n '+needle)
for suffix,text in [('strict',strict_main),('timing',main)]:
    src = W/f'{suffix}_main.c'
    src.write_text(text)
    obj = W/f'{suffix}_main.o'
    run([L/'clang',*FLAGS,'-c',src,'-o',obj])
    build = build_program([*objects,R/'capsule/data.o',obj],W/(suffix+'_build'),target='gemmini',max_loaded_bytes=None)
    audit = audit_elf(build.elf.read_bytes())
    assert audit['status']=='pass'
    save(W/f'{suffix}_audit.json',audit)
    spike = subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(build.elf)],capture_output=True,text=True,timeout=300)
    (W/f'{suffix}_spike.log').write_text(spike.stdout+spike.stderr)
    assert spike.returncode==0 and 'ORIGINAL_OUTLINE_COMPLETE PASS' in spike.stdout+spike.stderr
    if suffix=='strict':
        assert 'OUTLINE_FIVE_FRM_FLAGS PASS' in spike.stdout+spike.stderr
        print('OUTLINE_STRICT_COMPLETE_FIVE_FRM_PASS',flush=True)
    else:
        record=dict(schema='generic_source_exact_loop_outline_capsule_v1',shape=[1,2,5632],original_native_i8_words=11264,strict_compared_words=56320,all5_frm_and_sticky_flags_match=True,dirty_guard_bytes=128,immutable_inputs_bytes=405504,cases=cases,source_sha256=sha(OLD/'source.mlir'),control_raw_llvm_sha256=sha(OLD/'control.target.ll'),scope='Complete original two-row preDown packet2 body incl dequantization/polynomial/reciprocal/three rounded multiplies/quantization/loads/stores/allocated temporary/final copy. Existing RNE policy identical in both arms; only generic loop-extract/noinline and exact mergefunc alter code organization.',whole_hardware_prediction=None,zero_FSM=True,elf_sha256=sha(build.elf),token_usage_available=False)
        save(W/'capsule_qualification.json',record)
        print('OUTLINE_TIMING_READY',flush=True)
        result=run_on_gsim(build.elf,target='gemmini',timeout_s=1800,max_cycles=30000000,stdout_path=W/'gsim.stdout')
        save(W/'gsim_receipt.json',asdict(result))
        console=(W/'gsim.stdout').read_text()
        assert 'ORIGINAL_OUTLINE_COMPLETE PASS' in console
        rows=[[int(x) for x in line.split()[1:]] for line in console.splitlines() if line.startswith('OUTLINE_COMPLETE_CYCLES ')]
        assert [(i,a) for i,a,c in rows]==[(0,0),(1,1),(2,1),(3,0)]
        old=[c for i,a,c in rows if a==0]
        new=[c for i,a,c in rows if a==1]
        final={**record,'paired_complete_gsim_cycles':rows,'before_mean':sum(old)/2,'after_mean':sum(new)/2,'saved_percent':100*(sum(old)-sum(new))/sum(old),'DONE':True,'console_sha256':sha(W/'gsim.stdout'),'decision':'Retain positive local code-organization screen for full qualification.'if sum(new)<sum(old)else'Reject promotion: complete local cost loses despite whole code-size reduction.'}
        save(W/'capsule_result.json',final)
        print('OUTLINE_COMPLETE_RESULT',rows,final['saved_percent'],flush=True)
