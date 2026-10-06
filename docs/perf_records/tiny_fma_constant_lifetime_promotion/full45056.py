from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import FOUR_FEATURE
from merlin.llvmlower.late_quant_rne import rewrite as rne
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
from generic_constant_fma_draft import rewrite
from rv64gc_fma_emitter_draft import RV64GCScalarFmaCapability
W=Path(__file__).resolve().parent;D=W/'full45056';D.mkdir(exist_ok=True);R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');C=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006');L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cycle=json.loads((W.parent.parent.parent.parent/'docs/perf_records/tiny_fma_constant_lifetime_capsule_cycles.json').read_text())
assert cycle['finish_done']and cycle['returncode']==0 and cycle['saved_percent']>0
source=(R/'source_capsule.mlir').read_text().replace(') -> tensor<1x8x5632xi8> {',') -> tensor<1x8x5632xi8> attributes {llvm.emit_c_interface} {');(D/'source.mlir').write_text(source)
ll=lower_to_llvm_ir(source,workdir=D/'lower4',features={FOUR_FEATURE,'lower_fma_to_intrinsic'}).replace('forward','materialized').replace('dealloc_helper','materialized_dealloc_helper')
native,nr=rne(ll,host_isa='portable');target,tr=rne(ll,host_isa='rv64gc',combine_clamp=True);assert len(nr['routes'])==len(tr['routes'])==4
selected,receipt=rewrite(target,emitter=RV64GCScalarFmaCapability().emit,ordinary_nontrapping=True,exception_flags_unobserved=True);assert len(receipt['packets'])==8,len(receipt['packets'])
(D/'native.ll').write_text(native);(D/'target.ll').write_text(selected);(D/'source_packet4.ll').write_text(target);(D/'rewrite.json').write_text(json.dumps(receipt,indent=2)+'\n')
flags=json.loads((W/'prepared.json').read_text())['flags'];nargv=[str(L/'clang'),'-O3','-fPIC','-shared',str(D/'native.ll'),'-lm','-o',str(D/'model.so')];subprocess.run(nargv,check=True,capture_output=True)
args=[np.load(R/'capture'/(n+'.npy'))for n in ['a','scale_a','b','scale_b']];expected=np.load(R/'capture/expected.npy');actual=np.full_like(expected,73);assert actual.size==45056
HostModel.load(str(D/'model.so'),name='materialized')([(x.ctypes.data,x.shape)for x in args]+[(actual.ctypes.data,actual.shape)]);assert np.array_equal(actual,expected);np.save(D/'native_output.npy',actual);print('FULL_ORIGINAL45056_NATIVE_PASS',flush=True)
argv=[str(L/'clang'),*flags,'-c',str(D/'target.ll'),'-o',str(D/'model.o')];subprocess.run(argv,check=True,capture_output=True)
main=(W/'efficient_timing/main.c').read_text().replace('11264','45056').replace('11392','45184').replace('{1,2,5632}','{1,8,5632}');(D/'main.c').write_text(main);subprocess.run([str(L/'clang'),*flags,'-c',str(D/'main.c'),'-o',str(D/'main.o')],check=True,capture_output=True)
objects=[C/'control.o',D/'model.o',R/'capsule/data.o',W/'primitive_source.o',W/'primitive_selected.o',D/'main.o'];b=build_program(objects,D/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass';(D/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
s=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,timeout=120);(D/'spike.stdout').write_text(s.stdout);(D/'spike.stderr').write_text(s.stderr);assert s.returncode==0 and'ORIGINAL_FMA_LIFETIME PASS 45056 128'in s.stdout,s.stdout+s.stderr
r=dict(schema='constant_fma_packet_full_original_capsule_qualification_v1',status='pass',source_shape=[1,8,5632],original_i8_words=45056,native_exact=True,strict_target_exact=True,percall_dirty_destination=True,halo_guard_bytes=128,all_input_bytes_unchanged=405504,source_FMA_steps=8,independent_packet_width=4,source_FMA_input_order_preserved=True,no_FSM=True,actual_GSIM_promotion_receipt_sha256=sha(W.parent.parent.parent.parent/'docs/perf_records/tiny_fma_constant_lifetime_capsule_cycles.json'),target_compile_argv=argv,native_compile_argv=nargv,pins={str(p.relative_to(D)):sha(p)for p in [D/'source.mlir',D/'target.ll',D/'native.ll',D/'model.o',D/'model.so',D/'rewrite.json',D/'main.c',D/'main.o',D/'spike.stdout',D/'spike.stderr',b.elf]},source_capture_receipt_sha256=sha(R/'capture/receipt.json'),scope='Full original45056word first preDown host stage only; original full256000model/Torch gate and stockFireSim remain required. SpikeROI is retiredinstructions, notcycles. Generic draft stillprivate.',token_usage_available=False)
(D/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
