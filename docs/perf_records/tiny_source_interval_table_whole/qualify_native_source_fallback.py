from pathlib import Path
import ctypes,hashlib,json,subprocess
import numpy as np
from merlin.runtime.backends.spike_model import _transform_host_ir
from mlir_oot.late_quant_rne import merlin_host_llvm_transform

T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';H=W/'host_llvm'
ORIGINAL=T.parent/'tiny-rectangular-whole-20261006/whole'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
G=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/capture')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
case=W/'control';case.mkdir(exist_ok=False)
selected,hook=_transform_host_ir(ORIGINAL/'lower/model.ll',case/'host_llvm',merlin_host_llvm_transform(LLVM,combine_clamp=True))
assert sha(selected)==sha(ORIGINAL/'host_llvm/model.ll')
assert sha(case/'host_llvm/model.native.ll')==sha(ORIGINAL/'host_llvm/model.native.ll')
commands=[]
for suffix in ('','native.'):
 source=selected if not suffix else case/'host_llvm/model.native.ll'
 out=case/('expanded.'+suffix+'ll');bridge=B/('host_llvm/expanded_bridge.'+suffix+'ll')
 argv=[str(LLVM/'llvm-link'),'-S',str(source),str(bridge),'-o',str(out)];commands.append(argv);subprocess.run(argv,check=True,capture_output=True)
 assert sha(out)==sha(ORIGINAL/('host_llvm/expanded.'+suffix+'ll'))
argv=[str(LLVM/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-c',str(case/'expanded.ll'),'-o',str(case/'model.o')];commands.append(argv);subprocess.run(argv,check=True,capture_output=True)
assert sha(case/'model.o')==sha(ORIGINAL/'model.o')
lib=ctypes.CDLL(str(W/'host/model.so'));env=ctypes.CDLL(None)
source=getattr(lib,'forward.extracted.531.__source_interval_original');selected=getattr(lib,'forward.extracted.531')
source.argtypes=selected.argtypes=[ctypes.c_void_p]*5
inputs=[np.ascontiguousarray(np.load(G/name)) for name in ('a.npy','scale_a.npy','b.npy','scale_b.npy')]
original_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in inputs]
expected=np.load(G/'expected.npy').reshape(-1)
def call(fn):
 guarded=np.full(45056+128,73,np.int8);address=guarded.ctypes.data+64
 fn(*[ctypes.c_void_p(a.ctypes.data) for a in inputs],ctypes.c_void_p(address))
 assert np.all(guarded[:64]==73) and np.all(guarded[-64:]==73)
 return guarded[64:-64].copy()
old=env.fegetround();events=[]
try:
 for mode in (0,0x400,0x800,0xc00):
  assert env.fesetround(mode)==0
  for flags in (0,1,4,8,16,32,61):
   env.feclearexcept(61);env.feraiseexcept(flags);a=call(source);af=env.fetestexcept(61)
   env.feclearexcept(61);env.feraiseexcept(flags);b=call(selected);bf=env.fetestexcept(61)
   assert np.array_equal(a,b)
   if mode:assert af==bf
   else:assert np.array_equal(b,expected)
   events.append({'mode':mode,'sticky_preset':flags,'all45056source_words_exact':True,'all128guards':True,'control_flags':af,'candidate_flags':bf,'flag_observation_contract':'exactoriginal onnonRNE fallback; RNEflags explicitlyunobserved'})
finally:assert env.fesetround(old)==0
assert original_hashes==[hashlib.sha256(a.tobytes()).hexdigest() for a in inputs]
record={'schema':'normal_source_interval_control_and_native_fallback_v1','control_normal_target_native_LLVM_expanded_objects_byteidentical2004':True,'all_original45056input_hashes_and_guards':True,'M8source_fallback_native_modes':events,'RNEflag_equality_required':False,'FP_numeric_policy_unchanged':True,'defaultoff':'Source_interval_table selection absent yields originalnormal2004OOTrne expandedIR/objectbyteidentical. No modelsourcefilemodified.','actual_object_sha256':sha(case/'model.o'),'normal_control_hook':hook,'commands':commands,'pins':{str(p):sha(p) for p in [Path(__file__),ORIGINAL/'model.o',ORIGINAL/'host_llvm/model.ll',ORIGINAL/'host_llvm/model.native.ll',ORIGINAL/'host_llvm/expanded.ll',ORIGINAL/'host_llvm/expanded.native.ll',W/'host/model.so',H/'source_binding.json',G/'receipt.json',*case.glob('*'),*case.glob('host_llvm/*'),*(G/name for name in ('a.npy','scale_a.npy','b.npy','scale_b.npy','expected.npy'))] if p.is_file()}}
(W/'control_and_fallback.json').write_text(json.dumps(record,indent=2)+'\n');print('NORMAL_DEFAULT_BYTEEXACT_AND_M8_SOURCE_FALLBACK_WORDS_FLAGS_GUARDS_PASS',flush=True)
