from pathlib import Path
import ctypes,json,hashlib,subprocess
import numpy as np
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';case=W/'native_direct_fenv';case.mkdir(exist_ok=False)
G=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/capture')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=case/'around_call.c';c.write_text('''#include <fenv.h>
typedef void(*helper)(void*,void*,void*,void*,void*);
int around_call(helper fn,void*a,void*sa,void*b,void*sb,void*out,int mode,int sticky,int*before,int*after){
 int old=fegetround();if(fesetround(mode))return 1;feclearexcept(FE_ALL_EXCEPT);if(feraiseexcept(sticky))return 2;
 *before=fetestexcept(FE_ALL_EXCEPT);fn(a,sa,b,sb,out);*after=fetestexcept(FE_ALL_EXCEPT);
 if(fesetround(old))return 3;return 0;
}
''')
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
argv=[clang,'-O2','-fPIC','-shared',str(c),'-lm','-o',str(case/'around_call.so')];subprocess.run(argv,check=True,capture_output=True)
lib=ctypes.CDLL(str(W/'host/model.so'));wrapper=ctypes.CDLL(str(case/'around_call.so')).around_call
wrapper.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2+[ctypes.POINTER(ctypes.c_int)]*2;wrapper.restype=ctypes.c_int
source=getattr(lib,'forward.extracted.531.__source_interval_original');candidate=getattr(lib,'forward.extracted.531')
inputs=[np.ascontiguousarray(np.load(G/name))for name in ('a.npy','scale_a.npy','b.npy','scale_b.npy')];pins=[hashlib.sha256(x.tobytes()).hexdigest()for x in inputs]
expected=np.load(G/'expected.npy').reshape(-1);events=[]
for mode in (0,0x400,0x800,0xc00):
 for preset in (0,1,4,8,16,32,61):
  outs=[np.full(45056+128,73,np.int8)for _ in range(2)];flags=[]
  for fn,out in zip((source,candidate),outs):
   before=ctypes.c_int();after=ctypes.c_int()
   assert wrapper(ctypes.cast(fn,ctypes.c_void_p),*[ctypes.c_void_p(x.ctypes.data)for x in inputs],ctypes.c_void_p(out.ctypes.data+64),mode,preset,ctypes.byref(before),ctypes.byref(after))==0
   flags.append((before.value,after.value))
  assert np.array_equal(outs[0],outs[1]);assert all(np.all(x[:64]==73)and np.all(x[-64:]==73)for x in outs)
  assert flags[0][0]&preset==preset and flags[1][0]&preset==preset
  if mode:assert flags[0]==flags[1]
  else:assert np.array_equal(outs[1][64:-64],expected)
  events.append(dict(mode=mode,sticky_preset=preset,source_before=flags[0][0],source_after=flags[0][1],candidate_before=flags[1][0],candidate_after=flags[1][1],all45056words_and128guards_exact=True,nonRNEflags_required=bool(mode)))
assert pins==[hashlib.sha256(x.tobytes()).hexdigest()for x in inputs]
record=dict(schema='normal_source_interval_M8_compiled_around_call_fenv_v1',status='pass',all45056originalwords_andinputhashes=True,events=events,prior_test_limitation='Original control_and_fallback.json allocated NumPy output after preset, so its stickyflag evidence was insufficient. This independent compiledC adapter performs preset/read around actualhelper with no intervening Python/NumPy operations. Prior originaloutput/defaultLLVM/object evidence remains unchanged.',RNEflags_unobserved=True,compile_argv=argv,pins={str(p):sha(p)for p in [Path(__file__),c,case/'around_call.so',W/'host/model.so',W/'control_and_fallback.json',*(G/name for name in ('a.npy','scale_a.npy','b.npy','scale_b.npy','expected.npy'))]})
(case/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print('NORMAL_M8_COMPILED_DIRECT_STICKY_FLAGS_ALL_MODES_GUARDS_PASS',flush=True)
