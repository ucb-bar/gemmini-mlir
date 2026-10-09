from pathlib import Path
import ctypes as C,json,hashlib,time
import numpy as np
from merlin.llvmlower.abi import make_descriptor
w=Path(__file__).resolve().parent;h=w/'native';t=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((h/'validation.json').read_text());assert sha(h/'model.so')==r['library_sha256']
lib=C.CDLL(str(h/'model.so'));D=make_descriptor(4);arrays=[np.load(t/f'input_{i}.npy')for i in range(11)];out=np.full((1,12,256,64),0xdead,np.uint16);gold=np.fromfile(t/'endpoint.bin',dtype=np.uint16).reshape(out.shape);before=[hashlib.sha256(x.tobytes()).hexdigest()for x in arrays]
def descriptor(x):
 d=D();d.allocated=x.ctypes.data;d.aligned=x.ctypes.data;d.offset=0
 for i in range(4):d.sizes[i]=x.shape[i];d.strides[i]=x.strides[i]//x.itemsize
 return d
views=[descriptor(x)for x in arrays]+[descriptor(out)];prior=bytes(views[-1]);fn=lib.native_attention_frontier_bridge;fn.argtypes=[C.POINTER(D)]*12+[C.c_void_p];fn.restype=None;start=time.time();fn(*[C.byref(d)for d in views],None)
assert bytes(views[-1])==prior and before==[hashlib.sha256(x.tobytes()).hexdigest()for x in arrays];assert np.array_equal(out,gold)
counts=(C.c_uint64*4)();lib.get_counts(counts);assert counts[1]==1
receipt={'scope':'Actual normal compiled complete original retained-source fallback, forced null workspace; no numeric substitute. Original196608 BF16endpoint bits and descriptor/input ownership preserved.','library_sha256':sha(h/'model.so'),'output_words':out.size,'mismatches':int(np.count_nonzero(out!=gold)),'output_descriptor_unchanged':True,'all_inputs_unchanged':True,'fallback_calls':int(counts[1]),'input_pins':{str(t/f'input_{i}.npy'):sha(t/f'input_{i}.npy')for i in range(11)},'seconds_functional_only':time.time()-start}
(h/'source_fallback_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print('ACTUAL_SOURCE_FALLBACK_PASS',flush=True)
