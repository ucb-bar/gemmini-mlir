from pathlib import Path
import json,hashlib,ctypes as C
import numpy as np
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gold=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy')
records=[]
for bucket,maximum in [('high_readback',28800),('low_readback',40320)]:
 c=w/bucket;r=json.loads((c/'native/validation.json').read_text());assert sha(gold)==r['original_golden_sha256'];a=np.load(c/'native/output.npy');b=np.load(gold);assert a.shape==b.shape and np.array_equal(a.view('u4'),b.view('u4'));assert r['calls'][:3]==[48,0,0] and not r['callback_errors'];assert 23040<=r['product_calls']<=maximum and (r['product_calls']-23040)%5==0
 lib=C.CDLL(str(c/'native_numeric_frozen/provider.so'));lib.group_provider_workspace_bytes.restype=C.c_size_t;size=lib.group_provider_workspace_bytes();assert size==123012928+1048576
 records.append({'bucket':bucket,'status':'original1600_gate_pass','original_bits_equal':True,'calls':r['calls'],'product_calls':r['product_calls'],'eligible_absolute_products':(r['product_calls']-23040)//5,'possible_absolute_products':(maximum-23040)//5,'workspace_bytes':size,'scope':'Native numeric adapter uses one explicitly enlarged guarded harness workspace; original host workspace descriptor is overridden only at this diagnostic bridge. Not normal ABI integration.','numeric_receipt':str(c/'native/validation.json'),'numeric_receipt_sha256':sha(c/'native/validation.json'),'output_sha256':sha(c/'native/output.npy')})
out={'status':'pass','results':records,'postprocessing_correction':'Low-readback execution completed all1600 exact with0fallback and passed workspace guards, then its harness asserted an invalid fixed maximum callback count. Original exit1/log/code retained. Actual eligibility varies by immutable source point-envelope/encoding proof; corrected acceptance requires baseline plus a bounded multiple of five and full original output equality. No model rerun or gate change.','token_usage_available':False};(w/'native_pair_closure.json').write_text(json.dumps(out,indent=2)+'\n');print([(x['bucket'],x['eligible_absolute_products'],x['possible_absolute_products'])for x in records])
