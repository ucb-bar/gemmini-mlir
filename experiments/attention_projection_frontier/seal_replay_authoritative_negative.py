"""Archive the fixed replay-authoritative policy's original-output failure."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/replay-authoritative-residual-rms'
core=Path('/scratch/agustin/tmp/merlin-replay-authoritative-rms-20261007')
r=json.loads((W/'native/validation.json').read_text())
assert r['calls'][0:3]==[48,11,0] and r['product_calls']==12078 and not r['callback_errors']
assert r['failed_original_elements']==84 and not r['allclose']
golden=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy')
compiled=B/'out/artifacts/probes/prepared-polynomial-constants/normal_native/output.npy'
g=np.load(golden);c=np.load(compiled);v=np.load(W/'native/output.npy')
assert np.array_equal(g.view('u4'),c.view('u4'))
assert np.count_nonzero(abs(v-g)>(.03125+.02*abs(g)))==84
paths=[p for p in W.rglob('*') if p.is_file()]
paths += [core/'src/merlin/llvmlower/approximate_residual_rms.py',core/'merlin/tests/runtime/test_approximate_residual_rms.py',core/'docs/reference/approximate_residual_rms.md',Path(__file__),B/'experiments/attention_projection_frontier/build_replay_authoritative_native.py',golden,compiled,B/'docs/perf_records/approximate_residual_rms_native_negative.json']
record={'schema':'source_replay_authoritative_native_negative_v1','status':'REJECTED_ORIGINAL_OUTPUT_GATE','core_head':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'policy':'fixed two-signed-byte representation; unchanged residual RMS4 factor and source RMS4; separately permissioned PV replay authority','source_nonidentity':'heuristic estimates never establish a source enclosure or equality; already-computed ordered PV replay replaces a private estimate even outside it','dependent_rebuild':'only unfinished private rows are refined; next frontier pass recomputes every head endpoint of each such row, then row scale/bin observations before publication','original_final_gate':{'atol':.03125,'rtol':.02,'elements':1600,'compiled_reference_equals_quantized_torch_capture_bits':True,'failed_elements':84},'native':r,'telemetry_scope':{'replay_and_miss_counters':'all attempted groups','eight_source_counters':'successful selected groups only; excludes11 failed attempts and original fallback work'},'limitations':['37 selected groups;11 source fallbacks','84 original gate failures even with fallback; no admission','no target cost or hardware campaign for failed policy','no precision/factor/threshold changes','original private normal pool capacity retained; no fresh whole ABI claim'],'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
out=B/'docs/perf_records/replay_authoritative_residual_rms_native_negative.json';out.write_text(json.dumps(record,indent=2)+'\n');print(out,len(record['pins']))
