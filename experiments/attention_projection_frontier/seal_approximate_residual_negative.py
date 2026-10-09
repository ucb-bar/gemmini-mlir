"""Preserve a refused approximation independently of final fallback accuracy."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/approximate-residual-rms'
core=Path('/scratch/agustin/tmp/merlin-approximate-residual-rms-20261007')
validation=json.loads((W/'native/validation.json').read_text());diagnostic=json.loads((W/'diagnostic/validation.json').read_text())
assert validation['calls'][0:3]==[48,47,0] and validation['product_calls']==11820
assert diagnostic['calls'][0:3]==[48,47,0] and diagnostic['product_calls']==11820
assert validation['bitwise_mismatches']==diagnostic['bitwise_mismatches']==0
assert 'PROVIDER_REFUSAL line=363' in (W/'diagnostic/stderr.log').read_text()
paths=[p for p in W.rglob('*') if p.is_file()]
paths += [core/'src/merlin/llvmlower/approximate_residual_rms.py',core/'merlin/tests/runtime/test_approximate_residual_rms.py',core/'docs/reference/approximate_residual_rms.md',Path(__file__),B/'experiments/attention_projection_frontier/build_approximate_residual_native.py']
paths += [Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007/src/merlin/llvmlower/two_signed_byte_block_float.py')]
record={'schema':'source_approximate_residual_native_negative_v1','status':'REJECTED_SELECTED_COVERAGE','core_head':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'policy':'Fixed4/sqrt(K) residual-L2 product estimate plus unchanged source RMS4; heuristic, not enclosure/equality','representation':'fixed2 signed radix256 bytes;3 degree callbacks/4 pair products; no precision or threshold tuning','original_final_gate':{'atol':.03125,'rtol':.02,'elements':1600},'native':validation,'diagnostic':diagnostic,'first_refusal':{'source_line':363,'function':'exact_source_partials','condition':'pl[ix] <= ordered_source_dot && ordered_source_dot <= ph[ix]','interpretation':'retained local source containment fails; not whole accuracy failure'},'limitations':['47 original source fallbacks invalidate selected-path admission','final bitexact is fallback-assisted','no target or hardware experiment','no performance result','historical native pool capacity retained; no refreshed whole ABI claim'],'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
out=B/'docs/perf_records/approximate_residual_rms_native_negative.json';out.write_text(json.dumps(record,indent=2)+'\n');print(out,len(record['pins']))
