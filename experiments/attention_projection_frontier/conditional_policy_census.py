"""One predefined BF16-spacing policy; no accuracy labels used for selection."""
from pathlib import Path
import importlib.util,json
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/initial_bounds'
r=json.loads((w/'projection_screen.json').read_text())['rows']
args=resolve_forward_args(Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle'))
spec=importlib.util.spec_from_file_location('numeric','/scratch/agustin/tmp/merlin-bf16-quantizer-enclosure-20261007/out/artifacts/quantizer-enclosure/prototype.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
results=[]
for row in r:
 i=row['block'];q=np.load(w/f'quantizer_{i:02d}.npz');z=np.load(w/f'projection_{i:02d}.npz')
 cs=p.widen(args[row['channel_scale_argument']].view(np.uint16))[None,:];bias=p.widen(args[row['bias_argument']].view(np.uint16))[None,:]
 value=p.bf(p.bf(p.bf(z['center'])*p.widen(q['scale_candidate'])[:,None])*cs);value=p.bf(value+bias)
 cw=(value.view(np.uint32)>>16).astype(np.uint16)
 largest=np.max(cw&0x7fff,axis=1);exponent=np.maximum((largest>>7).astype(int)-134,-133);budget=np.ldexp(np.ones(1024),exponent)
 error=np.maximum(abs(p.widen(z['lower']).astype(np.float64)-value),abs(p.widen(z['upper']).astype(np.float64)-value)).max(axis=1)
 admitted=error<=budget
 results.append(dict(block=i,admitted_rows=int(admitted.sum()),new_admitted_rows=int(admitted.sum())-row['eager_resolved_rows'],continuation_rows=int((~admitted).sum())))
(w/'conditional_policy_census.json').write_text(json.dumps(dict(policy='conditional_bf16_row_max_spacing_v1',results=results),indent=2)+'\n');print(json.dumps(results))
