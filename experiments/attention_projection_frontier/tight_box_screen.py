"""Best independent activation-box projection bounds, diagnostic cost unpriced."""
from pathlib import Path
import importlib.util,json
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/initial_bounds'
rows=json.loads((w/'projection_screen.json').read_text())['rows']
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005');args=resolve_forward_args(root/'bundle')
spec=importlib.util.spec_from_file_location('numeric','/scratch/agustin/tmp/merlin-bf16-quantizer-enclosure-20261007/out/artifacts/quantizer-enclosure/prototype.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
result=[]
for r in rows:
 index=r['block'];z=np.load(w/f'quantizer_{index:02d}.npz');weight=args[r['weight_argument']].T.astype(np.float64);positive=np.maximum(weight,0);negative=np.minimum(weight,0)
 low=z['lower'].astype(np.float64)@positive+z['upper'].astype(np.float64)@negative
 high=z['upper'].astype(np.float64)@positive+z['lower'].astype(np.float64)@negative
 assert np.max(abs(low))<=2**24 and np.max(abs(high))<=2**24
 sl=p.widen(z['scale_lower'])[:,None];sh=p.widen(z['scale_upper'])[:,None];cs=p.widen(args[r['channel_scale_argument']].view(np.uint16))[None,:];bias=p.widen(args[r['bias_argument']].view(np.uint16))[None,:]
 corners=[p.bf(v*s) for v in (p.bf(low),p.bf(high)) for s in (sl,sh)]
 a,b=np.minimum.reduce(corners),np.maximum.reduce(corners);corners=[p.bf(v*cs) for v in (a,b)]
 a=p.bf(np.minimum.reduce(corners)+bias);b=p.bf(np.maximum.reduce(corners)+bias);resolved=a.view(np.uint32)==b.view(np.uint32)
 eager=np.all(z['lower']==z['upper'],axis=1)&(z['scale_lower']==z['scale_upper'])
 entry=dict(block=index,resolved_outputs=int(resolved.sum()),resolved_rows=int(np.count_nonzero(np.all(resolved,axis=1))),new_complete_rows=int(np.count_nonzero(np.all(resolved,axis=1)&~eager)),extra_logical_MACs=int(4*1024*768*768));result.append(entry);print(json.dumps(entry),flush=True)
(w/'tight_box_screen.json').write_text(json.dumps(dict(scope='Diagnostic sharp independent i8 box only; correlations discarded, RMS4 not made rigorous. Four extra scalar products per projection used to quantify theoretical box limitation, not an admitted performance candidate.',rows=result),indent=2)+'\n')
