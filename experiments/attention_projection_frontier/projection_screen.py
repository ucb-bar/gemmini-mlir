"""Complete twelve-boundary source-interval feasibility; no policy admission."""
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
from xdsl.dialects import func
from xdsl.ir import BlockArgument
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.integer_projection_enclosure import prepare_projection_weight_bounds
from merlin.llvmlower.bf16_projection_enclosure import enclose_projection_epilogue
from merlin.llvmlower.source_numeric_capability import SourceNumericContract

base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/initial_bounds'
audit=Path('/scratch/agustin/tmp/gemmini-attention-boundary-audit-20261007/out/boundaries/projection_generic_final.json')
a=json.loads(audit.read_text());src=Path(a['source']);assert hashlib.sha256(src.read_bytes()).hexdigest()==a['source_sha256']
module=parse_mlir_text(src.read_text());f=next(o for o in module.body.block.ops if isinstance(o,func.FuncOp) and o.sym_name.data=='forward');ops=list(f.body.block.ops)
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005');args=resolve_forward_args(root/'bundle')
prototype=Path('/scratch/agustin/tmp/merlin-bf16-quantizer-enclosure-20261007/out/artifacts/quantizer-enclosure/prototype.py');spec=importlib.util.spec_from_file_location('numeric',prototype);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def scalarops(op):return [x.name for x in op.body.block.ops]
def peel(v):
 while v.owner.name in ('tensor.expand_shape','tensor.collapse_shape'):v=v.owner.operands[0]
 return v.owner
rows=[];start=time.time()
for index,c in enumerate(a['closures']):
 mm=ops[c['integer_projection_position']];target=ops[c['target_position']]
 assert scalarops(target)==['arith.addf','linalg.yield']
 assert isinstance(target.inputs[1],BlockArgument)
 bias_index=target.inputs[1].index
 channel=peel(target.inputs[0]);assert scalarops(channel)==['arith.mulf','linalg.yield']
 assert isinstance(channel.inputs[1],BlockArgument);channel_index=channel.inputs[1].index
 row=channel.inputs[0].owner;assert scalarops(row)==['arith.mulf','linalg.yield']
 cast=row.inputs[0].owner;assert scalarops(cast)==['arith.sitofp','linalg.yield'] and cast.inputs[0] is mm.results[0]
 trans=mm.inputs[1].owner;assert trans.name=='linalg.transpose' and list(trans.permutation.get_values())==[1,0] and isinstance(trans.inputs[0],BlockArgument)
 weight_index=trans.inputs[0].index
 weights=np.ascontiguousarray(args[weight_index].T);assert weights.dtype==np.int8 and weights.shape==(768,768)
 channel_words=np.asarray(args[channel_index]).view(np.uint16);bias_words=np.asarray(args[bias_index]).view(np.uint16)
 assert channel_words.shape==bias_words.shape==(768,)
 plan=prepare_projection_weight_bounds(weights.tobytes(),reduction=768,columns=768,source_min=-127,source_max=127)
 z=np.load(w/f'quantizer_{index:02d}.npz');ql=z['lower'];qh=z['upper'];qc=z['candidate'];delta=z['deviation']
 center=qc.astype(np.float64)@weights.astype(np.float64);assert np.all(center==np.rint(center))
 radius=np.minimum(delta.sum(axis=1,dtype=np.int64)[:,None]*np.asarray(plan.column_linf),delta.max(axis=1).astype(np.int64)[:,None]*np.asarray(plan.column_l1))
 low=np.maximum(center-radius,np.asarray(plan.source_lower));high=np.minimum(center+radius,np.asarray(plan.source_upper))
 assert np.max(abs(low))<=2**24 and np.max(abs(high))<=2**24
 sl=p.widen(z['scale_lower'])[:,None];sh=p.widen(z['scale_upper'])[:,None];cs=p.widen(channel_words)[None,:];bias=p.widen(bias_words)[None,:]
 # Exact operation sequence; interval multiplication uses all four corners.
 b=[p.bf(v) for v in (low,high)]
 corners=[p.bf(v*s) for v in b for s in (sl,sh)]
 lower=np.minimum.reduce(corners);upper=np.maximum.reduce(corners)
 corners=[p.bf(v*cs) for v in (lower,upper)]
 lower=p.bf(np.minimum.reduce(corners)+bias);upper=p.bf(np.maximum.reduce(corners)+bias)
 words=lambda x:(x.view(np.uint32)>>16).astype(np.uint16)
 lw,uw=words(lower),words(upper)
 for r,j in [(0,0),(0,767),(511,253),(1023,767)]:
  expected=enclose_projection_epilogue((int(low[r,j]),int(high[r,j])),(int(z['scale_lower'][r]),int(z['scale_upper'][r])),int(channel_words[j]),int(bias_words[j]),contract=SourceNumericContract(*([True]*7)))
  assert expected==(int(lw[r,j]),int(uw[r,j]))
 resolved=lw==uw;eager=np.all(ql==qh,axis=1)&(z['scale_lower']==z['scale_upper'])
 record=dict(block=index,weight_argument=weight_index,channel_scale_argument=channel_index,bias_argument=bias_index,weight_sha256=plan.weight_sha256,resolved_projection_outputs=int(resolved.sum()),total_projection_outputs=resolved.size,fully_resolved_rows=int(np.count_nonzero(np.all(resolved,axis=1))),eager_resolved_rows=int(eager.sum()),new_complete_rows=int(np.count_nonzero(np.all(resolved,axis=1)&~eager)),uncertain_projection_outputs=int(np.count_nonzero(~resolved)),max_radius=int(radius.max()))
 rows.append(record);print(json.dumps(record),flush=True)
 np.savez(w/f'projection_{index:02d}.npz',lower=lw,upper=uw,center=center,radius=radius)
record=dict(scope='Source-produced RMS4 interval feasibility, no source-containment promotion; immutable weight metadata and exact conditional integer theorem, sequenced BF16 endpoint rounding. Original exact continuation cost not eliminated merely by partially resolved projection outputs.',source_sha256=a['source_sha256'],rows=rows,seconds=time.time()-start)
(w/'projection_screen.json').write_text(json.dumps(record,indent=2)+'\n')
