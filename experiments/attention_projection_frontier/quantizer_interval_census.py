"""Diagnostic producer intervals, not a captured-original-output hull."""
from pathlib import Path
import importlib.util
import json
import numpy as np
base=Path(__file__).resolve().parents[2]
w=base/'out/observation_frontier/initial_bounds'
p=Path('/scratch/agustin/tmp/merlin-bf16-quantizer-enclosure-20261007/out/artifacts/quantizer-enclosure/prototype.py')
spec=importlib.util.spec_from_file_location('prototype',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
kw=dict(divisor=127.,epsilon_word=0x3728,qmin=-127,qmax=127)
def words(v):return (v.view(np.uint32)>>16).astype(np.uint16)
rows=[]
for block in range(12):
 a=np.concatenate([np.load(w/f'group_{g:02d}.npy') for g in range(block*4,block*4+4)],axis=2)
 assert a.shape==(3,12,1024,64)
 a=a.transpose(0,2,1,3).reshape(3,1024,768)
 assert np.isfinite(a).all() and np.all(a[0]<=a[1])
 lo,hi,center=[words(mod.bf(v)) for v in a]
 ql,qh,sl,sh=mod.enclose(lo,hi,**kw)
 qc,_,sc,_=mod.enclose(center,center,**kw)
 d=np.maximum(abs(qc.astype(np.int16)-ql),abs(qc.astype(np.int16)-qh))
 np.savez(w/f'quantizer_{block:02d}.npz',lower=ql,upper=qh,candidate=qc,scale_lower=words(sl),scale_upper=words(sh),scale_candidate=words(sc),deviation=d)
 rows.append(dict(block=block,uncertain_q=int(np.count_nonzero(ql!=qh)),uncertain_scale=int(np.count_nonzero(sl!=sh)),uncertain_rows=int(np.count_nonzero(np.any(ql!=qh,axis=1))),deviation_l1_max=int(d.sum(axis=1).max()),deviation_linf_max=int(d.max())))
(w/'quantizer_census.json').write_text(json.dumps(dict(scope='RMS4 source-produced initial interval diagnostic; containment not upgraded, actual source quantizer constants pinned by historical typed closure. Original refinement remains enabled.',rows=rows),indent=2)+'\n');print(json.dumps(rows),flush=True)
