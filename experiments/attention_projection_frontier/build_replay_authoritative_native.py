"""Separately permissioned replay authority; no change to fixed residual estimate."""
from pathlib import Path
import json,shutil,subprocess
from merlin.llvmlower.approximate_residual_rms import ReplayAuthoritativeObservationPolicy,c_replay_header
B=Path(__file__).resolve().parents[2];old=B/'out/artifacts/probes/approximate-residual-rms';W=B/'out/artifacts/probes/replay-authoritative-residual-rms';W.mkdir(parents=True,exist_ok=False)
D=W/'native_numeric';D.mkdir()
for p in (old/'native_numeric').iterdir():
 if p.suffix in ('.c','.h'):shutil.copyfile(p,D/p.name)
(D/'replay_authoritative.h').write_text(c_replay_header(ReplayAuthoritativeObservationPolicy(*([True]*8))))
s=(D/'provider.c').read_text()
s='#include "replay_authoritative.h"\nstatic uint64_t replay_seen,replay_misses,success_stats[8];\nvoid replay_observation_counts(uint64_t*out){out[0]=replay_seen;out[1]=replay_misses;for(int i=0;i<8;i++)out[2+i]=success_stats[i];}\n'+s
needle='if(!(pl[ix]<=value&&value<=ph[ix]))return 0;pl[ix]=ph[ix]=value;'
assert s.count(needle)==1
s=s.replace(needle,'if(!merlin_replay_authoritative_point(value,pl+ix,ph+ix,&replay_seen,&replay_misses))return 0;')
needle=' if(!certify_frontier(w))return 0;'
assert s.count(needle)==1
s=s.replace(needle,needle+'\n for(int i=0;i<3;i++)success_stats[i]+=w->softcounts[i];for(int i=0;i<5;i++)success_stats[3+i]+=w->refinement[i];')
(D/'provider.c').write_text(s)
commands=[]
for cmd in json.loads((old/'build.json').read_text())['commands']:
 cmd=[x.replace(str(old/'native_numeric'),str(D)) for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd)
(W/'build.json').write_text(json.dumps({'commands':commands,'policy':'replay-authoritative PV private observation; unchanged fixed residual RMS estimate','invalidations':'every changed PV partial is in an uncertified row; next frontier pass rebuilds all head endpoints for that row before any observation/publication'},indent=2))
h=W/'native';h.mkdir()
s=(old/'native/run.py').read_text().replace(str(old/'native_numeric'),str(D)).replace(str(old/'native'),str(h))
s=s.replace("'policy': 'explicit source-and-representation residual RMS4; heuristic only'", "'policy':'separate replay-authoritative observation with fixed residual RMS4; heuristic only'")
needle="assert not errors and counts[0]"
idx=s.index(needle)
s=s[:idx]+'''extra=(C.c_uint64*10)();lib.replay_observation_counts(extra)
r['replay_and_success_stats']=list(extra)
r['failed_original_elements']=int(np.count_nonzero(abs(out-g)>(.03125+.02*abs(g))))
r['rms_error']=float(np.sqrt(np.mean(np.square(out.astype(np.float64)-g.astype(np.float64)))))
(h/'validation.json').write_text(json.dumps(r,indent=2)+'\\n');print('REPLAY_RESULT',json.dumps(r),flush=True)
'''+s[idx:]
(h/'run.py').write_text(s)
print(h/'run.py',flush=True)
