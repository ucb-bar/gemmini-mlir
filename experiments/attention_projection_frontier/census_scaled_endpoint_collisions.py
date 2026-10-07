"""Read-only collision census of the current source polynomial scale operation."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text()
s=s.replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/scaled_endpoint_census'")
needle="(dest/'provider.c').write_text(s)"
rep='''header=dest/'prepared_polynomial_batch.h';text=header.read_text()
needle='  for(int i=0;i<8;i++) {'
assert text.count(needle)==1
text=text.replace(needle,''' + repr('''  for(int i=0;i<4;i++){
   endpoint_census[0]++;endpoint_census[1]+=x[i].hi<s->cutoff;
   endpoint_census[2]+=x[i].lo==x[i].hi;
   endpoint_census[3]+=scaled[2*i]==scaled[2*i+1];
   endpoint_census[4]+=(scaled[2*i]==scaled[2*i+1]&&x[i].lo>=s->cutoff&&x[i].lo!=x[i].hi);
  }
''')+'''+needle)
header.write_text('static unsigned long long endpoint_census[5];\\n'+text)
s+='\\nvoid get_endpoint_census(unsigned long long*out){for(int i=0;i<5;i++)out[i]=endpoint_census[i];}\\n'
(dest/'provider.c').write_text(s)
'''
s=s.replace(needle,rep).replace("np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))","pass")
s=s.replace("record=dict(scope=","ec=(C.c_uint64*5)();lib.get_endpoint_census(ec)\nrecord=dict(scaled_endpoint_counts=list(ec),scope=")
(base/'out/scaled_endpoint_census_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
