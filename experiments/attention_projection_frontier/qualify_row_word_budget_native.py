"""One prepared row word budget; original full48 native gate, no thresholds."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text()
s=s.replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/row_word_budget_native'")
s=s.replace("(dest/'provider.c').write_text(s)",'''from merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix
from merlin.llvmlower.prepared_row_word_budget import prepare_row_word_budget
s=prepare_row_word_budget(prepare_exact_row_radix(s))
s=s.replace('static int certify_frontier(struct attention_workspace *w){','static unsigned long long row_budget_counts[8];\\nvoid diagnostic_row_counts(unsigned long long *out){for(int i=0;i<8;i++)out[i]=row_budget_counts[i];}\\nstatic int certify_frontier(struct attention_workspace *w){')
needle=' return 1;\\n}\\n/* Diagnostic counters are valid only following success on this workspace. */'
assert s.count(needle)==1
s=s.replace(needle,' for(int i=0;i<3;i++)row_budget_counts[i]+=w->softcounts[i];for(int i=0;i<5;i++)row_budget_counts[i+3]+=w->refinement[i];\\n'+needle)
(dest/'provider.c').write_text(s)''')
s=s.replace("np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))","pass")
s=s.replace("(h/'qualification.json').write_text", "aggregate=(C.c_uint64*8)();lib.diagnostic_row_counts(aggregate);record['aggregate_numeric_stats']=list(aggregate)\n(h/'qualification.json').write_text")
s=s.replace('Diagnostic snapshots before eager final quantizer refinement; original refinement remains active. RMS4 intervals are approximate, not rigorous source certificates.','Prepared common source-row polynomial word radius; original RMS4 approximate score policy, denominator order and ambiguous replay retained.')
(base/'out/row_word_budget_native_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
