"""Original full48 native selected route, exact encoder implementation only."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text()
s=s.replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/exact_row_native'")
s=s.replace("(dest/'provider.c').write_text(s)","from merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix\ns=prepare_exact_row_radix(s)\n(dest/'provider.c').write_text(s)")
s=s.replace("np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))","pass")
s=s.replace('Diagnostic snapshots before eager final quantizer refinement; original refinement remains active. RMS4 intervals are approximate, not rigorous source certificates.','Exact finite BF16 row span proof hoists per-word rounding and reconstruction; RMS4/source/consumers/defaults unchanged.')
(base/'out/exact_row_native_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
