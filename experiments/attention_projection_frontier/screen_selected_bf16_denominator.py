"""One fixed approximate selected-denominator native whole-model screen."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text()
s=s.replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/selected_bf16_denominator'")
s=s.replace("(dest/'provider.c').write_text(s)",'''from merlin.llvmlower.selected_bf16_denominator import SelectedBF16DenominatorPolicy,prepare_selected_denominator
policy=SelectedBF16DenominatorPolicy(*([True]*6))
s=prepare_selected_denominator(s,policy)
(dest/'provider.c').write_text(s)
(h/'policy.json').write_text(json.dumps(dict(name='selected_bf16_ordered_denominator_v1',permissions=vars(policy),selection='fixed source graph; no golden values or threshold selection',scope='BF16 probability values replace original F32 polynomial values only in ordered denominator; original interval work retained in this feasibility screen',gate=dict(atol=.03125,rtol=.02)),indent=2)+'\\n')''')
s=s.replace("np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))","pass  # No duplicate endpoint arrays needed for this numerical screen.")
s=s.replace("record=dict(scope=", "record=dict(elementwise_failures=int(np.count_nonzero(np.abs(out-gold)>.03125+.02*np.abs(gold))),max_abs=float(np.max(np.abs(out-gold))),rms=float(np.sqrt(np.mean((out.astype(float)-gold)**2))),scope=")
s=s.replace("Diagnostic snapshots before eager final quantizer refinement; original refinement remains active. RMS4 intervals are approximate, not rigorous source certificates.","Explicit selected BF16 denominator graph; not an enclosure of original F32 denominator. Source products/nonlinears/order and original whole gate unchanged; old interval costs retained, no performance claim.")
s=s.replace("assert snapshots==48 and not errors and product_calls==23040 and not record['bit_mismatches'] and counts[1]==0","assert snapshots==48 and not errors and product_calls==23040 and counts[1]==0")
(base/'out/selected_bf16_denominator_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
