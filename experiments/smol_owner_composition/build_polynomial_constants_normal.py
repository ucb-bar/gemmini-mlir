"""Fresh normal source composition with immutable polynomial context selection."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'experiments/smol_owner_composition/build_exact_row_normal.py').read_text()
s=s.replace("'out/exact_row_normal'","'out/artifacts/probes/prepared-polynomial-constants/normal'")
s=s.replace("/'out/exact_row_normal", "/'out/artifacts/probes/prepared-polynomial-constants/normal")
anchor="        if name == 'prepare_source.py':"
extra='''        if name == 'build_provider.py':
            selected = 'from experiments.attention_projection_frontier.bind_polynomial_constants import bind_source,bind_headers\\n' + selected
            selected = replace_once(selected, "        (dest / 'provider.c').write_text(selected)", "        selected=bind_source(selected)\\n        bind_headers(dest)\\n        (dest / 'provider.c').write_text(selected)")
'''
assert s.count(anchor)==1;s=s.replace(anchor,extra+anchor)
anchor='        selected = selected.replace("\'build_v6\'", "\'build_v1\'")'
extra='''        if name == 'prepare_source.py':
            selected = 'from experiments.attention_projection_frontier.bind_polynomial_constants import bind_source\\n' + selected
            selected = replace_once(selected, " expected=prepare_exact_row_radix(expected)", " expected=bind_source(prepare_exact_row_radix(expected))")
'''
assert s.count(anchor)==1;s=s.replace(anchor,extra+anchor)
s=s.replace("'explicit_option': 'exact_row_radix_pack'", "'explicit_option': 'exact_row_radix_pack + immutable rounded polynomial context'")
output=B/'out/artifacts/probes/prepared-polynomial-constants/normal_driver.py';output.write_text(s);exec(compile(s,str(__file__),'exec'))
