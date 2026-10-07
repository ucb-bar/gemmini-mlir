"""Source-exact direct widening under coefficient identity, complete group."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/qualify_sparse_dyadic_control.py').read_text()
s=s.replace("w=base/'out/sparse_dyadic_group';d=w/'control'","w=base/'out/exact_coefficient_group';d=w/'candidate'")
s=s.replace('commands=[]','''from merlin.llvmlower.exact_coefficient_widen import prepare_exact_coefficient_widen
original=(dest/'provider.c').read_text()
(dest/'provider.c').write_text(prepare_exact_coefficient_widen(original))
commands=[]''')
s=s.replace("assert (dest/'provider.o').read_bytes()==(normal/'target_numeric/provider.o').read_bytes()","assert (base/'out/sparse_dyadic_group/control/target_numeric/provider.o').read_bytes()==(normal/'target_numeric/provider.o').read_bytes()")
s=s.replace('provider_default_byte_identity=True','provider_default_byte_identity=True,change="source-proved coefficient reconstruction identity only"')
s=s.replace("t=w/'control_strict'","t=w/'strict'").replace("'/spike','--isa", "'/spike','-g','--isa")
s=s.replace("'/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa", "'/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa")
(base/'out/exact_coefficient_group_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
