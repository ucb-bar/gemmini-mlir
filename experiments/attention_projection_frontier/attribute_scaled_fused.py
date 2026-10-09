"""Source attribution of the completed exact-scaling screen; no rerun."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'experiments/smol_owner_composition/current_pc_attribution.py').read_text()
s=s.replace("out=root/'out/current2072_attribution'", "out=root/'out/artifacts/probes/scaled-fused-radix/attribution'")
s=s.replace("base=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose')", "base=root/'out/artifacts/probes/scaled-fused-radix'")
s=s.replace("json.loads((base/'candidate/target_numeric/compile.json').read_text())['commands'][0]", "json.loads((base/'build.json').read_text())['roles']['target_numeric']['commands'][0]")
s=s.replace("base/'candidate/target_numeric/provider.o'", "base/'target_numeric/provider.o'")
s=s.replace("base/'candidate/build.json'", "base/'target_build.json'").replace("base/'candidate/model.elf'", "base/'model.elf'")
s=s.replace("base/'histogram/spike.stderr'", "base/'strict/stderr'")
(B/'out/artifacts/probes/scaled-fused-radix/attribution_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
