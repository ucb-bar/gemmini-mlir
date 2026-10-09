"""Companion-only attribution of the already executed candidate PC histogram."""
from pathlib import Path
B=Path(__file__).resolve().parents[2]
s=(B/'experiments/smol_owner_composition/current_pc_attribution.py').read_text()
s=s.replace("out=root/'out/current2072_attribution'","out=root/'out/artifacts/probes/prepared-polynomial-constants/attribution'")
s=s.replace("base=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose')","base=root/'out/artifacts/probes/prepared-polynomial-constants'")
s=s.replace("json.loads((base/'candidate/target_numeric/compile.json').read_text())['commands'][0]","json.loads((base/'candidate/build.json').read_text())['roles']['target_numeric']['commands'][0]")
s=s.replace("json.loads((base/'candidate/build.json').read_text())['link']","json.loads((base/'candidate/target_build.json').read_text())['link']")
s=s.replace("base/'histogram/spike.stderr'","base/'strict/stderr'")
output=B/'out/artifacts/probes/prepared-polynomial-constants/attribution_driver.py';output.write_text(s);exec(compile(s,str(__file__),'exec'))
