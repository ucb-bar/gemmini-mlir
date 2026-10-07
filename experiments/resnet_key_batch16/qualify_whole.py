"""Original whole1000 native and strict gates for ordinary and controlled arms."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
from mlir_oot.no_fsm_audit import audit_elf
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['normal','controlled'],required=True);a=ap.parse_args()
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/key_batch16_normal';llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');spike=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('whole_gate',B/'tests/fused_whole_model_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
gate=W/('qualification_'+a.arm);gate.mkdir();build=W/'normal/build_direct'
if a.arm=='controlled':
 build=gate/'build_view';build.mkdir()
 for p in(W/'controlled2101/candidate').iterdir():
  if p.name!='device_catalog':(build/p.name).symlink_to(p.resolve())
 (build/'device_catalog').mkdir();old=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal/controlled2095/qualification/build_view/device_catalog/device_catalog.json');catalog=json.loads(old.read_text());oracle=Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_normal_binding_v1/key_bundle/native_oracle.c');fresh=W/'key_bundle/native_oracle.c';assert catalog['native_oracle_sources'].count(str(oracle))==1
 catalog['native_oracle_sources']=[str(fresh)if x==str(oracle)else x for x in catalog['native_oracle_sources']]
 catalog['controlled_key_batch16_binding']=dict(previous=str(old),previous_sha256=sha(old),previous_oracle_sha256=sha(oracle),new_oracle_sha256=sha(fresh),target_link_sha256=sha(W/'controlled2101/controlled_link.json'),selected_bundle_sha256=sha(W/'key_bundle/joint.json'),scope='only selected key kernel/ordinary adapter changed; final actual symbol and body separately audited')
 (build/'device_catalog/device_catalog.json').write_text(json.dumps(catalog,indent=2))
native=probe.native_validate(W/'normal/capture',build,gate/'host',llvm,atol=0.,rtol=0.);print(a.arm,'NATIVE_PASS',flush=True)
strict=probe.spike_validate(W/'normal/capture',build,spike,gate,atol=0.,rtol=0.);print(a.arm,'STRICT_PASS',flush=True)
audit=audit_elf((build/'model.elf').read_bytes());assert audit['status']=='pass';(gate/'result.json').write_text(json.dumps(dict(status='PASS',native=native,spike=strict,nofsm=audit,gate='all1000originalwords exact0/0'),indent=2))
