"""Reclose directed narrowing, including the retained carrier-version refusal."""
from pathlib import Path
import argparse,hashlib,json
from mlir_oot.no_fsm_audit import audit_elf

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.repository.resolve()
base=root/'out/artifacts/probes/smol-monotone-polynomial-20261006'
pins={}
def pin(path,expected=None):
 path=Path(path).absolute();value=hashlib.sha256(path.read_bytes()).hexdigest()
 if expected is not None:assert value==expected,str(path)
 pins[str(path)]=value;return value
def read(path):pin(path);return json.loads(Path(path).read_text())
control=read(base/'row_invariants_target/timed.spike.json')
native=read(base/'directed_cast/result.json')
build=read(base/'directed_cast_native_oracle_target/build_receipt.json')
strict=read(base/'directed_cast_native_oracle_target/timed.spike.json')
diagnostic=read(base/'directed_cast_target/timed.spike.json')
assert strict['rc']==0 and strict['pass'] is True and 'ENDPOINT PASS' in strict['markers']
assert diagnostic['rc']!=0 and diagnostic['pass'] is False
for path,expected in native['pins'].items():pin(path,expected)
pin(native['tap_path'],native['tap_sha256'])
for key in ('objects','pins'):
 for path,expected in build[key].items():pin(path,expected)
for key in ('native_oracle','native_gate','control_build','control_strict'):
 pin(build[key+'_path'],build[key+'_sha256'])
for arm,receipt in [('directed_cast_target',diagnostic),('directed_cast_native_oracle_target',strict)]:
 pin(base/arm/'timed.elf',receipt['elf_sha256']);pin(base/arm/'timed.spike.log',receipt['log_sha256'])
audit=audit_elf((base/'directed_cast_native_oracle_target/timed.elf').read_bytes())
assert audit==build['nofsm_audit'] and audit['status']=='pass'
assert build['unchanged_original_input_and_source_gold_object_sha256']==pin(base/'row_invariants_target/data.o')
assert native['original_i8_words_exact'] and native['original_scale_words_exact']
assert (native['i8_words'],native['scale_words'],native['changed_endpoint_words'])==(196608,256,126)
assert native['counts']['replay_total_fmas']==4460864
assert 'DIRECTED_CAST PASS 65985' in strict['markers']
assert 'DIRECTED_MATH PASS 5316' in strict['markers']
assert 'ENDPOINT_PLANES 480 86507520 0' in strict['markers']
counter=lambda r:next(int(s.split()[1]) for s in r['markers'] if s.startswith('ENDPOINT_CYCLES '))
before,after=counter(control),counter(strict)
for script in ('prepare_directed_cast.py','qualify_directed_cast_carriers.py'):
 pin(base/script)
pin(base/'directed_cast_attempt1_diagnostic.json')
receipt=dict(schema='directed_source_bound_narrowing_qualification_v1',
 scope='Complete original12head/256query source attention group; functional Spike mcycle instruction proxy, not stock FireSim cycles or a whole-model forecast.',
 source_gate=dict(original_i8_words=196608,original_scales=256,bit_mismatches=0,original_input_and_source_gold_object_unchanged=True,accuracy_gate_changed=False),
 before_instruction_proxy=before,after_instruction_proxy=after,reduction_percent=100*(before-after)/before,
 source_replay_fmas_before=4461440,source_replay_fmas_after=4460864,
 native=native,build=build,strict=strict,audit=audit,
 retained_carrier_version_refusal=dict(strict=diagnostic,
  cause='Previous implementation carrier reference differs from the newly qualified implementation at one additional internal carrier. Complete closed consumer observations define the policy. This reference refusal is not an original source quantization or accuracy failure.',
  resolution='New pinned independent native carrier object; original source gold/input data object byte-identical and target original quantization oracle unchanged.'),
 mathematical_qualification=dict(exact_rational_conversion_cases=13197,ambient_target_rounding_modes=5,actual_target_conversion_checks=65985,directed_f64_checks=5316,source_rounding_environment_unchanged=True),
 ownership=dict(mathematical_bounds_and_optional_conversion_hooks='Merlin',fixed_CPU_instruction_definitions_and_Gemmini_products='OOT'),
 production_whole_target_qualified=False,stock_hardware_measured=False,token_usage_available=False,pins=pins)
args.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(pins_rehashed=len(pins),before=before,after=after,reduction_percent=receipt['reduction_percent'],original_gate='pass')))
