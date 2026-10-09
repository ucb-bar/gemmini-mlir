"""Independent original full TinyLlama output and target qualification."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent/'whole'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
N=Path(__file__).resolve().parent.parent/'tiny-broadcast-math-20261005/whole'
BUNDLE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
native=json.loads((W/'host/validation.json').read_text());link=json.loads((W/'controlled_link.json').read_text());normal=json.loads((W/'normal_lower_recipe.json').read_text())
assert native['status']=='pass' and native['original_compiled_bits_exact'] and native['allclose']
assert native['selected_llvm_sha256']==sha(W/'host_llvm/expanded.native.ll')
assert link['baseline_byte_exact'] and link['baseline_reproduced_sha256']==sha(N/'model.elf')
for path,expected in link['candidate_objects'].items():assert sha(path)==expected
for path,expected in normal['source_abi_pins'].items():assert sha(path)==expected
output=np.load(W/'host/output.npy');golden=np.load(BUNDLE/'golden.npy');raw_sha=hashlib.sha256(output.astype('<f4').tobytes()).hexdigest()
assert output.shape==(1,8,32000) and output.dtype==np.float32 and output.size==256000
assert raw_sha=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
assert sha(BUNDLE/'golden.npy')=='3a4d0d1105d4fad150b9ff1b80d4b7838d9fcdb3933a72b71000d6ecc0f5caee'
assert np.allclose(output,golden,atol=.03125,rtol=.02)
elf=W/'model.elf';assert sha(elf)==link['elf_sha256'];audit=audit_elf(elf.read_bytes());assert audit['status']=='pass';(W/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
command=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x400000000',str(elf)]
receipt={'schema':'tiny_constant_fma_norm_controlled1926_full_target_v1','status':'running','started_utc':now(),'argv':command,'elf_path':str(elf),'elf_sha256':sha(elf),'marker':link['inherited_marker'],'marker_contract':link['marker_contract'],'reference_path':str(W/'host/output.npy'),'reference_sha256':sha(W/'host/output.npy'),'torch_golden_path':str(BUNDLE/'golden.npy'),'torch_golden_sha256':sha(BUNDLE/'golden.npy'),'torch_atol':.03125,'torch_rtol':.02,'torch_allclose':True,'outputs':256000,'original_compiled_raw_sha256':raw_sha,'normal_lower_recipe_path':str(W/'normal_lower_recipe.json'),'normal_lower_recipe_sha256':sha(W/'normal_lower_recipe.json'),'controlled_link_path':str(W/'controlled_link.json'),'controlled_link_sha256':sha(W/'controlled_link.json'),'native_qualification_path':str(W/'host/validation.json'),'native_qualification_sha256':sha(W/'host/validation.json'),'nofsm_audit':audit,'before_verified_stock_job':1926,'before_verified_stock_cycles':461389700,'actual_candidate_hardware_cycles':None,'hardware_status':'not submitted','timing_scope':'Source-bound complete coupled packet4/constant-lifetime M2 capsule measured in actual GSIM; full Spike mcycle counts retired instructions. Whole candidate composes separately verified1926 norm; no whole forecast.','source_scope':link['scope'],'token_usage_available':False}
p=W/'spike_validation.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
with (W/'spike.log').open('w')as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);receipt['spike_pid']=process.pid;p.write_text(json.dumps(receipt,indent=2)+'\n')
 try:code=process.wait(timeout=3600)
 except subprocess.TimeoutExpired:
  process.terminate();process.wait(timeout=20);receipt.update(status='timeout',finished_utc=now());p.write_text(json.dumps(receipt,indent=2)+'\n');raise
console=(W/'spike.log').read_text().replace('\r','');metrics={line.split()[1]:line.split()[2]for line in console.splitlines()if line.startswith('METRIC ')}
passed=(code==0 and 'DONE'in console and metrics.get('memref_rank_mismatch')=='0'and metrics.get('build_hash')==link['inherited_marker']and console.count('OUT_SHA256 f32le 256000 1024000 '+raw_sha)==1)
receipt.update(status='pass'if passed else'fail',finished_utc=now(),exit_code=code,metrics=metrics,spike_console_path=str(W/'spike.log'),spike_console_sha256=sha(W/'spike.log'),functional_instructions_not_hardware_cycles=int(metrics.get('cycles','0')),spike_full_output_match=passed,spike_output_sha256=raw_sha if passed else None)
assert sha(elf)==receipt['elf_sha256'];p.write_text(json.dumps(receipt,indent=2)+'\n');assert passed
adapter={k:receipt[k]for k in ['reference_path','reference_sha256','torch_golden_path','torch_golden_sha256','torch_atol','torch_rtol','torch_allclose','spike_console_path','spike_console_sha256','spike_output_sha256','spike_full_output_match','elf_sha256','normal_lower_recipe_path','normal_lower_recipe_sha256','controlled_link_path','controlled_link_sha256']}
adapter.update(schema='reference_validation_v1',all_original_compiled_words_exact=True,original_reference_elements=256000,original_reference_shape=[1,8,32000],source_qualification_path=str(p),source_qualification_sha256=sha(p),inherited_marker=receipt['marker'],marker_contract=receipt['marker_contract'],scope=receipt['source_scope']);(W/'reference_validation.json').write_text(json.dumps(adapter,indent=2)+'\n')
print('FULL_ORIGINAL_STRICT_PASS',receipt['elf_sha256'],receipt['functional_instructions_not_hardware_cycles'],flush=True)
