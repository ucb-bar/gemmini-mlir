from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.golden_device_profile import parse_profile
W=Path(__file__).resolve().parent;P=W/'profile'
SOURCE=W.parent/'tiny-rectangular-whole-20261006/whole'
OLD=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/firesim1901_tiny1880_profile_verified.json')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
build=json.loads((P/'profile_build.json').read_text());overlay=json.loads((W/'overlay_receipt.json').read_text());native=json.loads((SOURCE/'host/validation.json').read_text());old=json.loads(OLD.read_text())
assert build['expected_device_calls']==155 and len(build['boundaries'])==5
for p,h in overlay['pins'].items():assert sha(p)==h
for p,h in build['objects'].items():assert sha(p)==h
assert build['model_object_sha256']==sha(SOURCE/'model.o')
assert native['status']=='pass' and native['original_compiled_bits_exact'] and native['allclose']
reference=Path(native['reference_path']);golden=Path(native['torch_golden_path'])
assert sha(reference)==native['reference_sha256'] and sha(golden)==native['torch_golden_sha256']
a=np.load(reference);g=np.load(golden);raw=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest()
assert a.shape==(1,8,32000) and a.size==256000 and raw=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
assert np.allclose(a,g,atol=.03125,rtol=.02)
elf=P/'model.elf';assert sha(elf)==build['elf_sha256'];audit=audit_elf(elf.read_bytes());assert audit['status']=='pass'
argv=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x400000000',str(elf)]
r={'schema':'current2004_profile_strict_v1','status':'running','started_utc':now(),'argv':argv,'elf_sha256':sha(elf),'source_stock_job':2004,'source_stock_cycles':422018733,'native_original_model_object_identity_gate':True,'native_qualification_path':str(SOURCE/'host/validation.json'),'native_qualification_sha256':sha(SOURCE/'host/validation.json'),'reference_path':str(reference),'reference_sha256':sha(reference),'torch_golden_path':str(golden),'torch_golden_sha256':sha(golden),'torch_atol':.03125,'torch_rtol':.02,'torch_allclose':True,'outputs':256000,'nofsm_audit':audit,'profile_build_path':str(P/'profile_build.json'),'profile_build_sha256':sha(P/'profile_build.json'),'token_usage_available':False,'scope':'Final-link instrumentation around same2004model.o and all155 originaldeviceboundaries. Native originalmodel/ABI/object equivalence retained; fresh strictfull originaloutput, exactordered counts and timerconservation. Spike counters areinstructions, not hardwarecycles. Device spans include CPUcommand/control/fence/wait plusaccelerator/DMA; outside includes profilerbookkeeping. Stockunmeasured.'}
(P/'spike_validation.json').write_text(json.dumps(r,indent=2)+'\n')
with(P/'spike.log').open('w')as log:
 proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['spike_pid']=proc.pid;(P/'spike_validation.json').write_text(json.dumps(r,indent=2)+'\n')
 try:code=proc.wait(timeout=3600)
 except subprocess.TimeoutExpired:
  proc.terminate();proc.wait(timeout=20);r.update(status='timeout',finished_utc=now());(P/'spike_validation.json').write_text(json.dumps(r,indent=2)+'\n');raise
text=(P/'spike.log').read_text().replace('\r','')
metrics={line.split()[1]:line.split()[2]for line in text.splitlines()if line.startswith('METRIC ')}
parsed=parse_profile(text,build)
old_order=[event[1]for event in old['boundary_profile']['events']]
assert [event[1]for event in parsed['events']]==old_order
passed=code==0 and 'DONE'in text and metrics.get('memref_rank_mismatch')=='0' and metrics.get('build_hash')=='37bdf9be0856' and text.count('OUT_SHA256 f32le 256000 1024000 '+raw)==1
r.update(status='pass'if passed else'fail',finished_utc=now(),exit_code=code,metrics=metrics,spike_console_path=str(P/'spike.log'),spike_console_sha256=sha(P/'spike.log'),functional_instructions_not_hardware_cycles=int(metrics.get('cycles','0')),spike_full_output_match=passed,spike_output_sha256=raw if passed else None,complete155_exact_original_callorder_counts_conservation=True,boundary_profile=parsed)
assert sha(elf)==r['elf_sha256'];(P/'spike_validation.json').write_text(json.dumps(r,indent=2)+'\n');assert passed
adapter={k:r[k]for k in ['elf_sha256','reference_path','reference_sha256','torch_golden_path','torch_golden_sha256','torch_atol','torch_rtol','torch_allclose','spike_console_path','spike_console_sha256','spike_output_sha256','spike_full_output_match','profile_build_path','profile_build_sha256']}
adapter.update(schema='firesim_reference_validation_v1',status='pass',build_hash='37bdf9be0856',full_output_match=True,original_receipt_path=str(P/'spike_validation.json'),original_receipt_sha256=sha(P/'spike_validation.json'),native_qualification_path=r['native_qualification_path'],native_qualification_sha256=r['native_qualification_sha256'],scope=r['scope'],inherited_marker_nonunique=True)
(P/'reference_validation.json').write_text(json.dumps(adapter,indent=2)+'\n')
print('CURRENT2004_PROFILE_STRICT_PASS',r['elf_sha256'],r['functional_instructions_not_hardware_cycles'],len(parsed['events']),flush=True)
