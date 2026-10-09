"""Use original whole memory extent; unchanged ELF after failed payload load."""
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from finite_scale_whole_target import TARGET,ORIGINAL,RAW,GCC,sha,save

first=TARGET/'target_validation.json'
record=json.loads(first.read_text());assert record['exit_code']==255 and not record['metrics']
elf=Path(record['elf_path']);assert sha(elf)==record['elf_sha256']
command=[str(GCC.with_name('spike')),'-g','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x400000000',str(elf)]
log=TARGET/'target/spike_m16g.log';started=datetime.now(timezone.utc).isoformat()
with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
text=log.read_text();metrics={}
for line in text.splitlines():
 if line.startswith('METRIC '):
  _,name,value=line.split(maxsplit=2);assert name not in metrics;metrics[name]=value
passed=result.returncode==0 and text.count('OUT_SHA256 f32le 256000 1024000 '+RAW)==1 and sum(line.strip()=='DONE' for line in text.splitlines())==1 and metrics.get('memref_rank_mismatch')=='0'
record.update(status='pass' if passed else 'fail',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),argv=command,exit_code=result.returncode,spike_console_path=str(log),spike_console_sha256=sha(log),spike_output_sha256=RAW if passed else None,spike_full_output_match=passed,metrics=metrics,functional_instructions_not_hardware_cycles=int(metrics.get('cycles','0')),original_failed_load_receipt=str(first),original_failed_load_receipt_sha256=sha(first),load_attempt_scope='First run omitted original16GiB mapped extent and failed during payload load, before model execution. Same immutable ELF now uses exact original whole Spike options.')
save(TARGET/'target_validation_m16g.json',record);assert passed and sha(elf)==record['elf_sha256']
print('FINITE_SCALE_ORIGINAL_WHOLE_STRICT_PASS',record['elf_sha256'],record['functional_instructions_not_hardware_cycles'],flush=True)
