"""Close original-source paired RTL evidence; never infer whole-model gain."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone

W=Path(__file__).resolve().parent
C=W/'capsule'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
q=json.loads((C/'qualification.json').read_text())
r=json.loads((C/'gsim_receipt.json').read_text())
assert r['returncode']==0 and r['finish'] and r['finish']['done'],r
log=(C/'gsim.stdout').read_text()
assert 'ORIGINAL_GATE_PACKET PASS 45056 128' in log
rows=[]
for line in log.splitlines():
 parts=line.split()
 if parts and parts[0]=='GATE_PACKET_CYCLES':
  assert len(parts)==4
  rows.append([int(value) for value in parts[1:]])
assert len(rows)==4 and [row[:2] for row in rows]==[[0,0],[1,1],[2,1],[3,0]],rows
assert all(row[2]>0 for row in rows)
control=sum(row[2] for row in rows if row[1]==0)/2
candidate=sum(row[2] for row in rows if row[1]==1)/2
for arm in q['cases']:
 name=arm['name']
 assert sha(C/(name+'.target.ll'))==arm['target_sha256']
 assert sha(C/(name+'.native.ll'))==arm['native_sha256']
 assert sha(C/(name+'.o'))==arm['object_sha256']
assert sha(q['elf_path'])==q['elf_sha256']
control_identity=json.loads((W/'normal_control/identity.json').read_text())
assert control_identity['raw_target_native_byteidentical']
paths=[W/'prepacket_receipt.json',W/'prepacket.mlir',W/'source_receipt.json',W/'source_capsule.mlir',
 W/'capture/receipt.json',W/'normal_control/identity.json',C/'qualification.json',C/'main.c',
 C/'source.mlir',C/'gsim_receipt.json',C/'gsim.stdout',C/'spike.stdout',C/'spike.stderr',
 C/'nofsm_audit.json',Path(q['elf_path'])]
paths.extend(W/'capture'/(name+'.npy') for name in ['a','scale_a','b','scale_b','expected'])
doc={'schema':'tiny_original_gate_broadcast_packet_complete_capsule_v1',
 'recorded_utc':datetime.now(timezone.utc).isoformat(),'source_output_words':45056,
 'original_full_model_readonly_capture_output_words':256000,'original_full_model_bits_exact':True,
 'original_capsule_native_and_strict_outputs_exact':True,'dirty_guards':128,
 'input_bytes_preserved':q['immutable_input_bytes'],'zero_FSM':True,
 'hardware_regime':'Pinned elaborated GSIM RTL, separate memory regime from stock FireSim',
 'sequence':rows,'control_mean_cycles':control,'broadcast_mean_cycles':candidate,
 'saved_percent':100*(control-candidate)/control,
 'metric_scope':q['complete_ROI'],'normal_original_feature_control_LLVM_byteexact':True,
 'whole_candidate_accuracy_and_cycles':None,
 'policy':'Optional generic two-lane broadcast map sharing; no approximation or source-name production selector',
 'decision':'positive isolated capsule; full normal/frozen qualification required' if candidate<control else 'reject capsule regression; no whole admission',
 'pins':{str(path):{'path':str(path),'sha256':sha(path)} for path in paths},
 'token_usage_available':False}
(W/'capsule_result.json').write_text(json.dumps(doc,indent=2)+'\n')
print({key:doc[key] for key in ['sequence','control_mean_cycles','broadcast_mean_cycles','saved_percent','decision']})
