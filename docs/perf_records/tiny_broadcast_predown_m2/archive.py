"""Freeze completed real-source M2 broadcast cost; retain oldM8 as unknown."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
W=Path(__file__).resolve().parent;O=W.parents[3]
R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006')
C=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
x=json.loads((W/'capsule_result.json').read_text());g=json.loads((W/'gsim_receipt.json').read_text())
assert x['DONE'] and g['finish']['done'] and g['returncode']==0
assert x['all5_frm_and_sticky_flags_match'] and x['saved_percent']>0
assert sha(W/'control.o')==sha(C/'out/artifacts/probes/reciprocal-rne-observer-20261006/cost_m2/control.o')
old=json.loads((R/'capsule/gsim_receipt.json').read_text());assert old['returncode']!=0 and old['finish']is None
paths=[W/p for p in ['build_capsule.py','run_cost.py','archive.py','capsule_qualification.json','capsule_result.json','gsim_receipt.json','gsim.stdout','strict_spike.log','timing_spike.log','strict_audit.json','timing_audit.json','strict_main.c','timing_main.c','control.native.ll','control.target.ll','broadcast.native.ll','broadcast.target.ll','control.o','broadcast.o','timing_build/layer.elf','strict_build/layer.elf']]
paths.extend(R/'capture'/(n+'.npy')for n in ['a','scale_a','b','scale_b','expected'])
paths.extend([R/'capture/receipt.json',R/'source_receipt.json',R/'source_capsule.mlir',R/'capsule/data.o',R/'capsule/gsim_receipt.json',R/'capsule/gsim.stdout',C/'src/merlin/llvmlower/scalar_pointwise_packet.py'])
receipt={
 'schema':'compiler_optimization_journey_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
 'hypothesis':'Typed broadcast-axis lane interleaving can share immutable per-channel scale loads and lower register/code overhead while preserving each source result.',
 'ownership':'Existing generic typed tensor schedule in Merlin; actual target source capsule/ABI/audit/cost evidence OOT. No ISA or model selector added.',
 'emitted_change':'Use existing explicit packet_scalar_pointwise_broadcast_2 in normal source lowering instead of ordinaryFMA/division packet2. Proved indexingmaps share scale operands across two independent rows; scalar operation/cast/rounding order, tensor values and output ownership preserved.',
 'before_after_metric':{'scope':'First2originalrows,all5632channels; complete source preDown body including loads/dequantization/FMApolynomial/fdiv/three rounded products/i8RNEclamp/stores/fresh temporary/final copy. Warm ABBA GSIM; flags/guards/input validation outside timer. Not full8-row source, whole model or stock FireSim.','outputs':11264,'paired_cycles':x['paired_complete_gsim_cycles'],'control_mean':x['before_mean'],'candidate_mean':x['after_mean'],'saved_percent':x['saved_percent']},
 'gate':{'native_original_words':11264,'strict_five_frm_compared_words':56320,'sticky_flags_match':True,'dirty_guards':128,'immutable_input_bytes_protected':405504,'used_input_bytes':135168,'original_control_object_byteidentical':True,'zero_FSM':True,'engine_DONE':True},
 'attempt':{'status':'COMPLETE','single_engine':True,'sigstop_held_after_launch_for_parent_conditional_sequencing':True,'resumed_after_parent_release':True,'precise_pause_wall_duration':'unavailable','scope':'Engine walltime includes the hold; ROI is RTL mcycle. No timing counters rely on host walltime, no rebuild/duplicate engine and no completed originalM2 pair rerun.'},
 'oldM8':{'status':'INCOMPLETE_TIMEOUT_UNKNOWN','scope':'Oldfull8-row real-source pair timed out1800s after only control5828044cycles. Candidate/DONE missing; no positive/negative comparison asserted and no identicalM8 rerun.','receipt_path':str(R/'capsule/gsim_receipt.json'),'receipt_sha256':sha(R/'capsule/gsim_receipt.json')},
 'result':'Positive9.76097%local completeM2screen. Whole current1967-source/norm composition not yet qualified; no whole speed or promotion claim.',
 'actual_whole_hardware_cycles':None,'token_usage_available':False,'pins':{str(p):sha(p)for p in paths}}
(W/'journey.json').write_text(json.dumps(receipt,indent=2)+'\n')
A=O/'docs/perf_records/tiny_broadcast_predown_m2';A.mkdir(exist_ok=False)
for p in paths[:16]:
 if p.parent==W:(A/p.name).write_bytes(p.read_bytes())
(O/'docs/perf_records/tiny_broadcast_predown_m2_journey.json').write_bytes((W/'journey.json').read_bytes())
print('BROADCAST_M2_ARCHIVED',x['saved_percent'])
