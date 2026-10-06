"""Independently compiled source integer/scale observations for all groups."""
from pathlib import Path
import hashlib,json,numpy as np
from merlin.llvmlower.abi import HostModel
w=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005');out=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/numeric_minmax_native_screen')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_receipt=json.loads((w/'quant_frontier/validation.json').read_text())
assert sha(w/'quant_frontier/frontier.so')==source_receipt['library_sha256']
run=HostModel.load(str(w/'quant_frontier/frontier.so'),name='source_quant_frontier')
current=json.loads((out/'runtime_group_calls.json').read_text());exact=json.loads((w/'exact_native_control/runtime_group_calls.json').read_text())
assert len(current)==len(exact)==48
same_inputs=[a['input_raw_sha256']==b['input_raw_sha256'] for a,b in zip(current,exact)]
assert all(same_inputs)
observations=[];endpoints=[]
for start in range(0,48,4):
    results=[]
    for label in [w/'exact_native_control',out]:
        values=np.concatenate([np.load(label/f'group_{i}_output.npy') for i in range(start,start+4)],axis=2)
        integer=np.empty((1,1024,768),np.int8);scale=np.empty((1,1024),np.uint16)
        run([(value.ctypes.data,value.shape) for value in [values,integer,scale]])
        results.append((integer,scale))
    row=dict(first_group=start,integer_words=results[0][0].size,integer_bit_mismatches=int(np.count_nonzero(results[0][0]!=results[1][0])),bf16_scale_words=results[0][1].size,bf16_scale_bit_mismatches=int(np.count_nonzero(results[0][1]!=results[1][1])),original_i8_raw_sha256=hashlib.sha256(results[0][0].tobytes()).hexdigest(),candidate_i8_raw_sha256=hashlib.sha256(results[1][0].tobytes()).hexdigest(),original_bf16_scale_raw_sha256=hashlib.sha256(results[0][1].tobytes()).hexdigest(),candidate_bf16_scale_raw_sha256=hashlib.sha256(results[1][1].tobytes()).hexdigest())
    observations.append(row);assert row['integer_bit_mismatches']==row['bf16_scale_bit_mismatches']==0
for i in range(48):
    a=np.load(w/'exact_native_control'/f'group_{i}_output.npy');b=np.load(out/f'group_{i}_output.npy')
    endpoints.append(dict(index=i,changed_bf16_words=int(np.count_nonzero(a!=b)),words=int(a.size),same_original_live_inputs=same_inputs[i]))
r=dict(schema='complete_source_quant_observation_pair_v1',whole_original_gate='all1600 source words bitexact; unchanged atol.03125/rtol.02',actual_source_calls=48,original_input_equal_groups=sum(same_inputs),observed_consumer_frontiers=12,compiled_source_integer_words=sum(row['integer_words']for row in observations),compiled_source_scale_words=sum(row['bf16_scale_words']for row in observations),integer_mismatches=0,bf16_scale_mismatches=0,changed_bf16_endpoint_words=sum(row['changed_bf16_words']for row in endpoints),total_bf16_endpoint_words=sum(row['words']for row in endpoints),observations=observations,endpoints=endpoints,source_compiled_quant_library_sha256=source_receipt['library_sha256'],source_quant_frontier_sha256=source_receipt['frontier_source_sha256'],native_source_pair_receipts=dict(original_exact=sha(w/'exact_native_control/native_validation.json'),candidate=sha(out/'native_validation.json')),scope='All current consumer outputs independently checked through actual original compiled source54op quant DAG; typed complete59op source closure matches every12consumer instance. No target/cycle admission.',token_usage_available=False)
(out/'complete_observation_pair.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({key:value for key,value in r.items()if key not in ['observations','endpoints']}),flush=True)
