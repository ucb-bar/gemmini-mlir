from pathlib import Path
import json,hashlib,collections
root=Path(__file__).resolve().parents[2]
p=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/preparation_census.json')
d=json.loads(p.read_text());requests=d['source_requests'];groups=collections.defaultdict(list)
for x in requests:
 if x['argument_index']==0:continue
 family='K' if x['argument_index'] in (1,6) else 'V'
 key=(family,x['root_operation_ordinal'],tuple(x['offsets']),tuple(x['sizes']),tuple(x['strides']))
 groups[key].append(x)
assert len(groups)==96 and all(len(v)==4 for v in groups.values())
rows=[]
for key,items in groups.items():
 family=key[0];x=items[0];heads=x['sizes'][1];words=x['logical_bf16_words'];rowcount=heads*(x['sizes'][2] if family=='K' else x['sizes'][3]);length=x['sizes'][3] if family=='K' else x['sizes'][2]
 rows.append({'family':family,'source_root_ordinal_trace_only':key[1],'offsets':key[2],'sizes':key[3],'strides':key[4],'consumer_calls':[v['call_index'] for v in items],'heads':heads,'rows':rowcount,'vector_length':length,'words':words,'encode_calls_before':heads*4,'encode_calls_unique':heads,'physical_format':{'planes':'[head,plane,k,column]','radix':128,'digits':3,'row_step':'source-derived exact power of two','source_and_reconstructed':'[head,column,k]','equality':'numeric source/reconstruction equality; does not prove source FMA exactness','rounding':'stable source RNE'},'prepared_bytes_conservative':19*words+9*rowcount})
unique=sum(r['words'] for r in rows);peak=sum(r['prepared_bytes_conservative'] for r in rows if 0 in r['consumer_calls'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
provider=root/'out/polynomial_outline/candidate_target/provider.c'
pins={str(p):sha(p),str(provider):sha(provider)}
for n in ['encoded_row_equality.h','bf16_radix.h']:
 q=provider.parent/n
 if q.exists():pins[str(q)]=sha(q)
result={'scope':'Read-only source/view/format census, not installed physical reuse or cycle prediction','source_sha256':d['source_sha256'],'original_full_source_sha256':d['original_full_source_sha256'],'per_group_encode_calls':{'Q':24,'K':24,'probability_parts':72,'V_parts':72,'total':192},'per_group_encoded_words':{'Q':393216,'K':786432,'probability_parts':3145728,'V_parts':786432,'total':5111808},'cross_group_KV':{'equivalent_view_groups':len(rows),'encode_calls_before':sum(r['encode_calls_before'] for r in rows),'encode_calls_unique':sum(r['encode_calls_unique'] for r in rows),'potential_calls_removed':sum(r['encode_calls_before']-r['encode_calls_unique'] for r in rows),'input_words_before':4*unique,'unique_words':unique,'potential_repeated_words_removed':3*unique,'potential_repeated_f32_bytes_removed':12*unique,'potential_repeated_original_bf16_bytes':6*unique,'conservative_peak_prepared_payload_bytes':peak,'payload_terms':'19 bytes/element = source f32 + reconstructed f32/f64 + three i8 planes;9 bytes/row = f64 step + equality flag. Excludes alignment, handles and optional norm metadata; scope one four-consumer owner group.','physical_lifetime_installed':False},'local_Q':{'potential_calls_removed_per_group':12,'potential_words_removed_per_group':196608,'decision':'Do not repeat prior negative local-query preparation experiment.'},'probability_parts':'72 distinct source-produced probability subviews per group; no cross-query reuse proof','requirements':['typed immutable source owner and static view identity, not pointer or value equality','explicit head-first transposed B physical format and exact encoding operation identity','prepared source/reconstructed/planes/steps/flags owners dominate all four consumers and survive their synchronous callbacks','fresh owner epoch, alias exclusion, no writes/invalidation until final consumer','existing scratch-bound equality proof cannot be reused after w->b/w->br overwrite; prepared-handle proof and consumer binding required','full allocation/gather/format/lifetime costs must be measured'],'rows':rows,'pins':pins,'cycles_saved':'unknown','fraction_of_355M_encoder_instructions_saved':'unknown; no per-family current PC attribution','token_usage_available':False}
out=root/'docs/perf_records/current_encoder_source_reuse_census.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['cross_group_KV'],indent=2))
