"""Read-only executed callsite and source-role census of frozen accepted2072."""
from pathlib import Path
import collections,hashlib,json,subprocess
base=Path(__file__).resolve().parents[2]
w=base/'out/current2072_attribution';frozen=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose')
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
hist={}
for line in (frozen/'histogram/spike.stderr').read_text().splitlines():
 t=line.split()
 if len(t)==2:
  try:hist[int(t[0],16)]=int(t[1])
  except ValueError:pass
records=[json.loads(x)for x in(w/'symbolized.jsonl').read_text().splitlines()]
locations={pc:r.get('Symbol',[]) for pc,r in zip(sorted(hist),records,strict=True)}
dis=subprocess.check_output([str(llvm/'llvm-objdump'),'-d','--no-show-raw-insn',str(w/'model.elf')],text=True)
(w/'executed_disassembly.txt').write_text(dis)
caller='';calls=[];entries={};classes=collections.Counter();by_function=collections.Counter()
for line in dis.splitlines():
 t=line.split()
 if len(t)==2 and t[1].startswith('<') and t[1].endswith('>:'):
  caller=t[1][1:-2];entries[caller]=hist.get(int(t[0],16),0);continue
 if not t or not t[0].endswith(':'):continue
 try:pc=int(t[0][:-1],16)
 except ValueError:continue
 count=hist.get(pc,0)
 if not count or len(t)<2:continue
 op=t[1];classes[(caller,op)]+=count;by_function[caller]+=count
 if op in ('jal','j') and len(t)>=4 and '<' in t[-1]:
  callee=t[-1].strip('<>')
  if '+0x' not in callee and callee!=caller:
   calls.append(dict(pc=hex(pc),caller=caller,callee=callee,count=count,operation=op,source=locations.get(pc,[])))
attr=json.loads((w/'attribution.json').read_text())
roles=collections.Counter();unknown=collections.Counter()
for r in attr['rows']:
 f=Path(r['file']).name;l=r['line'];fn=r['function'];count=r['instructions']
 if fn=='encode_operand':role='canonical_encoding'
 elif fn=='evaluate_products':
  if f=='provider.c' and 105<=l<=114:role='fused_integer_reconstruction'
  elif f=='provider.c' and l==437:role='row_scale_finish'
  elif f=='provider.c' and 426<=l<=436:role='product_callback_loop_and_admission'
  elif f=='provider.c' and l==0:role='products_unresolved_line_zero'
  else:role='dot_bounds_and_norms'
 elif '_products_' in fn:role='primitive_callback_instruction_body'
 elif fn=='group_provider':
  if f=='provider.c' and l==153:role='source_dot_replay_inline'
  elif f=='provider.c' and 385<=l<=413:role='source_gather_and_mask_validation'
  elif f=='bf16_quant_frontier.h':role='final_quantizer_observer'
  elif f=='provider.c' and 237<=l<=287:role='softmax_control_and_observation'
  elif f=='provider.c' and l==151:role='source_polynomial_replay_inline'
  elif 'polynomial' in f:role='polynomial_endpoint_propagation'
  elif f in ('f32_interval_endpoint.h','prepared_softmax_interval.h','prepared_softmax_word_domain.h','prepared_bf16_interval.h','private_softmax_interval.h'):role='interval_and_observer_helpers'
  else:role='provider_other_or_unresolved'
 else:continue
 roles[role]+=count
 if role=='provider_other_or_unresolved':unknown[(f,l)]+=count
selected=['group_provider','evaluate_products','encode_operand','source_quant_frontier','__truncsfbf2','rintf','nearbyintf','roundevenf']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
driver=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/frontier_i64_allocated_pair_v2/driver.c')
files=[driver,driver.with_suffix('.o'),frozen/'candidate/build.json',frozen/'histogram/terminal.json',frozen/'candidate/model.elf',frozen/'candidate/target_numeric/provider.c',frozen/'histogram/spike.stderr',frozen/'histogram/spike.stdout',w/'attribution.json',w/'symbolized.jsonl',w/'executed_disassembly.txt',Path(__file__)]
report=dict(scope='Accepted stock2072 exact ELF; complete first group only. Executed PC counts are instructions, never cycle percentages. Whole histogram includes two post-ROI original-consumer executions. Source roles below are exclusive caller-body instructions; shared callee costs not allocated implicitly.',elf_sha=sha(frozen/'candidate/model.elf'),entry_counts={x:entries.get(x) for x in selected},exclusive_bodies={x:by_function[x]for x in selected},callsites=[c for c in calls if c['callee'] in selected],roles=dict(roles),unresolved_provider_lines=[dict(file=f,line=l,instructions=c)for (f,l),c in unknown.most_common(30)],opcode_classes={x:{op:c for (fn,op),c in classes.items()if fn==x}for x in selected},pins={str(p):sha(p)for p in files})
assert sum(c['count'] for c in calls if c['callee']=='__truncsfbf2' and c['caller']=='source_quant_frontier')==entries['__truncsfbf2']
assert not any(c for c in calls if c['callee']=='__truncsfbf2' and c['caller']!='source_quant_frontier')
report['roi_instructions']=1734429991
report['post_roi_exclusion']={'__truncsfbf2':by_function['__truncsfbf2'],'source_quant_frontier':by_function['source_quant_frontier'],'reason':'Actual frozen driver calls original and candidate compiled consumers only after elapsed=tick()-start. All helper entry executions are accounted by those consumer callsites.'}
report['shared_rintf_scope']={'post_roi_roundeven_calls':entries['roundevenf'],'provider_nearbyint_calls':entries['nearbyintf'],'exclusive_body_instructions':by_function['rintf'],'per_call_body_split':'UNKNOWN: input-dependent shared body; do not distribute by call-count ratios.'}
report['unassigned_roi_instructions']=report['roi_instructions']-sum(roles.values())
report['scope_warning']='Source roles do not include shared callees; line-zero provider work remains unresolved. Callback instruction bodies include CPU issue code, not accelerator busy cycles, wait time or physical traffic.'
(base/'docs/perf_records/accepted2072_executed_source_roles.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k]for k in ('entry_counts','exclusive_bodies','roles')},indent=2))
