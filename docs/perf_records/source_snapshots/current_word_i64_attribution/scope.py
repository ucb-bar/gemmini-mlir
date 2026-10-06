from pathlib import Path
import json,re,subprocess,bisect,collections,hashlib
w=Path(__file__).parent;h=w.parent/'current_word_i64_pc_histogram';old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');nm='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm'
def symbols(p):
 out=[]
 for l in subprocess.check_output([nm,'-S','-n',str(p)],text=True).splitlines():
  a=l.split()
  if len(a)==4 and a[2] in ('t','T','w','W'):out.append((int(a[0],16),int(a[0],16)+int(a[1],16),a[3]))
 return sorted(out)
syms=symbols(old/'candidate/model.elf');starts=[s[0]for s in syms];owned={s[2]for s in symbols(old/'numeric_frozen/provider.o')}-{'group_provider_statistics'}
counts={}
for line in (h/'spike.log').read_text().splitlines():
 m=re.fullmatch(r'([0-9a-f]+) (\d+)',line)
 if m:counts[int(m[1],16)]=int(m[2])
frames={int(x['Address'],16):x['Symbol']for x in map(json.loads,(w/'symbolized.jsonl').read_text().splitlines())};cats=collections.Counter();unknown=collections.Counter();mapped=collections.Counter()
for pc,n in counts.items():
 i=bisect.bisect_right(starts,pc)-1;fn=syms[i][2]if i>=0 and pc<syms[i][1]else'<unmapped>'
 if fn in owned:
  fs=frames[pc];names=[f['FunctionName']for f in fs]
  if any(a in names for a in ('source_dot','exact_source_partials','exact_source_denominator')):cat='source_order_replay_inline'
  elif 'encode_operand'in names:cat='radix_encoding_inline'
  elif any(a.startswith('merlin_radix_integer_')for a in names):cat='integer_reconstruction_inline'
  elif 'dot_bounds'in names or fn=='merlin_fma_product_row_prepare_l1':cat='dot_domain_bounds_and_norms_inline'
  elif any('polynomial'in a for a in names):cat='polynomial_interval_inline'
  elif any(a.startswith('merlin_frontier_')for a in names):cat='quant_consumer_certificate_inline'
  elif 'endpoint_intervals'in names:cat='endpoint_interval_finalization_inline'
  elif 'soft_details'in names or 'soft_details_checked'in names:cat='softmax_max_denominator_alpha_inline'
  elif 'gather_head'in names:cat='input_gather_inline'
  else:cat='provider_control_copy_setup_and_unclassified_inline'
  cats[cat]+=n;mapped[fn]+=n
 elif re.fullmatch(r'(qk|pv192|pv128)_products_[0-4]',fn)or fn=='products':cats['device_callback_and_command_instructions']+=n
 elif fn=='attention_frontier_writer':cats['ranked_descriptor_bridge']+=n
 elif fn=='main'and 0x800127e8<=pc<0x8001286a:cats['roi_main_allocation_alignment_guard_setup']+=n
 elif fn=='malloc':cats['malloc_all_calls_upper_bound_not_assigned']+=n
 else:unknown[fn]+=n
roi=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',(h/'spike.log').read_text()).group(1))
# malloc can also be reached by post-ROI checks: retain its full census separately.
malloc=cats.pop('malloc_all_calls_upper_bound_not_assigned',0);known=sum(cats.values());assert known<=roi
r={'schema':'current_provider_pc_roi_census_v1','scope':'Exact71193a complete original12head group; retired instructions, not cycles. No altered execution image.','roi_instructions':roi,'program_histogram_instructions':sum(counts.values()),'known_roi_categories':dict(cats.most_common()),'known_roi_sum':known,'roi_unresolved_shared_callees_and_boundary':roi-known,'post_roi_plus_pre_roi_total':sum(counts.values())-roi,'shared_or_outside_function_counts':dict(unknown.most_common()),'malloc_full_program_upper_bound':malloc,'provider_physical_function_counts':dict(mapped.most_common()),'limitations':['Source categories count only instructions physically in admitted provider and exact inline chains; external libm/memcpy cost is not charged to arbitrary caller.','Device callbacks count retired command/loop instructions, not accelerator array work or elapsed latency. Readback is executed by commands; no separate host-cycle attribution.','Main start-counter is included and end-counter excluded; the residual retains counter-boundary convention uncertainty.','Original compiled consumer and validation outside ROI are not attributed to provider.','Inline line mapping may combine scheduled instructions from adjacent source operations.'],'token_usage_available':False}
(w/'roi_attribution.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ('roi_instructions','known_roi_categories','known_roi_sum','roi_unresolved_shared_callees_and_boundary')},indent=2))
