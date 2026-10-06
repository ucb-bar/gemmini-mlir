from pathlib import Path
import json,subprocess,bisect,collections
w=Path(__file__).resolve().parent;prefix='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-';elf=Path('/scratch/agustin/tmp/gemmini-probability-point-20261006/out/probability_points/candidate/model.elf');provider=elf.parent.parent/'candidate_numeric/provider.o'
def symbols(path):
 out=[]
 for line in subprocess.check_output([prefix+'nm','-S','-n',str(path)],text=True).splitlines():
  a=line.split()
  if len(a)==4 and a[2] in ['t','T','w','W']:out.append((int(a[0],16),int(a[0],16)+int(a[1],16),a[3]))
 return sorted(out)
syms=symbols(elf);starts=[x[0] for x in syms];owned={x[2] for x in symbols(provider)}-{'group_provider_statistics'};counts={}
for line in (w/'spike.log').read_text().splitlines():
 a=line.split()
 if len(a)!=2:continue
 try:pc,n=int(a[0],16),int(a[1])
 except ValueError:continue
 counts[pc]=n
frames={int(x['Address'],16):x['Symbol'] for x in map(json.loads,(w/'symbolized.jsonl').read_text().splitlines())}
ops={};disassembly=subprocess.check_output([prefix+'objdump','-d',str(elf)],text=True);(w/'disassembly.txt').write_text(disassembly)
for line in disassembly.splitlines():
 a=line.split()
 if len(a)<3 or not a[0].endswith(':'):continue
 try:pc=int(a[0][:-1],16)
 except ValueError:continue
 ops[pc]=(a[2],' '.join(a[3:]))
functions=collections.Counter();by=collections.defaultdict(collections.Counter);categories=collections.Counter();category_ops=collections.defaultdict(collections.Counter);call_sites=[]
for pc,n in counts.items():
 i=bisect.bisect_right(starts,pc)-1;fn=syms[i][2] if i>=0 and pc<syms[i][1] else '<unmapped>';op,args=ops.get(pc,('<unknown>',''));functions[fn]+=n;by[fn][op]+=n
 if fn not in owned:continue
 names=[x['FunctionName'] for x in frames.get(pc,[])]
 if 'source_dot' in names:cat='source_ordered_dot_replay'
 elif any('polynomial' in a for a in names):cat='polynomial_enclosure_or_exact_polynomial'
 elif 'encode_operand' in names:cat='radix_encoding'
 elif any(a.startswith('merlin_radix_integer_') for a in names):cat='integer_reconstruction'
 elif 'dot_bounds' in names or fn=='merlin_fma_product_row_prepare_l1':cat='bounds_and_norms'
 elif 'exact_source_denominator' in names:cat='exact_denominator_replay_non_dot'
 elif 'exact_source_partials' in names:cat='exact_PV_replay_non_dot'
 elif any(a.startswith('merlin_frontier_') for a in names):cat='consumer_certificate'
 elif 'endpoint_intervals' in names:cat='endpoint_finalization'
 elif 'soft_details' in names or 'soft_details_checked' in names:cat='softmax_max_denominator_alpha'
 elif 'gather_head' in names:cat='input_gather'
 else:cat='provider_control_copy_setup_unclassified'
 categories[cat]+=n;category_ops[cat][op]+=n
 if op in ['jal','jalr']:call_sites.append({'pc':hex(pc),'function':fn,'instruction':op+' '+args,'count':n,'inline_chain':names})
def classes(counter):
 out=collections.Counter()
 for op,n in counter.items():
  if op.startswith(('fmadd','fmsub','fnmadd','fnmsub')):cat='fma_'+op[-1]
  elif op.startswith('fdiv'):cat='division_'+op[-1]
  elif op.startswith('fcvt'):cat='conversion_'+op
  elif op in ['flw','fld','fsw','fsd','lw','lwu','ld','lb','lbu','lh','lhu','sw','sd','sb','sh']:cat='load_store_'+op
  elif op.startswith(('fadd','fsub','fmul','fsqrt','fmin','fmax')):cat='arithmetic_'+op
  elif op in ['div','divu','divw','divuw','rem','remu','remw','remuw']:cat='integer_division'
  else:cat='other'
  out[cat]+=n
 return dict(out.most_common())
roi=None
for line in (w/'spike.log').read_text().splitlines():
 if line.startswith('WORKSPACE_GROUP_INSTRUCTIONS '):roi=int(line.split()[1])
r={'scope':'Exact stock2024 ELF, whole-program retired-PC histogram. Provider function bodies execute inside original ROI; external shared callees remain unassigned. Debug-only linked image is never executed and all102allocated sections match.','roi_instructions':roi,'program_instructions':sum(counts.values()),'provider_known_instructions':sum(categories.values()),'roi_other_callbacks_bridge_shared_libraries_and_boundary':roi-sum(categories.values()),'largest_physical_provider_bodies':[{'name':fn,'instructions':n,'entry_count':counts.get(next(a for a,b,name in syms if name==fn),0),'instruction_classes':classes(by[fn]),'opcodes':dict(by[fn].most_common())} for fn,n in functions.most_common() if fn in owned][:3],'provider_source_categories':dict(categories.most_common()),'category_instruction_classes':{k:classes(v)for k,v in category_ops.items()},'all_function_counts':dict(functions.most_common()),'provider_call_sites':call_sites,'limitations':['Retired instructions are not CPU cycles or accelerator array work.','Function entry counts count executed entry PCs, including any tail entry; inline helper call counts are not inferred from global callee costs.','Source categories are debug inline attribution, not separately timed regions.','Shared memcpy/libm/allocator calls may also serve post-ROI original-consumer validation. Their full program counts are not assigned to ROI.'],'token_usage_available':False}
(w/'current_roi_census.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['roi_instructions','provider_known_instructions','provider_source_categories']},indent=2));print([(x['name'],x['instructions'],x['entry_count'])for x in r['largest_physical_provider_bodies']])
