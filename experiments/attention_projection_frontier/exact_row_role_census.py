"""Exclusive executed-PC roles of the immutable row-proof first group."""
from pathlib import Path
import collections, hashlib, json, subprocess
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'out/exact_row_attribution';G=ROOT/'out/exact_row_group'
OUT=ROOT/'out/exact_row_role_census';OUT.mkdir(exist_ok=False)
hist={}
for line in (G/'strict/stderr').read_text().splitlines():
 t=line.split()
 if len(t)==2:
  try:hist[int(t[0],16)]=int(t[1])
  except ValueError:pass
records=[json.loads(x)for x in (A/'symbolized.jsonl').read_text().splitlines()]
locations={p:r.get('Symbol',[])for p,r in zip(sorted(hist),records,strict=True)}
dis=subprocess.check_output(['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-objdump','-d','--no-show-raw-insn',str(A/'model.elf')],text=True)
(OUT/'disassembly.txt').write_text(dis)
check_pcs=[];fn='';classes=collections.Counter();calls=[];entries={}
for line in dis.splitlines():
 t=line.split()
 if len(t)==2 and t[1].startswith('<')and t[1].endswith('>:'):
  fn=t[1][1:-2];entries[fn]=hist.get(int(t[0],16),0);continue
 if not t or not t[0].endswith(':'):continue
 try:pc=int(t[0][:-1],16)
 except ValueError:continue
 count=hist.get(pc,0)
 if not count:continue
 loc=locations.get(pc,[])
 if any(Path(x.get('FileName','')).name=='source_rms_point_products.h'and x.get('Line')==57 for x in loc):
  check_pcs.append(dict(pc=hex(pc),count=count,assembly=' '.join(t[1:]),function=fn));classes[t[1]]+=count
 if t[1]in('jal','j')and len(t)>=4 and '<'in t[-1]:calls.append(dict(caller=fn,callee=t[-1].strip('<>'),count=count))
rows=json.load(open(A/'attribution.json'))['rows'];roles=collections.Counter();rms=collections.Counter()
for r in rows:
 f=Path(r['file']).name;l=r['line'];fn=r['function'];c=r['instructions']
 if f=='source_rms_point_products.h':rms[l]+=c
 if fn=='encode_operand':role='encoding'
 elif fn=='evaluate_products':
  if f=='provider.c'and 126<=l<=135:role='fused_integer_reconstruction'
  elif f=='provider.c'and l==530:role='row_scale_finish'
  elif f=='provider.c'and 515<=l<=528:role='product_loop_and_admission'
  elif f=='source_rms_point_products.h'and l==57:role='rms_per_output_checks'
  elif f=='source_rms_point_products.h'and 35<=l<=46:role='rms_operand_norm_scans'
  else:role='bounds_and_norms_other'
 elif '_products_'in fn:role='callback_CPU_body'
 elif fn in ('group_provider_with_rhs','group_provider','certify_frontier'):
  if f=='provider.c'and 413<=l<=434:role='gather_mask_validation'
  elif f=='bf16_quant_frontier.h':role='quantizer_observation'
  elif 'polynomial'in f:role='polynomial_endpoint_propagation'
  elif f in ('f32_interval_endpoint.h','prepared_softmax_interval.h','prepared_softmax_word_domain.h','prepared_bf16_interval.h','private_softmax_interval.h'):role='interval_observer_helpers'
  elif f=='provider.c'and 224<=l<=350:role='softmax_and_ordered_source_refinement'
  elif f=='provider.c'and 534<=l<=581:role='final_observer_refinement'
  else:role='provider_other_unresolved'
 else:continue
 roles[role]+=c
roi=1713849505;roles['unassigned_ROI_shared_callees_and_runtime']=roi-sum(roles.values());assert sum(roles.values())==roi
assert sum(x['count']for x in check_pcs)==26607616
assert sum(x['count']for x in calls if x['callee']=='__truncsfbf2'and x['caller']=='source_quant_frontier')==entries['__truncsfbf2']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(__file__),A/'attribution.json',A/'symbolized.jsonl',A/'model.elf',G/'strict/stderr',G/'strict/stdout',G/'candidate/model.elf',G/'candidate/target_numeric/provider.c',G/'candidate/target_numeric/source_rms_point_products.h',OUT/'disassembly.txt']
r=dict(scope='One actual complete12-head group, immutable exact-row source. Instructions only, not stock cycles or full48 prediction.',elf_sha256=sha(G/'candidate/model.elf'),ROI=roi,roles=dict(roles),rms_header_by_line=dict(rms),checks=check_pcs,check_opcodes=dict(classes),entries={k:entries.get(k)for k in ('evaluate_products','encode_operand','group_provider_with_rhs','certify_frontier')},row_model=dict(scope='Amortized accounting per head/query row, not measured isolated row: shared K/V encoding, callbacks, refinement and allocation remain allocated averages.',rows=12*256,roles={k:v/(12*256)for k,v in roles.items()},total=roi/(12*256)),source_authority=dict(producer='successful five synchronous canonical complete readouts; bounded i64 weighted prefix; single exact f64 conversion; power-of-two row scaling',missing_contract='Current encoded-row proof authenticates source/reconstruction operands only. A producer-issued center witness must additionally bind callback contract, complete plane coverage, shape, steps, output owner and live epoch.',maximum_check_saving=26607616,maximum_ROI_fraction=26607616/roi,normal_group_geometry='No current per-head/per-QK/PV histogram split; aggregate execution cannot be relabeled as a particular attention row.'),post_ROI_excluded='source_quant_frontier and all __truncsfbf2 executions occur only in two original/candidate compiledconsumer validations. Shared rintf remains unassigned; no proportional call allocation.',pins={str(p.resolve()):sha(p)for p in paths})
(OUT/'census.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'roles':roles,'checks':check_pcs,'rows':r['row_model']['total']},indent=2))
