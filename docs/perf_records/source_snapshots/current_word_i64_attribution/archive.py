from pathlib import Path
import json,hashlib,datetime,shutil
w=Path(__file__).parent;r=w.parents[1];hist=w.parent/'current_word_i64_pc_histogram';old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p,h=None):
 p=Path(p).resolve();s=sha(p)
 if h is not None:assert s==h,(p,s,h)
 pins[str(p)]=s
q=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/docs/perf_records/word_soft_i64_complete_group_qualification.json');qualification=json.loads(q.read_text())
for p,h in qualification['pins'].items():pin(p,h)
pin(q)
receipt=json.loads((hist/'receipt.json').read_text());assert receipt['status']=='pass';pin(old/'candidate/model.elf',receipt['elf_sha256']);pin(hist/'spike.log',receipt['log_sha256'])
for p in list(w.glob('*'))+list(hist.glob('*')):
 if p.is_file():pin(p)
roi=json.loads((w/'roi_attribution.json').read_text());assert sum(roi['known_roi_categories'].values())+roi['roi_unresolved_shared_callees_and_boundary']==roi['roi_instructions']==2537394919
snap=r/'docs/perf_records/source_snapshots/current_word_i64_attribution';snap.mkdir(parents=True,exist_ok=True)
for p in list(w.glob('*.py'))+[hist/'watch.py',r/'out/profile_pc_counts.py']:
 d=snap/p.name;shutil.copyfile(p,d);pin(d)
result={'schema':'current_word_i64_provider_attribution_qualification_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','elf_sha256':receipt['elf_sha256'],'execution':receipt,'binary_identity':json.loads((w/'generic_binary_closure.json').read_text()),'roi_attribution':roi,'source_attribution':json.loads((w/'attribution.json').read_text()),'unchanged_original_consumer_and_all_eight_stats':True,'whole_model_execution':False,'hardware_admission':False,'pins':pins,'token_usage_available':False}
out=r/'docs/perf_records/current_word_i64_provider_attribution.json';out.write_text(json.dumps(result,indent=2)+'\n');print(out,len(pins))
