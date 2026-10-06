from pathlib import Path
from dataclasses import asdict
import hashlib,json,re
from merlin.perf.layer_bench import run_on_gsim
W=Path(__file__).resolve().parent/'complete_m2'
q=json.loads((W/'qualification.json').read_text())
for path,digest in q['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
assert not (W/'gsim_receipt.json').exists()
r=run_on_gsim(Path(q['elf_path']),target='gemmini',timeout_s=3600,max_cycles=45000000,stdout_path=W/'gsim.stdout')
(W/'gsim_receipt.json').write_text(json.dumps(asdict(r),indent=2,default=str)+'\n')
text=(W/'gsim.stdout').read_text()
events=[tuple(map(int,m)) for m in re.findall(r'SOURCE_TABLE_PAIR_CYCLES (\d+) (\d+) (\d+)',text)]
if len(events)==4 and 'ORIGINAL_SOURCE_TABLE_PAIR PASS 11264 128' in text:
    costs={arm:[cycles for index,a,cycles in events if a==arm] for arm in (0,1)}
    means={arm:sum(values)/len(values) for arm,values in costs.items()}
    paired={'schema':'source_expression_interval_lookup_complete_gsim_pair_v1','events':events,'baseline_mean_cycles':means[0],'candidate_mean_cycles':means[1],'fraction_change':means[1]/means[0]-1,'engine':'actual pinned Gemmini/Rocket GSIM, not stock FireSim; simulator total is outside ROI','scope':q['scope'],'entire_table_bytes':8388608,'table_initialization':'compile-time immutable data shipped in ELF; no runtime generation excluded','first_original45056_certification':'45043 accepted,13 source replays; original complete M2 timing includes actual fallback at these original inputs','instruction_warning':'568974 approximately baseline vs630956 candidate retired instructions, no cycles projection','gates':q['gates'],'whole_model':'UNKNOWN; no production/default/hardware promotion','qualified_elf_sha256':q['elf_sha256'],'qualification_sha256':hashlib.sha256((W/'qualification.json').read_bytes()).hexdigest(),'gsim_receipt_sha256':hashlib.sha256((W/'gsim_receipt.json').read_bytes()).hexdigest(),'gsim_stdout_sha256':hashlib.sha256((W/'gsim.stdout').read_bytes()).hexdigest()}
    (W/'paired_cost.json').write_text(json.dumps(paired,indent=2)+'\n');print(json.dumps(paired,indent=2),flush=True)
else:print(json.dumps(asdict(r),indent=2,default=str),flush=True)
