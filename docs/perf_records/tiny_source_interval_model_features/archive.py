from pathlib import Path
import hashlib,json,shutil
T=Path(__file__).resolve().parent;O=T.parents[3];D=O/'docs/perf_records/tiny_source_interval_model_features';D.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
normal=read(T/'normal_helper_applicability.json');local=read(T/'table_m2_shared_fp_distances.json');pins={}
for p,h in normal['pins'].items():assert sha(p)==h;pins[p]=h
for entry in local['pins']:assert sha(entry['path'])==entry['sha256'];pins[entry['path']]=entry['sha256']
for p in T.glob('*'):
 if p.is_file()and p.name!='archive.py':shutil.copyfile(p,D/p.name);pins[str(p)]=sha(p)
shutil.copyfile(__file__,D/'archive.py');pins[str(Path(__file__))]=sha(__file__)
record=dict(schema='compiler_performance_feature_applicability_adjunct_v1',scope='Read-only actualnormal source/table code addresses and unweighted static opcode counts; existing completeM2sameELF observed hist/CPUFPdistanceproducer and frozenMerlinRAWgraph. No newexecution/simulation, no changes to earlier qualified receipts/ELFs, no fitted/whole prediction.',result='Useful heldout contrastingmixedCPU/8MiBtable workloads: additional memory/branch/rounding work withfewer DIV/FMA and moreinstructions butlowerlocalcycles. Keep completefirstM2actualcost and normalwholeunknown scopes separate. CPU21 classes are represented, table regime/integer/memory/loop/crossBB dependencies remain unpriced.',root_request='Tune performance model with actual complete source instruction/context and calibration applicability; no fit to requested300Mtarget.',tokens=dict(token_usage_available=False,scope='Root owns shared campaign snapshots; exclusive attribution unavailable'),pins=pins)
(O/'docs/perf_records/tiny_source_interval_model_features.json').write_text(json.dumps(record,indent=2)+'\n');print('MODEL_FEATURE_ADJUNCT_ARCHIVED',len(pins),flush=True)
