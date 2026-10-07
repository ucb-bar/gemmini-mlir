"""Source-derived16-bit representation and bounded partition cost census."""
from pathlib import Path
import collections,hashlib,json
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/exponent_partition';p=w/'qualification.json';d=json.loads(p.read_text());groups=collections.defaultdict(list)
for row in d['exponent_census']:groups[(row['m'],row['n'],row['k'])].append(row)
summary=[]
for shape,rows in groups.items():
 totals={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('index','m','n','k')}
 total=totals['full_hot_macs']+totals['full_cold_macs'];totals.update(shape=shape,calls=len(rows),useful_macs=total,full_hot_fraction=totals['full_hot_macs']/total,block16_hot_fraction=totals['block16_hot_macs']/total);summary.append(totals)
files={x.resolve() for x in w.rglob('*') if x.is_file()};files.update([base/'experiments/attention_projection_frontier/screen_exponent_partition.py',Path(__file__).resolve(),base/'out/exponent_partition_driver.py',base/'out/normal_composition/qualification_source_rebound.json'])
for command in d['commands']:
 for item in command:
  q=Path(item)
  if q.is_file():files.add(q.resolve())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(scope='Read-only actual48 current normal source/RMS4 operand consumptions; no new representation executed. Original1600 outputs bitexact,23040 callbacks/zero fallback. Fixed16-coordinate partition is analytical only; counts are not hardware prices.',fixed_representation='Two signed8 radix256 limbs with fixed+128 integer offset span the full signed16 integer domain. Exact offset correction needs source row/column integer sums and per-output arithmetic; not implemented or priced here.',summary=summary,required_costs=['mandatory source exponent/range scans and owner epoch binding','irregular eligible row/column compaction and scatter','three i32 readouts per independently scaled K partition','ordered/certified host finishing of differently scaled partial outputs','cold source or original9-term fallback and exact original observation/replay','fixed+128 row/column sum correction and overflow proof'],conclusion='QK source values have few unrepresentable words; PV probabilities have many. Fixed16-coordinate partitions increase aggregate hot readout traffic before cold work, so naive cold source fallback cannot establish a5B whole route. A source-derived sparse exact correction may screen QK-like domains without stage/name selection; no policy or eligibility granted.',pins={str(x):sha(x) for x in sorted(files)})
p=base/'docs/perf_records/attention_exponent_partition_census.json';p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(summary,indent=2));print('PINS',len(files))
