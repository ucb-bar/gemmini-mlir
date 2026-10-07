"""Attribute current source polynomial PCs without guessing line-zero roles."""
from pathlib import Path
import collections, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'out/exact_row_attribution'; G=ROOT/'out/exact_row_group'
D=ROOT/'out/exact_row_role_census/disassembly.txt'
out=ROOT/'out/artifacts/audits/prepared-polynomial-opcodes';out.mkdir(parents=True,exist_ok=False)
hist={}
for line in (G/'strict/stderr').read_text().splitlines():
 t=line.split()
 if len(t)==2:
  try:hist[int(t[0],16)]=int(t[1])
  except ValueError:pass
records=[json.loads(x) for x in (A/'symbolized.jsonl').read_text().splitlines()]
locations={p:r['Symbol'] for p,r in zip(sorted(hist),records,strict=True)}
opcodes=collections.Counter();lines=collections.defaultdict(collections.Counter);pcs=[];fn=''
for line in D.read_text().splitlines():
 t=line.split()
 if len(t)==2 and t[1].startswith('<'):fn=t[1][1:-2];continue
 if not t or not t[0].endswith(':'):continue
 try:pc=int(t[0][:-1],16)
 except ValueError:continue
 count=hist.get(pc,0)
 hits=[s for s in locations.get(pc,[]) if Path(s['FileName']).name=='prepared_polynomial_batch.h']
 if not count or not hits:continue
 assert fn=='group_provider_with_rhs',fn
 loc=hits[0];opcodes[t[1]]+=count;lines[loc['Line']][t[1]]+=count
 pcs.append(dict(pc=hex(pc),count=count,line=loc['Line'],assembly=' '.join(t[1:])))
assert lines[12]=={'flt.s':3145728,'bnez':3145728}
assert not any(x['assembly'].split()[0] in ('jal','jalr','call') for x in pcs)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=G/'candidate/target_numeric'
paths=[Path(__file__),D,A/'symbolized.jsonl',G/'strict/stderr',G/'strict/stdout',G/'candidate/model.elf',source/'provider.c',source/'prepared_polynomial_batch.h',source/'prepared_softmax_interval.h']
r=dict(schema='prepared_polynomial_exclusive_opcode_census_v1',scope='Actual current-row first group executed PCs; retired instructions, not cycles. Only header-attributed PCs; caller setup and shared callees remain outside this subset.',instructions=sum(opcodes.values()),opcodes=dict(opcodes),lines={str(k):dict(v) for k,v in lines.items()},pcs=pcs,source_domain=dict(preparation='Positive finite scale and finite FLT_MAX*scale and twice that product admit every finite scaled endpoint and subtraction.',span='Original producer epoch consumption or complete finite/ordered active-span scan; checked exact-source replay retains containment.',maximum='Every active upper*scale <= finite checked source maximum before subtraction. Stable RNE monotonicity gives upper score <= zero.',cutoff='Original clamp before multiplication maps active evaluation to [source cutoff, zero]; below-cutoff zero branch remains. All-masked dummy inputs are zero.',effects='Existing immutable plan/private span, RNE, nontrapping and unobserved flags/errno contracts. No new callback or FENV mutation inside region.',restriction='This does not establish arbitrary input interval validity or permit changing live F32 denominator to BF16.'),finding='Input domain branch costs 6,291,456 instructions. Most line-zero instructions are real arithmetic and loads, not FRM/domain checks. No helper-attributed calls execute. Repeated stack loads and immutable plan constants warrant source-context specialization; gain remains UNKNOWN.',pins={str(p.resolve()):sha(p) for p in paths})
(out/'census.json').write_text(json.dumps(r,indent=2)+'\n')
(ROOT/'docs/perf_records/prepared_polynomial_opcode_census.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'instructions':r['instructions'],'opcodes':r['opcodes']},indent=2))
