"""Strict qualification of the common-address mcycle group pair."""
from pathlib import Path
import hashlib,json,subprocess,time
from mlir_oot.no_fsm_audit import audit_elf
root=Path(__file__).resolve().parents[2];w=root/'out/exact_row_clock_pair'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records={};maps=[]
for arm in ('control_aligned','candidate_aligned'):
 d=w/arm;a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'nofsm.json').write_text(json.dumps(a,indent=2)+'\n')
 symbols=subprocess.check_output(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm','-S',str(d/'model.elf')],text=True)
 maps.append({t[3]:t[:3]for line in symbols.splitlines()if len(t:=line.split())==4 and t[2]in'bBdDrRsS'})
 run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
 with(d/'stdout').open('w')as out,(d/'stderr').open('w')as err:p=subprocess.run(run,stdout=out,stderr=err)
 text=(d/'stdout').read_text();assert p.returncode==0 and 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in text
 expected=[4391,0,42210,63537,4066368,63,1848,315392]
 assert all(f'WORKSPACE_STAT {i} {x}\n'in text for i,x in enumerate(expected))
 assert 'UNOBSERVED_CARRIER_DIFFERENCES 3\n'in text
 record=dict(command=run,returncode=p.returncode,seconds=time.time()-start,elf_sha256=sha(d/'model.elf'),strict_counter_scope='Spike mcycle retirement proxy; not stock cycles',stats=expected)
 (d/'terminal.json').write_text(json.dumps(record,indent=2)+'\n');records[arm]=record;print(arm,text,flush=True)
assert maps[0]==maps[1]
paths=[p for p in w.rglob('*')if p.is_file()]
for arm in records:
 paths.extend(Path(x)for x in json.load(open(w/arm/'build.json'))['link']if Path(x).is_file())
paths.extend(root/'experiments/attention_projection_frontier'/n for n in ('build_exact_row_clock_pair.py','align_exact_row_clock_pair.py','qualify_exact_row_clock_pair.py'))
r=dict(schema='exact_row_mcycle_common_address_pair_v1',records=records,data_symbols_equal=maps[0],clock='literal CSR mcycle; WORKSPACE_GROUP_CYCLES',historical_pair='First label-only pair retained; data shifted192 bytes. Aligned successor has all data symbols identical.',pins={str(p.resolve()):sha(p)for p in paths})
(w/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('PINS',len(r['pins']))
