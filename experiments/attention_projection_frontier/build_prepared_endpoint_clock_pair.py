"""Matched explicit mcycle driver and all shared data addresses, frozen bodies."""
from pathlib import Path
import hashlib,json,subprocess,time
from mlir_oot.no_fsm_audit import audit_elf
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-endpoint-dag-v3/clock_pair';W.mkdir(exist_ok=False)
OLD=B/'out/artifacts/probes/prepared-polynomial-constants/clock_pair';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(W/'driver.c').write_bytes((OLD/'driver.c').read_bytes());(W/'common_address.ld').write_bytes((OLD/'common_address.ld').read_bytes())
cmd=[s.replace(str(OLD/'driver.c'),str(W/'driver.c')).replace(str(OLD/'driver.o'),str(W/'driver.o')) for s in json.load(open(OLD/'build.json'))['driver_compile']]
subprocess.run(cmd,check=True);assert (W/'driver.o').read_bytes()==(OLD/'driver.o').read_bytes()
source_link=json.load(open(OLD/'candidate/build.json'))['link'];records={};maps=[]
bodies=[B/'out/artifacts/probes/prepared-polynomial-constants/candidate/target_numeric/provider.o',W.parent/'target_numeric/provider.o']
for arm,body in zip(('control','candidate'),bodies,strict=True):
 d=W/arm;d.mkdir()
 link=[str(body) if s==str(bodies[0]) else str(W/'driver.o') if s==str(OLD/'driver.o') else str(W/'common_address.ld') if s==str(OLD/'common_address.ld') else str(d/'model.elf') if s==str(OLD/'candidate/model.elf') else s for s in source_link]
 assert str(body)in link and str(d/'model.elf')in link
 subprocess.run(link,check=True,capture_output=True)
 audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass'
 symbols=subprocess.check_output(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm','-S',str(d/'model.elf')],text=True);(d/'symbols.txt').write_text(symbols)
 maps.append({t[3]:t[:3]for line in symbols.splitlines()if len(t:=line.split())==4 and t[2]in'bBdDrRsS'})
 records[arm]=dict(link=link,provider_sha256=sha(body),elf_sha256=sha(d/'model.elf'),audit=audit)
 (d/'build.json').write_text(json.dumps(records[arm],indent=2)+'\n')
common=set(maps[0])&set(maps[1]);delta={k:[maps[0][k],maps[1][k]] for k in common if maps[0][k]!=maps[1][k]}
(W/'address_comparison.json').write_text(json.dumps(dict(shared_symbols=len(common),delta=delta,control_only={k:maps[0][k]for k in maps[0].keys()-common},candidate_only={k:maps[1][k]for k in maps[1].keys()-common}),indent=2)+'\n');assert not delta,delta
assert (W/'control/model.elf').read_bytes()==(OLD/'candidate/model.elf').read_bytes()
for arm in ('candidate',):
 d=W/arm;run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
 with (d/'stdout').open('w')as out,(d/'stderr').open('w')as err:r=subprocess.run(run,stdout=out,stderr=err)
 text=(d/'stdout').read_text();assert r.returncode==0 and 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
 stats=[4391,0,20866,29436,1883904,29,1656,282624]
 assert all(f'WORKSPACE_STAT {i} {value}\n'in text for i,value in enumerate(stats));assert f'UNOBSERVED_CARRIER_DIFFERENCES {2}\n'in text
 (d/'terminal.json').write_text(json.dumps(dict(command=run,returncode=r.returncode,seconds=time.time()-start,stats=stats,clock='Spike mcycle retirement proxy; stock timing unknown'),indent=2)+'\n');print(arm,text,flush=True)
(W/'build.json').write_text(json.dumps(dict(driver_compile=cmd,records=records),indent=2)+'\n')
