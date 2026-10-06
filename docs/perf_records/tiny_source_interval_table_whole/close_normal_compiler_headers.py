from pathlib import Path
import json,hashlib,subprocess,shlex
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';D=W/'compiler_header_closure';D.mkdir(exist_ok=False)
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
records=[]
for label,source,expected,cflags in [
 ('native_lookup',W/'lookup.c',W/'host_llvm/lookup.ll',['-O3','-ffp-contract=off','-fPIC']),
 ('native_guard',W/'host_llvm/native_guard.c',W/'host_llvm/native_guard.ll',['-O3','-ffp-contract=off','-fPIC']),
 ('target_lookup',W/'target_v2/lookup.c',W/'target_v2/host_llvm/lookup.ll',flags),
 ('target_guard',W/'target_v2/host_llvm/target_guard.c',W/'target_v2/host_llvm/target_guard.ll',flags),
]:
 dep=D/(label+'.d');out=D/(label+'.ll')
 argv=[str(LLVM/'clang'),*cflags,'-MD','-MF',str(dep),'-S','-emit-llvm',str(source),'-o',str(out)]
 p=subprocess.run(argv,capture_output=True,text=True);assert p.returncode==0,p.stderr
 assert sha(out)==sha(expected)
 text=dep.read_text().replace('\\\n',' ');names=shlex.split(text.split(':',1)[1]);paths=[Path(n).resolve()for n in names]
 assert source.resolve()in paths and all(p.is_file()for p in paths)
 records.append(dict(label=label,actual_argv=argv,recorded_compile_LLVM_sha256=sha(expected),dependency_flag_rebuild_byteidentical=True,pins={str(p):sha(p)for p in [*paths,source,expected,out,dep,LLVM/'clang']}))
(D/'receipt.json').write_text(json.dumps(dict(schema='source_interval_normal_compiler_reported_header_closure_v1',scope='Fresh -MD compiler-reported local/resource/system header closure and LLVM rebuild byte-identical to actual qualified helpercompileoutputs; underlying qualified outputs/ELFs unchanged.',records=records),indent=2)+'\n')
print('COMPILER_REPORTED_HEADER_AND_LLVM_REBUILD_IDENTITY_PASS',sum(len(r['pins'])for r in records),flush=True)
