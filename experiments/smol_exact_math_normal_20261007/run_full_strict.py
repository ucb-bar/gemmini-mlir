from pathlib import Path
import hashlib,json,subprocess,time
WORK=Path(__file__).resolve().parent
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
native=json.loads((WORK/'native/validation.json').read_text())
assert native['elements']==1600 and native['bitwise_mismatches']==0 and native['allclose']
elf=WORK/'build/model.elf'; audit=json.loads((WORK/'build/model.nofsm_audit.json').read_text())
assert sha(elf)==audit['elf_sha256'] and audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
spike=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike')
argv=['timeout','--signal=TERM','--kill-after=15s','7200s',str(spike),'-g','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x400000000',str(elf)]
pins={str(p):sha(p)for p in [elf,spike,WORK/'native/validation.json',WORK/'build/model.nofsm_audit.json',WORK/'build/compilation_recipe.json',WORK/'build/lower/lowering_recipe.json',WORK/'build_normal.py',Path(__file__)]}
(WORK/'strict_admission.json').write_text(json.dumps(dict(schema='smol_normal_exact_math_full_strict_admission_v1',status='native_gate_and_final_nofsm_pass',argv=argv,pins=pins,whole_firesim_cycles='UNKNOWN'),indent=2)+'\n')
start=time.monotonic()
with (WORK/'spike.stdout').open('wb')as out,(WORK/'spike.stderr').open('wb')as err:r=subprocess.run(argv,stdout=out,stderr=err,check=False)
terminal=dict(schema='smol_normal_exact_math_full_strict_terminal_v1',returncode=r.returncode,elapsed_seconds=time.monotonic()-start,argv=argv,elf_sha256=sha(elf),stdout_sha256=sha(WORK/'spike.stdout'),histogram_sha256=sha(WORK/'spike.stderr'),scope='Functional source gate and retired-instruction evidence, not FPGA timing.')
(WORK/'strict_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps(terminal),flush=True)
