from pathlib import Path
import json,subprocess
work=Path(__file__).parent;receipt=json.loads((work/'build.json').read_text())
command=[arg.replace(str(work/'check.c'),str(work/'diagnose.c')).replace(str(work/'check.o'),str(work/'diagnose.o')).replace(str(work/'check.d'),str(work/'diagnose.d'))for arg in receipt['compile']]
subprocess.run(command,check=True)
link=[arg.replace(str(work/'check.o'),str(work/'diagnose.o')).replace(str(work/'check.elf'),str(work/'diagnose.elf'))for arg in receipt['link']]
subprocess.run(link,check=True)
with(work/'diagnose.log').open('w')as file:
 code=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(work/'diagnose.elf')],stdout=file,stderr=subprocess.STDOUT,timeout=10).returncode
print(json.dumps({'returncode':code,'log':(work/'diagnose.log').read_text()}))
