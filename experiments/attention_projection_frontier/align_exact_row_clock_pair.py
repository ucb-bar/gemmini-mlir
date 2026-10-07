from pathlib import Path
import json,subprocess
w=Path(__file__).resolve().parents[2]/'out/exact_row_clock_pair'
old=Path('/scratch/agustin/tmp/gemmini-attention-prefix-guard-20261005/out/artifacts/probes/attention-prefix-guard-20261005/v0_h1_factored/build/link.derived.ld')
s=old.read_text().replace('.text : { *(.text) }','.text : { *(.text .text.*) }\n  . = ALIGN(0x10000);\n  .rodata : { *(.rodata .rodata.*) }')
for key in ('.data :','.sdata :','.sbss :','.bss :','.tdata :','.tbss :'):
 s=s.replace('  '+key,'  . = ALIGN(0x10000);\n  '+key)
(w/'common_address.ld').write_text(s)
b=json.load(open(w/'build.json'))
for arm,r in b['records'].items():
 d=w/(arm+'_aligned');d.mkdir(exist_ok=False)
 cmd=[str(w/'common_address.ld')if x==str(old)else str(d/'model.elf')if x==str(w/arm/'model.elf')else x for x in r['link']]
 subprocess.run(cmd,check=True,capture_output=True)
 (d/'build.json').write_text(json.dumps({'link':cmd}))
