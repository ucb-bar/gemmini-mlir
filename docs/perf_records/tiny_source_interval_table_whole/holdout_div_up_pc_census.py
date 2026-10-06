"""Actual same-ELF instruction histogram; model-independent opcode census."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,subprocess
W=Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006/out/artifacts/probes/pointwise-dependency-schedule-20261006/div_up_packet_m2')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
elf=W/'build/layer.elf'
objdump=subprocess.check_output([str(LLVM/'llvm-objdump'),'-d','--no-show-raw-insn',str(elf)],text=True)
(W/'disassembly.txt').write_text(objdump)
nm=subprocess.check_output([str(LLVM/'llvm-nm'),'-S','--defined-only',str(elf)],text=True)
(W/'symbols.txt').write_text(nm)
symbols={m[4]:(int(m[1],16),int(m[2],16)) for line in nm.splitlines() if (m:=re.fullmatch(r'([0-9a-f]+) ([0-9a-f]+) ([A-Za-z]) (.+)',line))}
hist={int(m[1],16):int(m[2]) for line in (W/'histogram.stderr').read_text().splitlines() if (m:=re.fullmatch(r'([0-9a-f]+) (\d+)',line))}
instructions={int(m[1],16):(m[2],m[3]) for line in objdump.splitlines() if (m:=re.match(r'\s*([0-9a-f]+):\s+([\w.]+)\s*(.*)',line))}
def kind(op):
    if op.startswith(('fmadd','fmsub','fnmadd','fnmsub')):return 'FP_FMA'
    if op.startswith('fdiv'):return 'FP_DIV'
    if op.startswith(('flw','fld')):return 'FP_load'
    if op.startswith(('fsw','fsd')):return 'FP_store'
    if op.startswith('fcvt'):return 'FP_conversion'
    if op.startswith(('fadd','fsub','fmul')):return 'FP_add_mul'
    if op.startswith(('fmax','fmin','feq','flt','fle')):return 'FP_compare_clamp'
    if op.startswith(('beq','bne','blt','bge','ble','bgt','j')):return 'branch_jump'
    if op in ('lb','lbu','lh','lhu','lw','lwu','ld'):return 'integer_load'
    if op in ('sb','sh','sw','sd'):return 'integer_store'
    return 'other'
records={}
for name in ('baseline','candidate'):
    base,size=symbols[name];pcs={pc:count for pc,count in hist.items() if base<=pc<base+size}
    calls=hist[base];ops=Counter();classes=Counter();stack=Counter()
    for pc,count in pcs.items():
        op,args=instructions[pc];ops[op]+=count;classes[kind(op)]+=count
        if '(sp)' in args:stack[kind(op)]+=count
    records[name]={'address':hex(base),'bytes':size,'actual_entry_calls':calls,'total_dynamic_instructions':sum(pcs.values()),'dynamic_opcodes':dict(ops),'dynamic_classes':dict(classes),'per_call_classes':{k:v/calls for k,v in classes.items()},'per_call_instructions':sum(pcs.values())/calls,'stack_traffic_dynamic':dict(stack),'scope':'Whole helper including prologue/epilogue+all repeated scalar loop elements; histogram includes originalwarm+ABBAcalls,3callsperarm, identicaloriginalinputs. SourceexactDIV/up interleaving preservesallnumericwords/flags; no arithmetic/tablepolicy change.'}
record={'schema':'same_elf_source_exact_division_up_pc_census_v1','elf_sha256':hashlib.sha256(elf.read_bytes()).hexdigest(),'records':records,'counter':'Actual Spike-g PC dispatch executions, not hardwarecycles; ordinary scalar helper bodies contain no CSR/serialization instruction. BothsourcehelperbodiesareCSRfree; nocyclelatencyinferred. No fitted latency.','baseline_call_scope':'3originalfullM2callsperarm, warm+ABBAcontrolandcandidate; no cache modelinferred.','pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [elf,W/'histogram.stdout',W/'histogram.stderr',W/'disassembly.txt',W/'symbols.txt',Path(__file__),LLVM/'llvm-objdump',LLVM/'llvm-nm']}}
(W/'pc_census.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:{s:v[s] for s in ('actual_entry_calls','per_call_instructions','per_call_classes')} for k,v in records.items()},indent=2),flush=True)
