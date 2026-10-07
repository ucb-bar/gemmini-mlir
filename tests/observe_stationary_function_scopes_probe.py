"""Attribute already qualified logical accelerator requests to ELF functions.

Scopes are experimental attribution, not compiler strategy selectors. Alias
ranges stay explicit and entry events are not assumed to be semantic calls.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
from collections import Counter
from pathlib import Path

from merlin.targetgen.elf_lanes import executable_sections
from mlir_oot.no_fsm_audit import _instruction_bytes
from mlir_oot.tables import rtl_facts as F


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def main(args):
    original=json.loads(args.original.read_text())
    if original['status'] != 'PASS':
        raise ValueError('complete original census is unqualified')
    elf=Path(original['pins']['elf']['path'])
    if sha(elf) != original['pins']['elf']['sha256']:
        raise ValueError('final ELF changed')
    old=original['providers']['stationary']
    for ref in old['pins'].values():
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('original complete output/PC/census changed')
    args.output.mkdir(parents=True,exist_ok=False)
    data=elf.read_bytes()
    custom=[]
    for _,offset,length,address in executable_sections(data):
        pos=0
        while pos<length:
            width=_instruction_bytes(struct.unpack_from('<H',data,offset+pos)[0])
            word=int.from_bytes(data[offset+pos:offset+pos+width],'little')
            if width==4 and word&0x7f == F.CUSTOM_OPCODE:
                custom.append(address+pos)
            pos+=width
    symbols=subprocess.run(['readelf','-sW',str(elf)],capture_output=True,text=True,check=True).stdout
    ranges={}
    for line in symbols.splitlines():
        fields=line.split()
        if len(fields)<8 or fields[3]!='FUNC' or fields[6] in ('UND','ABS'):
            continue
        start=int(fields[1],16)
        size=int(fields[2],16 if fields[2].startswith('0x') else 10)
        selected=[pc for pc in custom if start<=pc<start+size]
        if not selected:
            continue
        key=(start,start+size)
        ranges.setdefault(key,{'begin':start,'end':start+size,'entry':min(selected),'symbols':[]})['symbols'].append(fields[-1])
    scopes=[]
    for i,(_,row) in enumerate(sorted(ranges.items())):
        if scopes and scopes[-1]['end']>row['begin']:
            raise ValueError('accelerator-containing ELF function ranges overlap')
        scopes.append({'id':i+1,**row})
    scope_path=args.output/'scopes.txt'
    scope_path.write_text(''.join(f"{r['id']} {r['begin']:#x} {r['end']:#x} {r['entry']:#x}\n" for r in scopes))
    command=list(old['command'])
    env=dict(os.environ)
    env.update(MERLIN_GEMMINI_TELEMETRY=str((args.output/'operands.json').resolve()),
               MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC='1',
               MERLIN_GEMMINI_TELEMETRY_SCOPES=str(scope_path.resolve()))
    start=time.monotonic()
    with (args.output/'stdout').open('wb') as stdout,(args.output/'histogram').open('wb') as stderr:
        result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=180)
    elapsed=time.monotonic()-start
    if result.returncode:
        raise ValueError('scoped complete functional execution failed')
    for name in ('stdout','histogram'):
        if (args.output/name).read_bytes()!=Path(old['pins'][name]['path']).read_bytes():
            raise ValueError('scope attribution altered any output or target PC count')
    new=json.loads((args.output/'operands.json').read_text())
    previous=json.loads(Path(old['pins']['operands.json']['path']).read_text())
    values=('commands','requested_load_bytes','requested_store_bytes','unknown_dma_commands','padded_mac_slots','padded_compute_rows','preload_real_stationary_b','preload_retained_stationary_b','preload_unknown_mode')
    def aggregate(record):
        totals=Counter()
        for row in record['rows']:
            key=tuple((k,v) for k,v in row.items() if k not in (*values,'scope_id','invocation','first_event','last_event'))
            for name in values:
                totals[(key,name)]+=row[name]
        return totals
    if aggregate(new)!=aggregate(previous):
        raise ValueError('any per-geometry request counter changed')
    groups={r['id']:{**r,'requests':{k:0 for k in values}} for r in scopes}
    for row in new['rows']:
        if row['scope_id'] not in groups:
            raise ValueError('executed custom instruction has no complete ELF function scope')
        for name in values:
            groups[row['scope_id']]['requests'][name]+=row[name]
    active=[row for row in groups.values() if row['requests']['commands']]
    record={'schema':'stationary_requests_by_elf_function_v1','status':'PASS',
            'command':command,'exit_code':result.returncode,'elapsed_seconds':elapsed,
            'whole_original_output_and_pc_histogram_byte_exact':True,
            'every_original_request_geometry_counter_exact':True,
            'groups':active,'inactive_functions':len(scopes)-len(active),
            'cycle_prediction':'UNKNOWN; geometry/request attribution has no fitted duration',
            'scope':'Actual accelerator-containing ELF functions, aliases explicit; scope-entry events not assumed semantic invocation counts. All target bytes, source words and existing logical requests conserved.',
            'pins':{k:pin(path) for k,path in {'original':args.original,'elf':elf,'scopes':scope_path,'stdout':args.output/'stdout','histogram':args.output/'histogram','telemetry':args.output/'operands.json','script':Path(__file__)}.items()}}
    (args.output/'request_functions.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'PASS','active_functions':len(active),'elapsed_seconds':elapsed,'padded_compute_rows':sum(x['requests']['padded_compute_rows'] for x in active)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
