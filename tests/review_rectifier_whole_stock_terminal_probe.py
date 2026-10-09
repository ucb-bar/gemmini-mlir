"""Independently close the exact whole-model stock replacement observation."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from mlir_oot.no_fsm_audit import audit_elf


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def close():
    root=Path(__file__).resolve().parents[1]
    records=root/'docs/perf_records'
    archive=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records')
    path=archive/'stock2066_rectifier_whole_terminal.json'
    uart_path=archive/'stock2066_rectifier_whole_uart.txt'
    release_path=records/'root_exact_rectifier_whole_stock_release_20261007.json'
    output=records/'root_resnet_stock2066_rectifier_terminal_review_20261007.json'
    if output.exists():
        raise ValueError('fresh independent terminal required')
    stock,release=[json.loads(p.read_text()) for p in (path,release_path)]
    for p,h in stock['pins'].items():
        if sha(p)!=h:
            raise ValueError('terminal evidence changed: '+p)
    v=stock['record']
    if v['job_id']!=2066 or v['state']!='DONE' or v['phase']!='DONE' or v['exit_code']!=0 or v['hw_config']!=release['hardware_alias']:
        raise ValueError('job terminal/config failed')
    if sha(v['elf'])!=v['elf_sha256'] or v['elf_sha256']!=release['candidate_elf_sha256'] or sha(uart_path)!=v['uart_sha256']:
        raise ValueError('released ELF/UART differs')
    uart=uart_path.read_text()
    if uart.splitlines().count('DONE')!=1 or uart.splitlines().count('METRIC memref_rank_mismatch 0')!=1 or re.findall(r'^METRIC cycles (\d+)\s*$',uart,re.M)!=['30169093']:
        raise ValueError('whole actualcounter/terminal failed')
    raw=release['raw_original_sha256']
    if f'OUT_SHA256 f32le 1000 4000 {raw}' not in uart.splitlines():
        raise ValueError('whole original digest failed')
    e=v['output_digest_evidence']
    if sha(e['reference'])!=e['reference_sha256'] or sha(e['validation'])!=e['validation_sha256']:
        raise ValueError('whole standardadapter/reference changed')
    words=np.load(e['reference'],allow_pickle=False)
    if words.dtype!=np.float32 or words.size!=1000 or hashlib.sha256(words.astype('<f4').tobytes()).hexdigest()!=raw or raw!=e['sha256']:
        raise ValueError('original1000 floatwords failed')
    if release['numeric_gate']!={'atol':0,'rtol':0} or release['control_job']!=stock['control']['job'] or release['control_cycles']!=stock['control']['cycles']:
        raise ValueError('whole control/gate differs')
    candidate=v['kernel_cycles']
    control=release['control_cycles']
    if 1-candidate/control!=stock['cycle_reduction_fraction']:
        raise ValueError('whole measured ratio differs')
    audit=audit_elf(Path(v['elf']).read_bytes())
    if audit['status']!='pass':
        raise ValueError('final noFSM failed')
    staged=Path(v['actual_simulated_elf'])
    if staged.is_file() and sha(staged)!=v['elf_sha256']:
        raise ValueError('actual staged executable differs')
    failure_path=archive/'stock2065_rectifier_infrastructure_failure.json'
    failure=json.loads(failure_path.read_text())
    if failure.get('kernel_cycles') is not None:
        raise ValueError('failedsetup cannot count as timingtrial')
    reference_cycles=22387449
    paths=(path,uart_path,release_path,Path(v['elf']),Path(e['reference']),Path(e['validation']),failure_path,Path(__file__))
    result={'schema':'root_resnet_stock2066_rectifier_terminal_review_v1','status':'champion_whole_model_stock_observation','job_id':2066,'stock_cycles':candidate,'control_job':2055,'control_cycles':control,'difference_cycles':candidate-control,'reduction_percent':100*(control-candidate)/control,'permitted_reference_job':1876,'permitted_reference_cycles':reference_cycles,'remaining_reference_gap_cycles':candidate-reference_cycles,'remaining_reference_gap_fraction':candidate/reference_cycles-1,'terminal_pins_reclosed':len(stock['pins']),'original1000_words_digest_independently_reparsed':True,'original_numeric_gate':release['numeric_gate'],'actual_staged_elf_available_for_root_rehash':staged.is_file(),'nofsm_all_executable_sections':audit,'hardware_identity':{k:v[k] for k in ('hw_config','hwdb_sha256','bitstream_sha256')},'failed2065_setup_is_not_measurement':True,'scope':'One successful actualworkload after infrastructure-only2065failure. Owned launcheridentityenvmatches successful2064; nosharedconfig/daemon/foreignaction. Onlyone exactfinite-source targetresidual route changes versus2055; distinctGSIM sectiongains notadditive wholeforecast. Reference fromownedZIP/1876 receipts only.','automatic_default_promotion':False,'pins':{str(p):sha(p) for p in paths}}
    output.write_text(json.dumps(result,indent=2)+'\n')
    for p in (path,uart_path):
        shutil.copy2(p,records/p.name)
    print(json.dumps({'status':'PASS','cycles':candidate,'reduction_percent':result['reduction_percent'],'reference_gap':result['remaining_reference_gap_cycles'],'original_words':1000,'setup_failure_counted':False}))


if __name__=='__main__':
    close()
