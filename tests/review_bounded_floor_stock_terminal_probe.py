"""Close actual group hardware and score immutable forecasts without refitting."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def close():
    root=Path(__file__).resolve().parents[1]
    records=root/'docs/perf_records'
    terminal_path=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records/stock2064_bounded_floor_terminal.json')
    release_path=records/'root_prepared_bounded_floor_stock_release_20261007.json'
    model_path=records/'root_smol_complete_group_blinded_retirement_screen_20261007.json'
    output=records/'root_bounded_floor_stock2064_terminal_review_20261007.json'
    if output.exists():
        raise ValueError('fresh terminal review required')
    stock,release,model=[json.loads(p.read_text()) for p in (terminal_path,release_path,model_path)]
    for receipt in (stock,model):
        for p,h in receipt['pins'].items():
            if sha(p)!=h:
                raise ValueError('frozen input changed: '+p)
    record=stock['record']
    if record['job_id']!=2064 or record['state']!='DONE' or record['phase']!='DONE' or record['exit_code']!=0:
        raise ValueError('stock terminal failed')
    if record['elf_sha256']!=release['candidate_elf_sha256'] or record['hw_config']!=release['hardware_alias'] or any(record[k]!=v for k,v in release['hardware_identity_required'].items()):
        raise ValueError('released executable/hardware differs')
    uart_path=Path(stock['raw_uart_archive'])
    if sha(uart_path)!=record['uart_sha256'] or sha(record['elf'])!=record['elf_sha256']:
        raise ValueError('final executable/UART changed')
    console=uart_path.read_text().replace('\r','')
    if re.findall(r'^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$',console,re.M)!=[str(record['kernel_cycles'])]:
        raise ValueError('actual hardware counter differs')
    if re.findall(r'^WORKSPACE_STAT (\d+) (\d+)$',console,re.M)!=[(str(i),str(v)) for i,v in enumerate(release['uart_required']['stats'])] or release['uart_required']['pass_marker'] not in console or re.findall(r'^UNOBSERVED_CARRIER_DIFFERENCES (\d+)$',console,re.M)!=['3'] or 'CAPSULE UNEXPECTED SOURCE REFUSAL' in console or '*** PASSED ***' not in console:
        raise ValueError('full consumer/statistics/input/guard check failed')
    audit=audit_elf(Path(record['elf']).read_bytes())
    if audit['status']!='pass':
        raise ValueError('final noFSM failed')
    actual=record['kernel_cycles']
    if actual!=4256146700 or stock['control']['cycles']!=release['control_cycles'] or 1-actual/release['control_cycles']!=stock['cycle_reduction_fraction']:
        raise ValueError('measured comparison differs')
    ratio=release['conditional_pretiming_screen']
    if model['existing_prequeue_count_ratio_forecast_unchanged']!=ratio or model['held_elf_sha256']!=record['elf_sha256'] or model['held_job']!=2064:
        raise ValueError('blinded fit or original prequeue forecast changed')
    comparisons=[{'name':'prequeue_control_retirement_ratio','predicted_cycles':ratio['candidate_group_cycles'],'actual_cycles':actual,'relative_error':abs(ratio['candidate_group_cycles']-actual)/actual,'coefficients_fitted':False,'new_label_fitted':False}]
    for forecast in model['forecasts']:
        if 'refused' in forecast:
            comparisons.append({'name':forecast['name'],'refused':forecast['refused'],'new_label_fitted':False})
            continue
        guessed=forecast['conditional_extrapolation_cycles']
        recomputed=forecast['fixed_cycles']+forecast['coefficients'][0]*release['strict_instructions_after']
        if guessed!=recomputed or forecast['domain_prediction']['resolved']:
            raise ValueError('frozen arithmetic or original range refusal changed')
        comparisons.append({'name':forecast['name'],'fit_sha256_unchanged':forecast['fit_sha256'],'predicted_cycles':guessed,'actual_cycles':actual,'relative_error':abs(guessed-actual)/actual,'domain_prediction_unchanged':forecast['domain_prediction'],'new_label_fitted':False,'scope':'Six historical same-workload fits, blind to2064label; not held-workload qualification or an approved extrapolation.'})
    paths=(terminal_path,uart_path,release_path,model_path,Path(record['elf']),Path(__file__))
    result={'schema':'root_bounded_floor_stock2064_terminal_review_v1','status':'VERIFIED_COMPLETE_SOURCE_GROUP_STOCK_WIN','job_id':2064,'stock_cycles':actual,'control_job':2060,'control_cycles':release['control_cycles'],'fraction_lower':1-actual/release['control_cycles'],'original2024_cycles':4857792055,'fraction_lower_than2024':1-actual/4857792055,'terminal_pins_reclosed':len(stock['pins']),'actual_ROI_UART_and_full_consumer_contract_reparsed':True,'original_whole_native_words':1600,'output_checksum':'NOT_EMITTED; compiled786432i8/1024BF16 consumer byte comparisons pass','nofsm':audit,'hardware_identity':{k:record[k] for k in ('hw_config','hwdb_sha256','bitstream_sha256')},'frozen_forecast_scores':comparisons,'new_label_fitted':False,'whole_hardware_cycles':'UNKNOWN','automatic_export':False,'staging_scope':'Actual simulation staging/bitstream receipts reclosed from collector; source ELF rehashed. Staged artifact may have been removed by teardown.','pins':{str(p):sha(p) for p in paths}}
    output.write_text(json.dumps(result,indent=2)+'\n')
    for p in (terminal_path,uart_path):
        shutil.copy2(p,records/p.name)
    print(json.dumps({'status':result['status'],'cycles':actual,'fraction_lower':result['fraction_lower'],'errors':{c['name']:c.get('relative_error') for c in comparisons},'whole':'UNKNOWN','refitted':False}))


if __name__=='__main__':
    close()
