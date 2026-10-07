"""Independently close the normal typed masked whole-model stock observation."""

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
    terminal=archive/'stock2062_tiny_masked_terminal.json'
    uart_path=archive/'stock2062_tiny_masked_uart.txt'
    release_path=records/'root_tiny_masked_whole_stock_release_20261007.json'
    output=records/'root_tiny_stock2062_masked_whole_terminal_review_20261007.json'
    if output.exists():
        raise ValueError('fresh terminal review required')
    observation=json.loads(terminal.read_text())
    release=json.loads(release_path.read_text())
    for p,h in observation['pins'].items():
        assert sha(p)==h,p
    v=observation['record']
    assert v['job_id']==2062 and v['state']==v['phase']=='DONE' and v['exit_code']==0
    assert v['hw_config']==release['hardware_alias']=='alveo_u250_firesim_gemmini_rocket_stock'
    elf=Path(v['elf'])
    assert sha(elf)==v['elf_sha256']==release['candidate_elf_sha256']
    assert str(elf)==release['candidate']
    assert sha(uart_path)==v['uart_sha256']==sha(v['uart'])
    uart=uart_path.read_text()
    assert uart.splitlines().count('DONE')==1 and uart.splitlines().count('METRIC memref_rank_mismatch 0')==1
    assert re.findall(r'^METRIC cycles (\d+)\s*$',uart,re.M)==['410147055']
    raw=release['raw_original_sha256']
    assert f'OUT_SHA256 f32le 256000 1024000 {raw}' in uart.splitlines()
    e=v['output_digest_evidence']
    assert sha(e['reference'])==e['reference_sha256'] and sha(e['validation'])==e['validation_sha256']
    words=np.load(e['reference'],allow_pickle=False)
    assert words.dtype==np.float32 and words.shape==(1,8,32000)
    assert hashlib.sha256(words.astype('<f4').tobytes()).hexdigest()==raw==e['sha256']
    assert release['Torch_gate']=={'atol':.03125,'rtol':.02,'passed':True}
    assert v['control_job']==observation['control']['job']==release['control_job']==2056
    control=release['control_cycles']
    candidate=v['kernel_cycles']
    assert control==observation['control']['cycles']==412672134 and candidate==410147055
    assert observation['cycle_reduction_fraction']==1-candidate/control
    audit=audit_elf(elf.read_bytes())
    assert audit['status']=='pass'
    staged=Path(v['actual_simulated_elf'])
    if staged.is_file():
        assert sha(staged)==v['elf_sha256']
    paths=(terminal,uart_path,elf,Path(e['reference']),Path(e['validation']),release_path,Path(__file__))
    result={'schema':'root_tiny_stock2062_masked_whole_terminal_review_v1','status':'champion_whole_model_stock_observation','job_id':2062,'stock_cycles':candidate,'control_job':2056,'control_cycles':control,'difference_cycles':candidate-control,'reduction_percent':100*(control-candidate)/control,'target_cycles':300000000,'gap_cycles':candidate-300000000,'terminal_pins_reclosed':len(observation['pins']),'whole_timer_done_rank_digest_independently_reparsed':True,'original256000_reference_words_independently_digest_exact':True,'original_Torch_gate_unchanged':release['Torch_gate'],'actual_staged_elf_available_for_root_rehash':staged.is_file(),'nofsm_all_executable_sections':audit,'bitstream_sha256':v['bitstream_sha256'],'scope':'One observation per arm, typed closed mask added to unchanged source continuation. Compound16.38% and strictretirement1.57% are not whole-cycle predictions. No refit or workload selector. Other11 linked objects remain as independently released.','pins':{str(p):sha(p) for p in paths}}
    output.write_text(json.dumps(result,indent=2)+'\n')
    for p in (terminal,uart_path):
        shutil.copy2(p,records/p.name)
    print(json.dumps({'status':'PASS','job':2062,'cycles':candidate,'reduction_percent':result['reduction_percent'],'original_outputs':256000}))


if __name__=='__main__':
    close()
