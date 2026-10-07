"""Release one exact, controlled whole-model target-route stock observation."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from merlin.targetgen.elf_lanes import executable_sections
from mlir_oot.executed_features import parse_pc_histogram
from mlir_oot.no_fsm_audit import audit_elf


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def code_sections(path):
    blob=path.read_bytes()
    return [(name,address,blob[offset:offset+size]) for name,offset,size,address in executable_sections(blob)]


def review(args):
    if args.output.exists():
        raise ValueError('fresh root release required')
    q=json.loads(args.packet.read_text())
    refs=q['flat_file_pins']
    bindings={}
    for ref in refs:
        p=Path(ref['path'])
        if sha(p)!=ref['sha256'] or p.stat().st_size!=ref['bytes']:
            raise ValueError('frozen artifact differs: '+str(p))
        if str(p) in bindings and bindings[str(p)]!=ref['sha256']:
            raise ValueError('conflicting pins')
        bindings[str(p)]=ref['sha256']
    if q['status']!='PASS' or q['control_job']!=2055 or q['control_stock_cycles']!=30550422 or not q['control_byte_identical']:
        raise ValueError('current exact control gate failed')
    control,candidate=[Path(q[k]['path']) for k in ('control_elf','candidate_elf')]
    if sha(control)!='5b8b00497b390d6225f1e5596f032be007977ccf923411ab2a975cebce44d113' or sha(candidate)!=q['candidate_elf']['sha256']:
        raise ValueError('released whole ELF differs')
    audit=audit_elf(candidate.read_bytes())
    if audit['status']!='pass' or audit['forbidden'] or audit['unknown']:
        raise ValueError('whole noFSM gate failed')
    for gate in (q['normal_whole_native'],q['normal_whole_spike'],q['controlled_whole_gate']['native'],q['controlled_whole_gate']['spike']):
        if gate['elements']!=1000 or not gate['exact_equal'] or gate['atol']!=0 or gate['rtol']!=0 or not gate['quality_pass']:
            raise ValueError('original whole exact gate failed')
    strict=q['controlled_whole_gate']['spike']
    reference=Path(strict['reference_path'])
    original=Path(q['normal_whole_native']['original_golden'])
    words,expected=[np.load(p,allow_pickle=False) for p in (reference,original)]
    if words.dtype!=np.float32 or words.size!=1000 or words.shape!=expected.shape or not np.array_equal(words.view(np.uint32),expected.view(np.uint32)):
        raise ValueError('native original words differ')
    raw=hashlib.sha256(words.astype('<f4').tobytes()).hexdigest()
    if raw!='0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787' or strict['spike_output_sha256']!=raw:
        raise ValueError('original whole digest differs')
    console_path=Path(strict['spike_console_path'])
    if sha(console_path)!=strict['spike_console_sha256']:
        raise ValueError('strict console changed')
    console=console_path.read_text()
    if console.splitlines().count('DONE')!=1 or console.splitlines().count('METRIC memref_rank_mismatch 0')!=1 or f'OUT_SHA256 f32le 1000 4000 {raw}' not in console.splitlines():
        raise ValueError('original whole target terminal differs')
    for identity in q['controlled_whole_gate']['executable_rebinding_identity']:
        before,after=[Path(identity[k]['path']) for k in ('before','after')]
        a,b=[code_sections(p) for p in (before,after)]
        if a!=b:
            raise ValueError('alias rebinding changes executable bytes')
    link=q['controlled_link']
    for p,h in link['retained_input_pins'].items():
        if sha(p)!=h:
            raise ValueError('controlled retained object differs')
    if not link['all_host_runtime_weights_other_target_objects_unchanged'] or not q['runtime_host_weights_other_target_objects_unchanged']:
        raise ValueError('controlled link scope differs')
    active=q['actual_selected_execution']
    hist_path=next(Path(p) for p,h in bindings.items() if h==active['histogram_sha256'])
    histogram=parse_pc_histogram(hist_path.read_text())
    entries={r['symbol']:histogram.get(r['start'],0) for r in active['symbol_ranges']}
    if entries!=active['observed_entry_counts'] or active['returncode']!=0 or list(entries.values())!=[1,1,0,0]:
        raise ValueError('actual selected/retained route differs')
    ranked=json.loads(Path(q['ranked_cost_receipt']['path']).read_text())
    earlier=json.loads(Path(q['earlier39_14_receipt']['path']).read_text())
    # Both immutable matched scopes are distinct; never sum their whole savings.
    if earlier['before']['cycles']!=1996286 or earlier['after']['cycles']!=1821983:
        raise ValueError('original complete39/14 pair changed')
    result={'schema':'root_exact_rectifier_whole_stock_release_v1','status':'RELEASED_ONE_CONTROLLED_WHOLE_STOCK_OBSERVATION','qualification_pins_reclosed':len(refs),'control_job':2055,'control_cycles':30550422,'control_elf_sha256':sha(control),'candidate':str(candidate),'candidate_elf_sha256':sha(candidate),'original_words_exact':1000,'raw_original_sha256':raw,'numeric_gate':{'atol':0,'rtol':0},'normal_source_route_and_controlled_actual2055_host_both_qualified':True,'one_changed_active_target_route':q['one_changed_active_target_route'],'active_entry_counts_rederived':entries,'retained_executable_rebinding_bytes_exact':True,'host_runtime_weights_other_target_objects_unchanged':True,'final_noFSM':audit,'ranked_cost':q['ranked_cost'],'earlier39_14_cost':{'before':1996286,'after':1821983,'fraction_lower':1-1821983/1996286},'whole_cycle_prediction':'UNKNOWN','hardware_alias':'alveo_u250_firesim_gemmini_rocket_stock','hardware_config':'FireSimGemminiRocketConfig','allowed_runs':1,'requirements':['Exact released finalELF and stockbitstream identity.','Complete1000 originalrawSHA/DONE/rank0/standardadapter; unchanged exact0/0gate.','One completewholeforward observation versus2055. DistinctGSIM pairs are not additive wholeforecasts.'],'default_promotion':False,'root_initial_refusal':'Reviewer expected two fields from an ELF section helper that returns four. Refused before release; corrected structural code extraction, candidate bytes and evidence unchanged.','scope':'Typed complete finite-source affine/ReLU certificate, resource/legal ABI proof and real-D SPAD primitive lowering. Onlyone active source residual route changes. No source/model IDs or capturedvalue selectors. Externalstore/reload/panel/publication fences preserved; restricted internalSPAD fences coalesced.','pins':{str(p):sha(p) for p in (args.packet,candidate,control,reference,original,console_path,hist_path,Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'pins_reclosed':len(refs),'candidate':result['candidate_elf_sha256'],'original_words':1000,'active_entries':entries,'whole_cycles':'UNKNOWN'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    review(parser.parse_args())
