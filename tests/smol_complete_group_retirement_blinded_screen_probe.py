"""Freeze historical complete-group fits without accessing the new stock label."""

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.jsonio import canonical_sha256
from merlin.perf import fast_estimate_validation as fast
from mlir_oot.no_fsm_audit import audit_elf


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def freeze(args):
    if args.output.exists():
        raise ValueError('fresh blinded forecast required')
    index=json.loads(args.index.read_text())
    for p,h in index['pins'].items():
        if sha(p)!=h:
            raise ValueError('indexed evidence changed: '+p)
    rows=index['rows']
    if [r['job'] for r in rows]!=[2001,2003,2008,2024,2054,2060,2064]:
        raise ValueError('historical cohort changed')
    held=rows[-1]
    if held['stock_cycles'] is not None or held['terminal_record'] is not None:
        raise ValueError('new label was unblinded')
    if any(r['callback_count']!=480 or r['readback_bytes']!=86507520 or not r['consumer_pass'] for r in rows):
        raise ValueError('same complete consumer/request scope required')
    audits={}
    for row in rows:
        if sha(row['elf'])!=row['elf_sha256']:
            raise ValueError('indexed executable changed')
        audit=audit_elf(Path(row['elf']).read_bytes())
        if audit['status']!='pass':
            raise ValueError('noFSM audit failed')
        audits[str(row['job'])]=audit
    domain=canonical_sha256({'hardware':index['hardware'],'scope':index['common_scope'],'pinned_index':sha(args.index),'model_kind':'historical_same_group_total_retirement_diagnostic'})
    workload=canonical_sha256({'scope':index['common_scope'],'source_family':'one_original_closed_attention_group','callbacks':480,'readback_bytes':86507520})
    pointer='/features/strict_roi_retired_instructions'
    evidence=tuple(sorted(set(index['pins'].values())|{sha(args.index)}))
    training=[fast.Observation(r['elf_sha256'],workload,'training_same_source_group',domain,{pointer:r['strict_roi_retired_instructions']},r['stock_cycles'],evidence) for r in rows[:-1]]
    forecasts=[]
    for include_fixed in (False,True):
        name='fixed_plus_retired' if include_fixed else 'retired_only'
        try:
            model=fast.fit_linear_screen(training,pointers=(pointer,),include_fixed=include_fixed,maximum_condition=1000)
        except ValueError as error:
            forecasts.append({'name':name,'refused':str(error),'new_label_accessed':False})
            continue
        values={pointer:held['strict_roi_retired_instructions']}
        prediction=model.predict(values,domain_sha256=domain)
        conditional=model.fixed_cycles+model.coefficients[0]*values[pointer]
        calibration=[{'job':r['job'],'actual_cycles':r['stock_cycles'],'prediction':model.fixed_cycles+model.coefficients[0]*r['strict_roi_retired_instructions']} for r in rows[:-1]]
        forecasts.append({'name':name,'fit_sha256':model.provenance_sha256,'coefficients':model.coefficients,'fixed_cycles':model.fixed_cycles,'training_jobs':[r['job'] for r in rows[:-1]],'training_feature_range':model.domains,'training_residuals':calibration,'held_job':2064,'held_feature':values,'domain_prediction':prediction.to_dict(),'conditional_extrapolation_cycles':conditional,'new_label_accessed':False,'all_source_variants_same_workload':True,'held_workload_ranking_qualified':False,'interpretation':'Empirical total complete-group relationship. This cohort does not identify CPU rates, callback/DDR service or a workload-general model; new lower-retirement extent remains outside training range.'})
    release=json.loads(args.release.read_text())
    if release['candidate_elf_sha256']!=held['elf_sha256']:
        raise ValueError('existing released successor differs')
    result={'schema':'root_smol_complete_group_blinded_retirement_screen_v1','status':'FROZEN_DIAGNOSTIC_BEFORE_NEW_LABEL_ACCESS','index_pins_reclosed':len(index['pins']),'index_sha256':sha(args.index),'hardware':index['hardware'],'source_scope':index['common_scope'],'held_job':2064,'held_elf_sha256':held['elf_sha256'],'new_hardware_label':'UNKNOWN; never opened by this driver','historical_training_jobs':[r['job'] for r in rows[:-1]],'forecasts':forecasts,'existing_prequeue_count_ratio_forecast_unchanged':release['conditional_pretiming_screen'],'noFSM':audits,'source_variants_are_not_independent_workloads':True,'automatic_export':False,'whole_prediction':'UNKNOWN','unknowns':index['unknowns']+['Changed FP/GPR dependence and linked code/cache regime','No calibrated uncertainty; one observation per historical arm','Historical single-workload fitting does not qualify unseen source families or held workload ranking'],'pins':{str(p):sha(p) for p in (args.index,args.release,Path(fast.__file__),Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'reclosed':result['index_pins_reclosed'],'forecasts':[{k:f.get(k) for k in ['name','conditional_extrapolation_cycles','refused']} for f in forecasts],'held_stock_label_accessed':False,'automatic_export':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','release','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    freeze(parser.parse_args())
