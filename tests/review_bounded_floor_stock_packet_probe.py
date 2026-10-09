"""Close measured RMS4 and freeze the next portable floor experiment."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf
from review_rms4_fast_stock_packet_probe import digest


def run(args):
    if args.output.exists() or args.terminal_review.exists():
        raise ValueError("independent reviews require fresh outputs")
    stock=json.loads(args.stock.read_text())
    q=json.loads(args.packet.read_text())
    features=json.loads(args.features.read_text())
    for receipt in (stock,q,features):
        for path,want in receipt["pins"].items():
            if digest(Path(path))!=want:
                raise ValueError("frozen artifact changed: "+path)
    record=stock["record"]
    release_path=next(Path(p) for p in stock["pins"] if Path(p).name=="root_rms4_fast_stock_packet_review_20261007.json")
    release=json.loads(release_path.read_text())
    if record["job_id"]!=2060 or record["state"]!="DONE" or record["phase"]!="DONE" or record["exit_code"]!=0 or record["elf_sha256"]!=release["candidate_elf_sha256"] or record["hw_config"]!=release["hardware_alias"]:
        raise ValueError("released stock2060 candidate/status/config differs")
    raw_path=Path(stock["raw_uart_archive"])
    if digest(raw_path)!=record["uart_sha256"]:
        raise ValueError("stock UART changed")
    console=raw_path.read_text().replace('\r','')
    measured=re.findall(r'^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$',console,re.M)
    if measured!=[str(record["kernel_cycles"])]:
        raise ValueError("actual stock ROI counter differs")
    stats=re.findall(r'^WORKSPACE_STAT (\d+) (\d+)$',console,re.M)
    if stats!=[(str(i),str(v)) for i,v in enumerate(release["uart_required"]["stats"])] or release["uart_required"]["pass_marker"] not in console or re.findall(r'^UNOBSERVED_CARRIER_DIFFERENCES (\d+)$',console,re.M)!=['3'] or 'CAPSULE UNEXPECTED SOURCE REFUSAL' in console or '*** PASSED ***' not in console:
        raise ValueError("full compiled consumer/guard/statistic/terminal gate failed")
    reduction=1-record["kernel_cycles"]/stock["control"]["cycles"]
    if reduction!=stock["cycle_reduction_fraction"]:
        raise ValueError("reported measured ratio differs")
    before_audit=audit_elf(Path(record["elf"]).read_bytes())
    if before_audit["status"]!='pass':
        raise ValueError("measured ELF custom-instruction audit failed")
    terminal={"schema":"root_rms4_stock2060_terminal_review_v1","status":"VERIFIED_COMPLETE_SOURCE_GROUP_STOCK_WIN","job_id":2060,"stock_cycles":record["kernel_cycles"],"control_job":2024,"control_cycles":stock["control"]["cycles"],"fraction_lower":reduction,"terminal_pins_reclosed":len(stock["pins"]),"actual_ROI_UART_and_full_consumer_contract_reparsed":True,"output_checksum":"NOT_EMITTED; linked complete786432i8/1024BF16 consumer comparisons pass","original_whole_native_elements":1600,"whole_hardware_cycles":"UNKNOWN","staged_elf_root_rehash_available":False,"staging_scope":"Pinned collector observations hash actual simulator-staged ELF/bitstream before teardown; root rehashes those receipts and the immutable source ELF.","hardware_identity":{k:record[k] for k in ('hw_config','hwdb_sha256','bitstream_sha256')},"nofsm":before_audit,"automatic_promotion":False,"pins":{str(p):digest(p) for p in (args.stock,release_path,raw_path,Path(__file__))}}
    args.terminal_review.write_text(json.dumps(terminal,indent=2)+'\n')
    if len(q['pins'])!=364 or len(features['pins'])!=381 or any(features['pins'].get(p)!=h for p,h in q['pins'].items()):
        raise ValueError("positive qualification/source census coverage differs")
    base=args.packet.parent.parent.parent/'out/rms4_fast_bounded_floor'
    control,candidate=base/'control/model.elf',base/'candidate/model.elf'
    if digest(control)!=record['elf_sha256'] or digest(candidate)!=q['target']['elf_sha256']:
        raise ValueError("measured current control/next candidate executable differs")
    audit=audit_elf(candidate.read_bytes())
    if audit['status']!='pass' or audit['forbidden'] or audit['unknown']:
        raise ValueError("next final executable zeroFSM gate failed")
    old_output_path=base.parent/'rms4_fast/native/output.npy'
    new_output_path=base/'native/output.npy'
    old,new=np.load(old_output_path),np.load(new_output_path)
    if old.dtype!=np.float32 or old.size!=1600 or new.dtype!=old.dtype or new.shape!=old.shape or not np.array_equal(old.view(np.uint32),new.view(np.uint32)):
        raise ValueError("next original1600 native words differ")
    old_native=json.loads((base.parent/'rms4_fast/native/validation.json').read_text())
    new_native=json.loads((base/'native/validation.json').read_text())
    if new_native['group_stats']!=old_native['group_stats'] or new_native['calls'][:3]!=[48,0,0] or new_native['product_calls']!=23040 or new_native['callback_errors']:
        raise ValueError("all48 source metadata/epoch/product coverage differs")
    stdout_path=base/'histogram/spike.stdout'
    stdout=stdout_path.read_text()
    old_stdout=(base.parent/'rms4_fast/histogram/spike.stdout').read_text()
    if re.sub(r'^WORKSPACE_GROUP_INSTRUCTIONS \d+$','ROI',stdout,flags=re.M)!=re.sub(r'^WORKSPACE_GROUP_INSTRUCTIONS \d+$','ROI',old_stdout,flags=re.M):
        raise ValueError("next compiled consumer/statistics/refusal text differs")
    strict=json.loads((base/'histogram/terminal.json').read_text())
    if strict['exit_code']!=0 or strict['elf_sha256']!=digest(candidate) or strict['stdout_sha256']!=digest(stdout_path):
        raise ValueError("next strict terminal/ELF/stdout binding failed")
    if (base.parent/'bounded_floor/probe.stdout').read_text().strip()!='BOUNDED_FLOOR_PASS checks=61440 modes=5' or not q['default_object_byte_identical']:
        raise ValueError("independent numeric/default compatibility gate differs")
    # Prospective diagnostic only. No new timing label enters any fitter.
    ratio=q['candidate_instructions']/q['control_instructions']
    result={"schema":"root_prepared_bounded_floor_stock_release_v1","status":"RELEASED_ONE_COMPLETE_GROUP_STOCK_EXPERIMENT","candidate":str(candidate),"candidate_elf_sha256":digest(candidate),"control_job":2060,"control_cycles":record['kernel_cycles'],"control_elf_sha256":digest(control),"qualification_pins_reclosed":364,"pretiming_census_pins_reclosed":381,"native_original_words_exact":1600,"native_source_groups":48,"target_independent_checks":61440,"target_rounding_modes":5,"strict_instructions_before":q['control_instructions'],"strict_instructions_after":q['candidate_instructions'],"same_device_callbacks":480,"same_logical_readback_bytes":86507520,"explicit_source_contract":q['contract'],"source_range_proof":q['source_proof'],"default_object_unchanged":True,"hardware_alias":record['hw_config'],"hardware_identity_required":{k:record[k] for k in ('hwdb_sha256','bitstream_sha256')},"uart_required":release['uart_required'],"allowed_runs":1,"whole_hardware_cycles":"UNKNOWN","conditional_pretiming_screen":{"hypothesis":"same complete device/callback workload; total retired-instruction ratio scales measured2060 groupcycles","coefficients_fitted":False,"ratio":ratio,"candidate_group_cycles":record['kernel_cycles']*ratio,"resolved_prediction":False,"unpriced":["changed FP conversions/dependence/branch distribution","cache/frame/physicalmemory and host/device overlap"],"scope":"Exploratory source-group count forecast frozen before next hardware label. It is not a physical service model, uncertainty bound, exact latency or whole prediction."},"requirements":["One same-stock immutable-ELF completegroup observation, originalconsumer/statistics/carrier3/guard PASS.","Score frozen count-ratio diagnostic withoutrefitting or replacing failures.","Preserve userwhole gate and defaultprovider; no normalwhole or automaticpromotion."],"pins":{str(p):digest(p) for p in (args.packet,args.features,args.stock,args.terminal_review,new_output_path,stdout_path,Path(__file__))}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stock2060':record['kernel_cycles'],'measured_fraction_lower':reduction,'floor_status':result['status'],'floor_instruction_fraction_lower':1-ratio,'conditional_floor_group_cycles':result['conditional_pretiming_screen']['candidate_group_cycles']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stock','packet','features','terminal-review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
