"""Read-only source/ELF admission for a user-authorized complete hardware trial."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

OOT = Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007')
CORE = Path('/scratch/agustin/tmp/merlin-host-llvm-helper-abi-20261007')
sys.path[:0] = [str(OOT), str(CORE/'src')]
from mlir_oot.no_fsm_audit import audit_elf
from merlin.runtime.backends.spike_model import SpikeModelError, parse_console
from merlin.runtime.output_digest import verify_output_sha256

SOURCE_ROOT = Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007')
SOURCE_PACKET = SOURCE_ROOT/'docs/perf_records/prepared_endpoint_dag_normal_whole_qualification.json'
BUILD = SOURCE_ROOT/'out/artifacts/probes/prepared-endpoint-normal/build_v3'
ELF = BUILD/'model.elf'
OUT = Path(__file__).parent
EXPECTED_ELF = 'f90b8c7e8d6c1ed668a0bb4eaf7918caaa2dc8e3fe4f93b6c14143f995889f07'
GOLDEN = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_silufix_flash_bundle/golden.npy')
HISTORICAL = Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/docs/perf_records/smol1906_stock_hardware.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def verify_protocol(text, reference):
    text = text.replace('\r','')
    if any(marker in text for marker in ('FATAL:', 'FAIL ', 'FAIL:', 'COMMAND_EXIT_CODE="1"')):
        raise ValueError('target reported failure')
    parsed = parse_console(text)
    digest = verify_output_sha256(text, reference)
    expected_prefix = reference.reshape(-1)[:1].astype('<f4').view('<u4')
    if parsed['output_dtype']!='f32' or not np.array_equal(parsed['raw_output_bits'], expected_prefix):
        raise ValueError('original raw output prefix differs')
    metrics = parsed['metrics']
    if metrics.get('memref_rank_mismatch')!=0 or not isinstance(metrics.get('cycles'),int) or metrics['cycles']<=0:
        raise ValueError('missing positive complete cycles/zero rank mismatch')
    if metrics.get('build_hash')!='b998822b59c3':
        raise ValueError('unexpected secondary build marker')
    if not np.array_equal(parsed['argmax'],np.argmax(reference.reshape(50,32),axis=1)):
        raise ValueError('original output row argmax differs')
    return {'full_output':digest,'raw_prefix_words':parsed['raw_output_bits'].tolist(),
        'cycles':metrics['cycles'],'rank_mismatches':0,
        'scope':'complete model reset/prepare/run/commit mcycle window; hashing, UART and simulator startup excluded'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source = json.loads(SOURCE_PACKET.read_text())
    pins = source['pins']
    def verify(item):
        path,digest=item
        if not Path(path).is_file() or sha(path)!=digest: raise ValueError('stale source packet input: '+path)
        return path
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(verify,pins.items()))
    assert len(pins)==191 and sha(ELF)==EXPECTED_ELF
    assert source['native']['elements']==1600 and source['native']['bitwise_mismatches']==0
    assert source['ordinary_routed_contractions']==297 and source['ordinary_signatures']==19
    reference = np.load(GOLDEN)
    assert reference.dtype==np.float32 and reference.size==1600 and reference.shape==(1,50,32)
    raw = reference.astype('<f4',copy=False).tobytes(order='C')
    expected_digest = hashlib.sha256(raw).hexdigest()
    assert expected_digest=='1e5b274d4c99cb6710994542f509769ba61ed37f435dc6ee1ed05ed1b8cc03de'
    recipe_path = BUILD/'compilation_recipe.json'
    recipe = json.loads(recipe_path.read_text())
    assert recipe['status']=='completed' and recipe['executable']['sha256']==EXPECTED_ELF
    for command in recipe['commands']:
        assert command['status']=='returned'
        for item in [command['executable'],*command['inputs'],command['output']]:
            assert sha(item['path'])==item['sha256'],item['path']
            pins[item['path']]=item['sha256']
    with ELF.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        audit = audit_elf(data)
        assert audit['status']=='pass' and audit['elf_sha256']==EXPECTED_ELF
        phoff = struct.unpack_from('<Q',data,32)[0]
        entsize,count = struct.unpack_from('<HH',data,54)
        assert entsize==56
        loads=[]
        for i in range(count):
            kind,flags,off,va,pa,fs,ms,align = struct.unpack_from('<IIQQQQQQ',data,phoff+i*entsize)
            if kind==1:
                assert va==pa and 0<fs<=ms and off+fs<=len(data)
                loads.append({'start':va,'end':va+ms,'file_bytes':fs,'memory_bytes':ms,'flags':flags,'alignment':align})
    audit_path=OUT/'final_nofsm.json';audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    gcc = Path(recipe['commands'][-1]['argv'][0])
    nm = gcc.with_name('riscv64-unknown-elf-nm')
    nm_text=subprocess.check_output([str(nm),'-n',str(ELF)],text=True)
    symbols={}
    for line in nm_text.splitlines():
        parts=line.split()
        if len(parts)==3 and parts[2] in {'_bss_start','_bss_end','_stack_top','_binary_weights_bin_start','_binary_weights_bin_end','MERLIN_STACK_BYTES','MERLIN_WEIGHTS_BASE'}:
            symbols[parts[2]]=int(parts[0],16)
    malloc = next(command for command in recipe['commands'] if Path(command['requested_output']).name=='malloc.o')
    defines={}
    for argument in malloc['argv']:
        if argument.startswith('-DMERLIN_ARENA_'):
            name,value=argument[2:].split('=',1);defines[name]=int(value.removesuffix('ULL'),0)
    arena_start=defines['MERLIN_ARENA_BASE_ADDR'];arena_bytes=defines['MERLIN_ARENA_SIZE_BYTES']
    assert arena_start==0xa2100000 and arena_bytes==8*1024**3
    assert symbols['MERLIN_STACK_BYTES']==16*1024**2
    stack_end=symbols['_stack_top'];stack_start=stack_end-symbols['MERLIN_STACK_BYTES']
    intervals=[('static_load_'+str(i),row['start'],row['end']) for i,row in enumerate(loads)]
    intervals.extend([('stack',stack_start,stack_end),('arena',arena_start,arena_start+arena_bytes)])
    for i,(left_name,left,right) in enumerate(intervals):
        assert 0<=left<right<2**64
        for right_name,start,end in intervals[i+1:]: assert right<=start or end<=left,(left_name,right_name)
    assert symbols['_binary_weights_bin_end']==0xa2090120
    # These prove address extents, not an unsealed loaded hardware capacity.
    historical=json.loads(HISTORICAL.read_text())
    timeout=14400
    assert historical['queue_started_to_ended_seconds']<timeout
    prefix=int(reference.reshape(-1).view(np.uint32)[0])
    expected_frame=('OUT 1 '+str(prefix)+'\nARGMAX 50 '+' '.join(map(str,np.argmax(reference.reshape(50,32),axis=1)))+
        '\nOUT_SHA256 f32le 1600 6400 '+expected_digest+
        '\nMETRIC cycles 1\nMETRIC build_hash b998822b59c3\nMETRIC memref_rank_mismatch 0\nDONE\n')
    verify_protocol(expected_frame,reference)
    rejected=0
    for mutation in (expected_frame.replace(expected_digest,'0'*64),expected_frame+'METRIC cycles 2\n',
                     expected_frame.replace('OUT 1 '+str(prefix),'OUT 2 '+str(prefix)),expected_frame.replace('DONE\n','')):
        try: verify_protocol(mutation,reference)
        except (ValueError,SpikeModelError): rejected+=1
        else: raise AssertionError('malformed terminal protocol admitted')
    frames=OUT/'protocol_admission_tests.json'
    frames.write_text(json.dumps({'status':'parser-only synthetic protocol checks, no target evidence','positive':1,'negative':rejected},indent=2)+'\n')
    memory_source=SOURCE_ROOT/'out/artifacts/probes/prepared-endpoint-normal/stock_memory_binding.json'
    for path in [SOURCE_PACKET,HISTORICAL,GOLDEN,recipe_path,BUILD/'cgen/model_gen.h',BUILD/'cgen/model_io.h',
        memory_source,Path(__file__),audit_path,frames,nm,CORE/'src/merlin/runtime/output_digest.py',
        CORE/'src/merlin/runtime/backends/spike_model.py',OOT/'mlir_oot/no_fsm_audit.py',OOT/'mlir_oot/tables/rtl_facts.py']:
        pins[str(path)]=sha(path)
    record={'schema':'smol_endpoint_full_hardware_evaluation_admission_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'READY_FOR_PARENT_AUTHORIZED_EMPIRICAL_STOCK_TRIAL_NOT_SUBMITTED',
        'elf':identity(ELF),'source_packet':identity(SOURCE_PACKET),'original_source_pins_reclosed':191,
        'normal_native':{'elements':1600,'bitwise_mismatches':0,'scope':source['native']['scope'],'native_record':source['native']},
        'numeric_policy':source['numerical_policy'],'whole_accuracy_policy':'original elementwise gate remains unchanged; matching original all1600 f32le digest provides stricter exact fixture evidence',
        'whole_target_fidelity':'UNMEASURED for this ELF; bounded root Spike run pending; predecessor gate not transferred',
        'whole_hardware_cycles':'UNKNOWN until complete trial','ordinary_contractions':297,'ordinary_signatures':19,
        'layout':{'load_regions':loads,'symbol_values':symbols,'stack':[stack_start,stack_end],
            'arena':[arena_start,arena_start+arena_bytes],'arena_bytes':arena_bytes,'regions_disjoint':True,
            'loaded_weights_bytes':symbols['_binary_weights_bin_end']-symbols['_binary_weights_bin_start'],
            'capacity_status':'UNKNOWN: current34-bit AXI header does not prove loaded bitstream physical capacity',
            'empirical_evaluation_authorized':True,'peak_allocation_usage':'UNKNOWN, bump free is no-op; exhaustion marker must fail evaluation'},
        'protocol':{'parser_file':str(Path(__file__)),'parser_function':'verify_protocol',
            'shared_parser':'merlin.runtime.backends.spike_model.parse_console',
            'shared_full_digest':'merlin.runtime.output_digest.verify_output_sha256','golden':str(GOLDEN),
            'encoding':'f32le','elements':1600,'bytes':6400,'expected_sha256':expected_digest,
            'dump_cap':1,'secondary_build_marker':'b998822b59c3','secondary_marker_is_unique_elf_identity':False,
            'completion_markers':['DONE','METRIC memref_rank_mismatch 0'],
            'root_must_also_require':['terminalDONE/exit0','simulatorPASSED','COMMAND_EXIT_CODE="0"','actualstagedELFsha','actualstagedbitsha','actualexecuteddriversha']},
        'execution_bound':{'seconds':timeout,'historical1906':identity(HISTORICAL),
            'historical_queue_seconds':historical['queue_started_to_ended_seconds'],
            'historical_engine_seconds':historical['engine_elapsed_seconds'],
            'scope':'bounded operational timeout, not a cycle forecast; queue setup/startup distinct from model timer'},
        'hardware_observation_required':{'alias':'alveo_u250_firesim_gemmini_rocket_stock',
            'live_before_teardown':['ELF','firesim.bit','actual driver executable and current driver bundle','UART','runworkload/current runtime config'],
            'historical_header_bit_build_equivalence':'UNKNOWN; retain limit rather than blocking authorized empirical trial'},
        'offline_model_scope':'no group×48 projection; actual current source/callback inventory must be joined before section prices or profiler claims',
        'no_fsm':audit,'pins':pins,'token_usage_available':False}
    path=OUT/'admission.json';path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'receipt':identity(path),'pins':len(pins),'arena_end':hex(arena_start+arena_bytes),
        'source_native_elements':1600,'timeout_seconds':timeout,'whole_cycles':'UNKNOWN'}))


if __name__=='__main__':main()
