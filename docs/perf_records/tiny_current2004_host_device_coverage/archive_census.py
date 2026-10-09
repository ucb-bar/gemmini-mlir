from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import collections

W = Path(__file__).resolve().parent
O = W.parents[3]
C = Path('/scratch/agustin/tmp/merlin-scalar-contraction-rectangular-20261006')
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
d=read(W/'source_census.json');caps=read(W/'existing_capabilities.json')
graph={o['id']:o for o in d['tensor_graph']}
assert len(d['norms'])==45
assert all(len(n['output_use_closure']['boundaries'])==1 and n['output_use_closure']['boundaries'][0]['results'][0]['dtype']=='i8' for n in d['norms'])
assert all(len(n['output_use_closure']['ids'])==8 for n in d['norms'])
assert collections.Counter(tuple(n['input_producer']['body'])for n in d['norms'])==collections.Counter({('arith.index_cast','linalg.index','tensor.extract','linalg.yield'):1,('arith.addf','linalg.yield'):44})
quant_norm=[]
for n in d['norms']:
    q=n['output_use_closure']['boundaries'][0]
    scale=graph[q['producers'][1]]
    assert scale['body']==['arith.constant','arith.divf','linalg.yield']
    # The scale is independently derived; it is absent from the norm live-use closure.
    assert scale['id'] not in n['output_use_closure']['ids']
    quant_norm.append(dict(reduction=n['reduction']['id'],input=n['input_producer']['id'],
                           normalized_to_i8_closed_use_ids=n['output_use_closure']['ids'],
                           i8_observer=q['id'], scale_producer=scale['id'],
                           source_scale_is_independent_of_norm=True,
                           input_still_live_as_residual=len(n['input_uses'])==3,
                           direct_existing_i8_representation=False))
pre_down=[]
for o in graph.values():
    if o['body']!=['arith.constant','arith.negf','math.exp','arith.addf','arith.divf','linalg.yield']:
        continue
    pending=[o['id']];seen=[];observers=[]
    while pending:
        i=pending.pop()
        assert i not in seen
        seen.append(i);q=graph[i]
        if q['results'][0].get('dtype')=='i8':observers.append(i)
        else:
            uses=[u['op']for group in q['result_uses']for u in group]
            assert len(uses)==1
            pending+=uses
    assert len(seen)==4 and len(observers)==1
    pre_down.append(dict(nonlinear=o['id'],closed_use_ids=seen,i8_observer=observers[0],elements=o['results'][0]['elements']))
assert len(pre_down)==22 and sum(x['elements']for x in pre_down)==991232
assert len(caps['direct_i32_consumer_match_results'])==155 and caps['direct_requant_matches']==0
families=d['live_families']
copies=[f for f in families if f['name']=='linalg.transpose' or (f['name']=='linalg.generic' and f['body']==['linalg.yield'])]
files={p for p in W.iterdir()if p.is_file()}
files.update([Path(d['source_path']),Path(caps['source_path']),C/'src/merlin/llvmlower/scalar_squared_sum.py',C/'src/merlin/llvmlower/scalar_contraction.py',C/'src/merlin/llvmlower/broadcast_math_hoist.py',C/'src/merlin/llvmlower/pipeline.py',C/'src/merlin/llvmlower/scalar_pointwise_packet.py'])
files.update(O/'mlir_oot'/p for p in ['contraction_patterns.py','golden_requant.py','golden_resadd.py','golden_gemm.py','golden_device_profile.py'])
prior=O/'docs/perf_records/tiny_rectangular_contraction_whole_qualification.json';files.add(prior)
profile=O/'docs/perf_records/tiny_current2004_device_profile_qualification.json';files.add(profile)
record=dict(schema='compiler_optimization_journey_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),
    hypothesis='Identify actual typed host regions with an existing legal integer device representation and closed observers before adding acceleration. Prove available producer dtype/use conditions rather than selecting a model/shape or interpreting captured float words as integer operands.',
    ownership='Typed source/use/numerical/ownership/frontier proofs and generic preparation Merlin; target primitive legality, layouts/resources and ISA lowering OOT.',
    emitted_change='NONE. Read-only complete source graph, conservative effectful-call liveness and existing target matcher census. Source uses bind obligations only; no model/name/ordinal strategy selectors.',
    whole_control=dict(stock_job=2004,stock_cycles=422018733,source_and_object_closure_path=str(prior),profile_qualification_path=str(profile),current_hw_split=None,current_section_cycles=None),
    coverage=[
      dict(family='existing integer projections',calls=155,source_MACs=8275361792,types='i8×i8→i32',route='Already device-offloaded. Current objects remain immutable; CPU code in device spans includes command/address/control/fence/wait.',missing='Physical DMA/mesh/overlap attribution remains unknown; current155-boundary stock profile qualified but not yet timed.'),
      dict(family='attention QK/PV',calls=44,source_rounded_multiply_add_pairs=5767168,additional_outer_pairs=256,types='f32×f32→f32',route='Host exact ordered2×4/K2 schedule. Existing stock Gemmini has no nativef32 datapath.',missing='Requires an explicit encoded-value/error certificate and complete live consumer DAG, with original source rounding/softmax/residual observations; no dtype override or direct i8 match.'),
      dict(family='squared sum and norm',calls=45,rows=360,source_squared_elements=737280,types='f32 residual/embedding→f32 norm→fixed-scale i8',route='Current exact scalar SQS and per-row rsqrt hoist. All45 normalized outputs close to one i8 quantizer; its scalar scale is independent of norm.',missing='ZERO inputs have a direct exact existing i8 representation:1embedding gather,44rounded f32 residual adds.44norm inputs remain live residuals. Integer Gram of another quantized tensor does not compute this source norm; new exact/enclosed encoding and selective quant-bin proof required.'),
      dict(family='post gate/up nonlinearity and quantization',calls=22,elements=991232,types='i32 projections→f32 scales/nonlinear/products→i8',route='Host fused polynomial/FMA and exact original quant observer; generic pureMultiply4/twoMultiply4 policies.',missing='Promising closed i8 frontier from integer producers, but existing uniform-constant store-scale matcher has no per-channel scale/nonlinear capability. Must bind current actual polynomial/source rounding and prove every observed i8 word or explicit original policy.'),
      dict(family='residual adds',calls=44,elements=720896,types='rounded f32 residual plus scaled i32 projection→f32',route='Host exact rounded dequant/add chain; floating result lives across layers.',missing='Existing i8 identity-matmul+D residual requires i8 operands; no equivalent source representation. Cannot discard live f32 residual or replace it with downstream i8.'),
      dict(family='i32 dequant epilogues',calls=155,types='i32→f32→two rounded scales (one per channel)',route='Host source arithmetic, already scheduled where eligible. All155 existing exact direct requant matcher attempts refuse four-operand f32 result geometry.',missing='Per-channel scales and live floating outputs differ from uniform scalar i32→i8 store contract. The provider must retain each source cast/multiply and live residual/nonlinear use or certify a closed i8 frontier.'),
      dict(family='live layout copies/transposes',source_families=copies,route='Generic source copies/views, then authoritative upstream bufferization; no claim every source transpose materializes.',missing='Need actual buffer alias/lifetime/stride and emitted traffic witness. Reversible immutable parameter packing is legal but measured packed RHS and stationary transpose were negative; do not infer copy savings from source counts.'),
    ],
    rms_quant_frontiers=quant_norm,pre_down_quant_frontiers=pre_down,
    integer_Gram_candidate_cost_scope=dict(original_source_shape=[8,2048],logical_Gram=[8,8],required_diagonal=8,full_Gram_MACs=131072,source_square_products=16384,logical_extra_factor=8,padded16x16_MAC_capacity=524288,padded_vs_source_factor=32,full_i8_diagonal_prefix_bound=33554432,int32_bound_ok=True,actual_input_encoding_available=False,actual_cycles=None),
    division_source_census=dict(sigmoid_like=991232,softmax=45056,mean=360,scalar_quant_reciprocals=89,total=1036737,scope='Logical source arithmetic count, not current PC instructions/cycles; fused current polynomial source lowering separately bound by immutable normal recipe.'),
    dead_source_operations_scope='Conservative backward source liveness roots all ordinary calls/returns and captured structured-body inputs. Unused float parameter dequant/transposes are excluded. This is source dependency evidence, not measured allocation/copy/traffic.',
    result='Coverage holes identified, no new arithmetic/offload enabled. Prioritize source-bound closed i8 frontier or explicit preparation proof; current whole section timing waits currentstockprofile.',
    actual_hardware_change=None,whole_cycle_projection=None,token_usage_available=False,
    pins={str(p):sha(p)for p in sorted(files)})
dest=O/'docs/perf_records/tiny_current2004_host_device_coverage';dest.mkdir(exist_ok=False)
for name in ['census_source.py','capability_match.py','capability_match.log','existing_capabilities.json','archive_census.py']:
    shutil.copyfile(W/name,dest/name)
(dest/'live_source_families.json').write_text(json.dumps(families,indent=2)+'\n')
(O/'docs/perf_records/tiny_current2004_host_device_coverage.json').write_text(json.dumps(record,indent=2)+'\n')
print('CURRENT_SOURCE_COVERAGE_CLOSED',len(record['pins']),len(quant_norm),len(pre_down),flush=True)
