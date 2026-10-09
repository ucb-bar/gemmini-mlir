from pathlib import Path
import json,hashlib
from xdsl.dialects import func
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_group_writer import ClosedGroupWriterContract
from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract
from merlin.llvmlower.consumer_observed_group_writer import ConsumerObservedGroupPreparation,ConsumerObservationContract
from merlin.llvmlower.prepared_operand_owner import PreparedRepresentation,PreparedStorageSpan
from merlin.llvmlower.prepared_operand_schedule import plan_prepared_borrows
from merlin.llvmlower.tensor_preparation_identity import canonical_tensor_read_view,TensorPreparationRequest,find_tensor_preparation_opportunities
w=Path('/scratch/agustin/tmp/gemmini-prepared-rhs-owner-20261006/out/prepared_rhs_normal_materialized')
OUT=Path(__file__).resolve().parents[2]/'out/normal_composition/source';OUT.mkdir(parents=True,exist_ok=False)
prior=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/bounded_native/build/device_prepared/source_control/source_group_calls.json')
r=json.loads(prior.read_text());p=Path(r['selected_path']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(p)==r['selected_sha256']
m=parse_mlir_text(p.read_text());records={x['symbol']for x in r['records']}
calls=[x for x in m.walk()if isinstance(x,func.CallOp)and x.callee.root_reference.data in records]
# This index map is the already qualified complete 11-input source grammar;
# the generic planner itself selects neither names, ordinals nor tensor values.
slots=(1,3,4,5,6,8,9,10)
format=dict(source_contract='complete_11_input_Q_K_mask_V_source',slots=slots,physical='head,entry,plane,k,column; source/reconstructed rows,k',radix=128,digits=3,step_dtype='binary64',equality='source_and_reconstruction',payload_bytes=30048912)
fmt=hashlib.sha256(json.dumps(format,sort_keys=True).encode()).hexdigest()
groups={}
for call in calls:
 key=tuple(canonical_tensor_read_view(call.arguments[i]).identity for i in slots)
 groups.setdefault(key,[]).append(call)
bundles=[]
for group in groups.values():
 bundle=[]
 for slot in slots:
  opportunities=find_tensor_preparation_opportunities([TensorPreparationRequest(call.arguments[slot],call,fmt)for call in group]);assert len(opportunities)==1;bundle.extend(opportunities)
 bundles.append(tuple(bundle))
rep=PreparedRepresentation(fmt,'2'*64,'3'*64,(PreparedStorageSpan('rhs_payload',30048912,64),))
plan=plan_prepared_borrows(m,bundles,rep)

from merlin.llvmlower.prepared_operand_schedule import install_prepared_borrows
from merlin.xdsl_dialects._common import text

from merlin.llvmlower.closed_group_writer import install_closed_group_writers, source_function_semantic_sha256
from merlin.llvmlower.private_workspace_pool import pool_private_writer_workspaces
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy
from merlin.llvmlower.prepared_attention_rhs import prepare_attention_rhs_owner
import ctypes
provider=OUT.parent/'provider'
build=json.loads((provider/'build.json').read_text())
for path,pin in build['pins'].items():
 if sha(path)!=pin:raise ValueError('provider input changed: '+path)
policy=ApproximateSourceRoundoffPolicy(*([True]*8));policy.validate()
lib=ctypes.CDLL(str(provider/'native_numeric/provider.so'))
for name in ('group_provider_workspace_bytes','group_provider_rhs_owner_bytes'):
 getattr(lib,name).restype=ctypes.c_size_t
workspace=lib.group_provider_workspace_bytes();payload=lib.group_provider_rhs_owner_bytes()
bridge=w/'bridge.c';identity=sha(bridge)
permission=dict(policy=policy.numerical_policy,approximation_explicit=True,original_output_gate={'atol':0.03125,'rtol':0.02},validation='pending; no exact observer theorem',provider=sha(provider/'target_numeric/provider.c'))
proofpin=hashlib.sha256(json.dumps(permission,sort_keys=True).encode()).hexdigest()
rep=PreparedRepresentation(fmt,proofpin,identity,(PreparedStorageSpan('rhs_payload',payload,64),))
plan=plan_prepared_borrows(m,bundles,rep)
functions={f.sym_name.data:f for f in m.body.block.ops if isinstance(f,func.FuncOp)}
contracts={}
for record in r['records']:
 symbol=record['symbol']
 contracts[symbol]=ClosedGroupWriterContract(source_function_semantic_sha256(functions[symbol]),'attention_frontier_writer',build['roles']['target_numeric']['source_sha256'],proofpin,policy.numerical_policy,identity,64,True,True,True,True,'rne_returned_values','explicit experimental RMS4 normal binding; original whole validation pending',private_workspaces=(PrivateWorkspaceContract(workspace,64,identity,True,True,True),),source_fallback_symbol='source_attention_frontier_fallback')
installed=install_closed_group_writers(m,r,contracts)
pool=pool_private_writer_workspaces(m,installed['fresh_writer_report'])
def physical(plan,wrapper,borrowed,calls):
 baseline=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate/target_numeric/provider.c')
 expected=prepare_attention_rhs_owner(baseline.read_text().replace('group_provider','@SYMBOL@')).replace('@SYMBOL@','group_provider')
 if (provider/'target_numeric/provider.c').read_text()!=expected:raise ValueError('physical owner emitter mismatch')
 if plan.capacity!=payload+len(bundles) or plan.alignment!=64:raise ValueError('physical owner capacity mismatch')
 if sha(bridge)!=identity:raise ValueError('bridge changed')
 if len(tuple(borrowed.function_type.inputs))!=13:raise ValueError('unexpected borrowed ABI')
placement=install_prepared_borrows(m,plan,installed['fresh_writer_report'],prepare_symbol='prepare_attention_rhs',borrowed_symbol='attention_frontier_prepared_borrowed',validate_physical=physical,materialize_shared_views=True)
m.verify()
selected=OUT/'normal_prepared.mlir';selected.write_text(text(m))
(OUT/'preparation.json').write_text(json.dumps(dict(policy=permission,installed=installed,pool=pool,placement=placement,source_sha256=sha(p),selected_sha256=sha(selected),workspace_bytes=workspace,provider_manifest=sha(provider/'build.json'),normal_full48='PENDING'),indent=2)+'\n')
print('EXPLICIT_APPROXIMATE_NORMAL_PREPARED',placement,flush=True)
