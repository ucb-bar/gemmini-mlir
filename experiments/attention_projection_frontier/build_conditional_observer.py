"""Native semantic-screen hook at the actual closed BF16 projection output."""
from pathlib import Path
import hashlib,json,subprocess
from xdsl.dialects import func,tensor
from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,StringAttr,UnitAttr,bf16
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import BlockArgument
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract,rewrite_fresh_tensor_writers
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.xdsl_dialects._common import text
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/conditional_native_v2';w.mkdir(parents=True,exist_ok=False)
p=base/'out/normal_composition/build_v6/lower/model.mlir';m=parse_mlir_text(p.read_text());forward=next(o for o in m.body.block.ops if isinstance(o,func.FuncOp) and o.sym_name.data=='forward')
# Diagnostic source-instance selection. Production numeric policy never selects by provenance.
selected=[o for o in forward.body.block.ops if isinstance(o,GenericOp) and 'self_attn.out_proj' in str(o.attributes.get('prov.fqn','')) and str(o.attributes.get('prov.aten',''))=='"aten.add.Tensor"' and o.results[0].type.element_type==bf16 and isinstance(o.inputs[1],BlockArgument)]
assert len(selected)==12
kind=selected[0].results[0].type;assert all(o.results[0].type==kind for o in selected)
name='conditional_projection_observer';borrowed='conditional_projection_observer_borrowed'
decl=func.FuncOp.external(name,[kind,kind],[kind]);decl.attributes['llvm.emit_c_interface']=UnitAttr();decl.properties['arg_attrs']=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(s)}) for s in ('read','write')]);m.body.block.add_op(decl)
records=[]
for index,o in enumerate(selected):
 old=o.results[0];uses=list(old.uses);empty=tensor.EmptyOp([],kind);call=func.CallOp(name,[old,empty.tensor],[kind]);forward.body.block.insert_ops_after([empty,call],o)
 for use in uses:use.operation.operands[use.index]=call.results[0]
 records.append(dict(index=index,source_operation_sha256=hashlib.sha256(text(o).encode()).hexdigest(),type=str(kind)))
report=rewrite_fresh_tensor_writers(m,[FreshTensorWriterContract(name,1,(1,),borrowed,64)])
selected_source=text(m);(w/'selected.mlir').write_text(selected_source)
(w/'source_binding.json').write_text(json.dumps(dict(scope='Diagnostic full-write native observer hook, original source still computes continuation before call; no avoided-work/performance claim.',source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bindings=records,writer=report,policy='conditional_bf16_row_max_spacing_v1'),indent=2)+'\n')
ll=lower_to_llvm_ir(selected_source,workdir=w/'lower',features=frozenset({'respect_captured_quantization_scope','lower_fma_to_intrinsic','outline_llvm_loops'}));(w/'model.ll').write_text(ll)
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang';cmd=[clang,'-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(w/'model.ll'),'-o',str(w/'model.o')];subprocess.run(cmd,check=True)
(w/'compile.json').write_text(json.dumps(dict(command=cmd,llvm_sha256=hashlib.sha256(ll.encode()).hexdigest(),object_sha256=hashlib.sha256((w/'model.o').read_bytes()).hexdigest()),indent=2)+'\n');print('CONDITIONAL_OBSERVER_COMPILED',flush=True)
