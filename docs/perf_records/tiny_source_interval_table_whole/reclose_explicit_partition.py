from pathlib import Path
import hashlib,json,subprocess,os
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import IntervalEffectContract,find_closed_scalar_i8_observers,build_source_interval_table,emit_source_interval_lookup,emit_immutable_bytes_llvm
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_lookup
from merlin.llvmlower.late_quant_rne import _tokens,_functions,_identity
from merlin.runtime.host_provider import _function_abis
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';D=W/'explicit_policy_reclosure';D.mkdir(exist_ok=False)
C=Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006');P=C/'out/artifacts/probes/source-expression-interval-promotion-20261006'
O=T.parent/'tiny-rectangular-whole-20261006/whole';LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=P/'normal_source_generic/typed_prepacket.generic.mlir';effects=IntervalEffectContract(True,True,True,True,True)
module=parse_mlir_text(source.read_text());proofs,refusals=find_closed_scalar_i8_observers(module,effects=effects);assert len(proofs)==22 and not refusals
# Partition/budget are now explicit experiment inputs, no utility defaults.
table=build_source_interval_table(proofs[0].expression,effects=effects,leading_bits=20,max_table_bytes=8388608)
assert table.data==(C/'out/artifacts/probes/source-expression-interval-table-20261006/interval_table.bin').read_bytes()
(D/'table.bin').write_bytes(table.data);data=emit_immutable_bytes_llvm(table.data,symbol='source_interval_table',alignment=64)
lookup=emit_source_interval_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation',leading_bits=20)
assert data==(W/'table.ll').read_text()==(W/'target_v2/table.ll').read_text()
assert lookup==(W/'lookup.c').read_text()==(W/'target_v2/lookup.c').read_text()
(D/'table.ll').write_text(data);(D/'lookup.c').write_text(lookup)
text=(O/'lower/model.ll').read_text();selected,report=rewrite_source_interval_lookup(text,proofs=proofs,table=table,lookup_symbol='source_lookup_activation',effects=effects)
tokens=_tokens(text);bodies=_functions(tokens);newtoken=_tokens(selected);newbody=_functions(newtoken);abis=_function_abis(text);edits=[]
for index in sorted({r['function_body_index']for r in report['routes']}):
 body=bodies[index];start=max(t.start for t in tokens if t.text=='define'and t.start<body[0].start);stop=next(t.end for t in tokens if t.text=='}'and t.start>body[-1].end)
 symbol=next(t.text for t in tokens if start<=t.start<body[0].start and t.text.startswith('@'));name=_identity(symbol)
 assert abis[name].result=='void'and abis[name].arguments==('ptr',)*5
 ns=max(t.start for t in newtoken if t.text=='define'and t.start<newbody[index][0].start);ne=next(t.end for t in newtoken if t.text=='}'and t.start>newbody[index][-1].end)
 clone=text[start:stop].replace('define internal void '+symbol,'define void @'+name+'.__source_interval_original',1)
 candidate=selected[ns:ne].replace('define internal void '+symbol,'define void @'+name+'.__source_interval_rne',1)
 edits.append((ns,ne,clone+'\n'+candidate,name))
for start,stop,replacement,name in sorted(edits,reverse=True):selected=selected[:start]+replacement+selected[stop:]
for _,_,_,name in sorted(edits):selected+='\ndeclare void @'+name+'(ptr,ptr,ptr,ptr,ptr)\n'
# Original builder declares names in source function order; edits sorted by
# source start reconstruct precisely that order, with no strategy selector.
assert selected==(W/'host_llvm/source_table.ll').read_text()==(W/'target_v2/host_llvm/source_table.ll').read_text()
assert report==json.loads((W/'host_llvm/source_binding.json').read_text())['source_bindings']==json.loads((W/'target_v2/host_llvm/source_binding.json').read_text())['source_bindings']
(D/'source_table.ll').write_text(selected);(D/'binding.json').write_text(json.dumps(report,indent=2)+'\n')
# Products are unchanged, so compile the exact original selected/linked LLVM
# as normal object identity and reconstruct the final actual link byte-exact.
link=json.loads((W/'target_v2/controlled_link.json').read_text());commands=[]
argv=list(link['candidate_compile_argv']);argv[-1]=str(D/'model.o');commands.append(argv);subprocess.run(argv,check=True,capture_output=True);assert sha(D/'model.o')==sha(W/'target_v2/model.o')
argv=list(link['candidate_link_argv']);argv=[str(D/'model.o')if arg==str(W/'target_v2/model.o')else arg for arg in argv];argv[-1]=str(D/'model.elf');commands.append(argv);subprocess.run(argv,check=True,capture_output=True);assert sha(D/'model.elf')==sha(W/'target_v2/model.elf')
(D/'model.elf').unlink();os.link(W/'target_v2/model.elf',D/'model.elf')
# Native selected bytes likewise remain authoritative; independent rebuilt
# object confirms no unnoticed compiler/context effect from API correction.
native=json.loads((W/'host/validation.json').read_text());argv=list(native['compile_argv']);argv[-1]=str(D/'native.o');commands.append(argv);subprocess.run(argv,check=True,capture_output=True);assert sha(D/'native.o')==sha(W/'host/model.o')
record=dict(schema='source_interval_explicit_partition_budget_reclosure_v1',status='pass',postcorrection_core_commit=subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip(),measured_core_snapshot=str(W/'measured_core_f53a65a69/manifest.json'),change='Remove generic20-bit/8MiB defaults. Both utilities now require caller partition; table requires storagebudget. The unchanged explicit experiment selects20bits/8388608bytes. No arithmetic/effect/IR/ABI/link change.',explicit_partition=20,explicit_storage_budget=8388608,regenerated_table_helperC_globalLLVM_selectedsourceLLVM_bindings_byteidentical=True,all22closed_observers_and44endpoint_bindings=True,target_native_object_and_whole_ELF_byteidentical=True,elf_sha256=sha(W/'target_v2/model.elf'),original_native_strict_gates_reused_by_actualidentity=True,commands=commands,pins={str(p):sha(p)for p in [Path(__file__),source,W/'measured_core_f53a65a69/manifest.json',C/'src/merlin/llvmlower/source_expression_interval.py',C/'src/merlin/llvmlower/source_expression_interval_llvm.py',W/'host_llvm/source_table.ll',W/'target_v2/host_llvm/source_table.ll',W/'host/model.o',W/'target_v2/model.o',W/'target_v2/model.elf',*D.glob('*')]if p.is_file()})
(D/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print('EXPLICIT_POLICY_TABLE_C_LLVM_OBJECT_ELF_BYTEIDENTITY_PASS',record['elf_sha256'],flush=True)
