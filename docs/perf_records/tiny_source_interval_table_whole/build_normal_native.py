"""Normal host-IR callback: typed closed observers, frozen full-model source.

This experiment owns target/host ABI mode glue only; the generic expression,
table, LLVM SSA selection and portable helper come from Merlin's source.
"""
from pathlib import Path
import hashlib,json,subprocess
from collections import Counter
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import *
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_lookup
from merlin.llvmlower.late_quant_rne import _tokens,_functions,_identity
from merlin.runtime.backends.spike_model import _transform_host_ir
from merlin.runtime.host_provider import _function_abis
from mlir_oot.late_quant_rne import merlin_host_llvm_transform

T=Path(__file__).resolve().parent
W=T/'normal_whole_v3';W.mkdir(exist_ok=False)
C=Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006')
P=C/'out/artifacts/probes/source-expression-interval-promotion-20261006'
ORIGINAL=T.parent/'tiny-rectangular-whole-20261006/whole'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(argv):
 commands.append(argv);p=subprocess.run(argv,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p
source=P/'normal_source_generic/typed_prepacket.generic.mlir'
module=parse_mlir_text(source.read_text());effects=IntervalEffectContract(True,True,True,True,True)
proofs,refusals=find_closed_scalar_i8_observers(module,effects=effects)
assert len(proofs)==22 and not refusals
table=build_source_interval_table(proofs[0].expression,effects=effects,leading_bits=20)
assert table.data==(P.parent/'source-expression-interval-table-20261006/interval_table.bin').read_bytes()
(W/'table.ll').write_text(emit_immutable_bytes_llvm(table.data,symbol='source_interval_table',alignment=64))
(W/'lookup.c').write_text(emit_source_interval_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation'))
raw=ORIGINAL/'lower/model.ll'
for name in ('source_activation','source_quantize'):
 text=(P.parent/'source-expression-interval-table-20261006'/(name+'.ll')).read_text()
 token=_tokens(text);function=_functions(token)[0]
 define=next(t for t in token if t.text=='define');opening=next(t for t in token if t.start<function[0].start and t.text=='{')
 text=text[:opening.start]+'alwaysinline '+text[opening.start:]
 (W/(name+'.ll')).write_text(text)

def transform(path,work):
 text=path.read_text();selected,report=rewrite_source_interval_lookup(text,proofs=proofs,table=table,lookup_symbol='source_lookup_activation',effects=effects)
 assert len(report['routes'])==44
 tokens=_tokens(text);bodies=_functions(tokens)
 selected_tokens=_tokens(selected);selected_bodies=_functions(selected_tokens)
 selected_functions={record['function_body_index'] for record in report['routes']}
 assert len(selected_functions)==22
 abis=_function_abis(text);edits=[];guards=[]
 for number,index in enumerate(sorted(selected_functions)):
  body=bodies[index];start=max(t.start for t in tokens if t.text=='define' and t.start<body[0].start)
  stop=next(t.end for t in tokens if t.text=='}' and t.start>body[-1].end)
  header_tokens=[t for t in tokens if start<=t.start<body[0].start]
  symbol=next(t.text for t in header_tokens if t.text.startswith('@'));name=_identity(symbol)
  abi=abis[name];assert abi.result=='void' and abi.arguments==('ptr',)*5 and not abi.variadic
  assert all(ch.isascii() and (ch.isalnum() or ch in '._') for ch in name)
  original=text[start:stop];assert original.startswith('define internal void ')
  base=name+'.__source_interval_original';candidate=name+'.__source_interval_rne'
  # Recompute selected body spans after earlier edits changed byte positions.
  selected_body=selected_bodies[index]
  newstart=max(t.start for t in selected_tokens if t.text=='define' and t.start<selected_body[0].start)
  newstop=next(t.end for t in selected_tokens if t.text=='}' and t.start>selected_body[-1].end)
  changed=selected[newstart:newstop]
  clone=original.replace('define internal void '+symbol,'define void @'+base,1)
  chosen=changed.replace('define internal void '+symbol,'define void @'+candidate,1)
  edits.append((newstart,newstop,clone+'\n'+chosen))
  guards.append({'original_symbol':name,'source_symbol':base,'candidate_symbol':candidate,'argument_count':5,'native_mode_guard':'fegetround()==FE_TONEAREST once per complete current source helper','source_body_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_body_sha256':hashlib.sha256(changed.encode()).hexdigest()})
 for start,stop,replacement in sorted(edits,reverse=True):selected=selected[:start]+replacement+selected[stop:]
 for g in guards:selected+='\ndeclare void @'+g['original_symbol']+'(ptr,ptr,ptr,ptr,ptr)\n'
 selected_path=work/'source_table.ll';selected_path.write_text(selected)
 c=['#include <fenv.h>']
 for number,g in enumerate(guards):
  args=','.join('void* a'+str(i) for i in range(5));values=','.join('a'+str(i) for i in range(5));types=','.join(['void*']*5)
  c.extend([f'extern void source_{number}({types}) __asm__("{g["source_symbol"]}");',f'extern void rne_{number}({types}) __asm__("{g["candidate_symbol"]}");',f'void guard_{number}({args}) __asm__("{g["original_symbol"]}");',f'void guard_{number}({args}){{if(fegetround()!=FE_TONEAREST)source_{number}({values});else rne_{number}({values});}}'])
 (work/'native_guard.c').write_text('\n'.join(c)+'\n')
 for csource in (W/'lookup.c',work/'native_guard.c'):
  out=work/(csource.stem+'.ll');run([str(LLVM/'clang'),'-O3','-ffp-contract=off','-fPIC','-S','-emit-llvm',str(csource),'-o',str(out)])
 linked=work/'normal_native.ll'
 inputs=[selected_path,W/'source_activation.ll',W/'source_quantize.ll',work/'lookup.ll',work/'native_guard.ll',W/'table.ll']
 run([str(LLVM/'llvm-link'),'-S',*map(str,inputs),'-o',str(linked)])
 # Ordinary OOT late source RNE callback creates the matching portable
 # companion (and target scalar RNE IR) without selecting this table strategy.
 legal=merlin_host_llvm_transform(LLVM,combine_clamp=True)(work/'normal_native.ll',work/'late_rne')
 native=work/'late_rne/model.native.ll'
 run([str(LLVM/'llvm-link'),'-S',str(native),str(B/'host_llvm/expanded_bridge.native.ll'),'-o',str(work/'expanded.native.ll')])
 # Exact ABI and all155 source-bound device-call references remain frozen.
 original_refs=Counter(t.text for t in _tokens((ORIGINAL/'host_llvm/expanded.native.ll').read_text()) if t.text.startswith('@'))
 candidate_refs=Counter(t.text for t in _tokens((work/'expanded.native.ll').read_text()) if t.text.startswith('@'))
 writer=json.loads((B/'device_host_abi/writer_contracts.json').read_text())
 bounds={'@'+n for route in writer['routes'] for n in (route['symbol'],route['borrowed_symbol'],route['symbol']+'__fresh_tensor_result')}
 assert {n:original_refs[n] for n in bounds}=={n:candidate_refs[n] for n in bounds}
 assert sum(route['calls'] for route in writer['routes'])==155
 record={'schema':'normal_source_interval_native_hook_v1','original_source_llvm_sha256':sha(path),'source_typed_census_sha256':sha(P/'normal_source_generic/census.json'),'source_bindings':report,'whole_helper_guards':guards,'physical_tables':1,'readonly_table_bytes':len(table.data),'table_sha256':table.sha256,'all_original_device155_references_conserved':True,'explicit_effects':vars(effects),'normal_hook':'_transform_host_ir on immutable normal current2004 LLVM; fresh typed22observer/compilerbinding and selected object. Existing full upstream pipeline receipt/source stays frozen. No fresh wholeupstream recapture/lowering claim.','objects':'table bytes included in normal hostIR/model.o, no hostprovider or kernelcatalog relaxation','runtime_mode':'Native fegetround complete-helper fallback; target guard is separately OOT and not built here','commands':commands,'pins':{str(p):sha(p) for p in [source,raw,ORIGINAL/'host_llvm/expanded.native.ll',B/'host_llvm/expanded_bridge.native.ll',B/'device_host_abi/writer_contracts.json',W/'table.ll',W/'lookup.c',W/'source_activation.ll',W/'source_quantize.ll',*work.glob('*'),*work.glob('late_rne/*'),C/'src/merlin/llvmlower/source_expression_interval.py',C/'src/merlin/llvmlower/source_expression_interval_llvm.py',Path(__file__),LLVM/'clang',LLVM/'llvm-link',LLVM/'opt'] if p.is_file()}}
 (work/'source_binding.json').write_text(json.dumps(record,indent=2)+'\n')
 return work/'expanded.native.ll'

selected,hookreceipt=_transform_host_ir(raw,W/'host_llvm',transform)
(W/'normal_hook_receipt.json').write_text(json.dumps(hookreceipt,indent=2)+'\n')
print('NORMAL_SOURCE_TABLE_22HELPERS_44LANES_NATIVE_IR_PASS',sha(selected),flush=True)
