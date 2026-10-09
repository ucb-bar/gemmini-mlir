from pathlib import Path
import ctypes as C
import hashlib,json,re,shlex,shutil,struct,subprocess
from dataclasses import asdict
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan,emit_source_attention_frontier
from merlin.runtime.numeric_provider_identity import validate_numeric_provider_identity
from mlir_oot.no_fsm_audit import audit_elf

w=Path(__file__).resolve().parent
old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group')
core=Path('/scratch/agustin/tmp/merlin-one-endpoint-word-20261006')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((old/'candidate/build.json').read_text())
assert original['elf_sha256']=='71193a5840916615d53c49a22bc8b22452d70515da51be75ee9c29f072dda7bc'
assert sha(old/'candidate/model.elf')==original['elf_sha256']
source=(old/'numeric_frozen/provider.c').read_text()
assert source==(old/'native_numeric_frozen/provider.c').read_text()
manifest=json.loads((old/'native_numeric_frozen/manifest.json').read_text())
a=manifest['source_contract'];q=manifest['quant_contract']
d={k:int(v)for k,v in re.findall(r'#define (HEADS|ROWS|DEPTH|CHUNK|SEGMENT|LANES) (\d+)',source)}
epsilon=struct.unpack('f',struct.pack('I',q['epsilon_bf16_word']<<16))[0]
plan=SourceAttentionFrontierPlan(d['HEADS'],d['ROWS'],d['DEPTH'],d['CHUNK'],d['SEGMENT'],d['LANES'],a['score_scale'],a['poly_cutoff'],a['poly_scale'],tuple(a['poly_coefficients']),a['poly_bit_multiplier'],a['poly_bit_bias'],q['scale_divisor_f32'],epsilon,*q['clamp'])
options=dict(prepare_endpoint_rows=True,word_interval_enclosure=True,prepare_zero_error_blocks=False,prepare_product_domain=True,prepare_required_norms=True,retain_certified_rows=True,separable_source_radius=True,prepare_softmax_domain=True,integer_reconstruction=True)
def historical_include_order(text):
 # This historical provider predates one include-order-only template change.
 assert text.count('#include "bf16_quant_frontier.h"\n')==1
 text=text.replace('#include "bf16_quant_frontier.h"\n','')
 return text.replace('#include "separable_fma_radius.h"\n','#include "separable_fma_radius.h"\n#include "bf16_quant_frontier.h"\n')
assert historical_include_order(emit_source_attention_frontier(plan,symbol='group_provider',**options))==source
selected=historical_include_order(emit_source_attention_frontier(plan,symbol='group_provider',**options,one_endpoint_word_enclosure=True))
target_recipe=json.loads((old/'numeric_frozen/compile.json').read_text())
commands=[]
for name in ['numeric_frozen','native_numeric_frozen']:
 dest=w/name;dest.mkdir(exist_ok=True)
 for path in (old/name).glob('*.h'):shutil.copyfile(path,dest/path.name)
 shutil.copyfile(core/'merlin/runtime/c/one_endpoint_word_polynomial.h',dest/'one_endpoint_word_polynomial.h')
 (dest/'provider.c').write_text(selected)
 if name=='numeric_frozen':
  for command in target_recipe['commands']:
   command=[arg.replace(str(old/name),str(dest))for arg in command]
   subprocess.run(command,check=True);commands.append(command)
 else:
  command=[arg.replace(str(old/name),str(dest))for arg in manifest['compile']]
  subprocess.run(command,check=True)
  shutil.copyfile(old/name/'workspace_frontier_adapter.py',dest/'workspace_frontier_adapter.py')
  depfile=Path(command[command.index('-MF')+1]);library=Path(command[command.index('-o')+1])
  text=depfile.read_text().replace('\\\n',' ');deps=shlex.split(text.partition(':')[2])
  pins={str(Path(p).resolve()):sha(p)for p in deps}
  current={'compile':command,'compile_commands':[command],'compile_cwd':str(Path.cwd()),
   'compiler_sha256':sha(command[0]),'dependency_file_sha256':sha(depfile),
   'transitive_compile_dependencies':pins,
   'local_pins':{p.name:sha(p)for p in dest.iterdir()if p.suffix in('.h','.c','.so','.py')},
   'source_contract':a,'quant_contract':q,
   'scope':'Fresh current generic one-endpoint policy; no inherited authoritative compile recipe'}
  (dest/'manifest.json').write_text(json.dumps(current,indent=2)+'\n')
  lib=C.CDLL(str(library));lib.group_provider_workspace_bytes.restype=C.c_size_t;lib.group_provider_workspace_alignment.restype=C.c_size_t
  size,alignment=lib.group_provider_workspace_bytes(),lib.group_provider_workspace_alignment()
  assert size==123012160
  dependency_pins=dict(pins)
  dependency_pins.update({str((dest/p).resolve()):h for p,h in current['local_pins'].items()})
  dependency_pins.update({str(Path(command[0]).resolve()):sha(command[0]),str(depfile.resolve()):sha(depfile),str(library.resolve()):sha(library)})
  witness={'workspace_bytes':size,'workspace_alignment':64,'native_shared_sha256':sha(library),
   'complete_compile_manifest_sha256':sha(dest/'manifest.json'),'numeric_dependency_pins':dependency_pins}
  validation=validate_numeric_provider_identity(witness,manifest_path=dest/'manifest.json',native_library_path=library,queried_workspace_bytes=size,queried_workspace_alignment=alignment)
  (dest/'identity_witness.json').write_text(json.dumps(witness,indent=2)+'\n')
  (dest/'identity_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
link_pins={str(Path(arg).resolve()):sha(arg)for arg in original['link']if Path(arg).is_file()}
for arm in ['control','candidate']:
 dest=w/arm;dest.mkdir(exist_ok=True)
 link=[str(dest/'model.elf')if arg==str(old/'candidate/model.elf')else
  str(w/'numeric_frozen/provider.o')if arm=='candidate'and arg==str(old/'numeric_frozen/provider.o')else arg for arg in original['link']]
 subprocess.run(link,check=True)
 if arm=='control':assert sha(dest/'model.elf')==original['elf_sha256']
 audit=audit_elf((dest/'model.elf').read_bytes());assert audit['status']=='pass'
 (dest/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 (dest/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(dest/'model.elf'),'changed_only_provider_object':True},indent=2)+'\n')
 if arm=='candidate':shutil.copyfile(old/'candidate/watch.py',dest/'watch.py')
native_script=(old/'validate_native.py').read_text()
assert str(old/'native') in native_script and str(old/'native_numeric_frozen')in native_script
native_script=native_script.replace(str(old/'native'),str(w/'native')).replace(str(old/'native_numeric_frozen'),str(w/'native_numeric_frozen'))
native_script=native_script.replace('explicit word/private-soft numeric composition','explicit generic one-endpoint enclosure composition')
(w/'native').mkdir(exist_ok=True);(w/'validate_native.py').write_text(native_script)
metadata={'source_plan':asdict(plan),'source_options':dict(options,one_endpoint_word_enclosure=True),
 'source_binding':'Current typed source grammar and immutable original input/consumer captures; no workload compiler selector',
 'default_source_include_order_normalized_equal':True,'control_elf_sha256':original['elf_sha256'],
 'candidate_elf_sha256':sha(w/'candidate/model.elf'),'target_compile':commands,
 'source_header_sha256':sha(core/'merlin/runtime/c/one_endpoint_word_polynomial.h'),
 'original_link_input_pins':link_pins,'native_identity_validation':validation,
 'scope':'Complete allocation-aware original 12-head group; same driver/device/readback/source consumer. New bound can change replay; no stock promotion.'}
(w/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'elf':metadata['candidate_elf_sha256'],'numeric_identity':validation}),flush=True)
