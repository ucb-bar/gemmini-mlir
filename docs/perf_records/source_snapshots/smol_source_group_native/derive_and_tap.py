from pathlib import Path
import json, hashlib, math
from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, UnitAttr
from xdsl.ir import Region
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups
from merlin.llvmlower.ordered_fma_group_outline import outline_ordered_fma_group
from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions
from merlin.xdsl_dialects._common import text
w=Path(__file__).resolve().parent
source=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle')
bundle=w/'bundle';bundle.mkdir(exist_ok=True)
module=parse_mlir_file(source/'model.mlir');print('PARSED',flush=True)
groups=analyze_ordered_fma_groups(module);print('GROUPS',len(groups),flush=True)
group=groups[0]
# Experimental ordinal selects the fixture only; shipped policy has no such selector.
outlined=outline_ordered_fma_group(module,group,'source_group_exact')
input_records=[];c=['#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n']
values=[*outlined.inputs,outlined.call.results[0]]
for index,value in enumerate(values):
 symbol='source_group_tap_'+str(index);typ=value.type;shape=tuple(typ.get_shape());rank=len(shape);dtype=str(typ.element_type)
 if dtype not in ('bf16','f32','i1'):raise ValueError('unsupported probe dtype '+dtype)
 ctype={'bf16':'uint16_t','f32':'uint32_t','i1':'uint8_t'}[dtype]
 declaration=func.FuncOp(symbol,([typ],[]),Region(),visibility='private',arg_attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr('read')})]));declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration)
 tap=func.CallOp(symbol,[value],[])
 outlined.call.parent.insert_op_after(tap,outlined.call) if index==len(values)-1 else outlined.call.parent.insert_op_before(tap,outlined.call)
 filename=w/('input_'+str(index)+'.bin' if index<len(values)-1 else 'endpoint.bin')
 c.append(f'typedef struct {{void *allocated,*aligned; int64_t offset,size[{rank}],stride[{rank}];}} desc_{index};\n')
 c.append(f'void _mlir_ciface_{symbol}(desc_{index} *d) {{\n')
 c.append(' if('+ ' || '.join(f'd->size[{axis}]!={extent} || d->stride[{axis}]<0'for axis,extent in enumerate(shape))+') abort();\n')
 c.append(f' FILE *f=fopen({json.dumps(str(filename))},"wb");if(!f)abort();\n')
 c.append(f' for(int64_t flat=0;flat<{math.prod(shape)};flat++){{int64_t rest=flat,offset=d->offset;\n')
 for axis,extent in reversed(list(enumerate(shape))):c.append(f' offset+=(rest%{extent})*d->stride[{axis}];rest/={extent};\n')
 c.append(f' {ctype} value=((const {ctype}*)d->aligned)[offset];if(fwrite(&value,sizeof(value),1,f)!=1)abort();}}\n if(fclose(f))abort();}}\n')
 input_records.append(dict(index=index,role='input'if index<len(values)-1 else'endpoint',shape=shape,dtype=dtype,path=str(filename),source_value_type=str(typ)))
module.verify()
# Preserve the exact extracted function for the independent endpoint oracle.
standalone=outlined.function.clone();standalone.properties['sym_visibility']=StringAttr('public');standalone.attributes['llvm.emit_c_interface']=UnitAttr()
from xdsl.dialects.builtin import ModuleOp
(w/'source_group_exact.mlir').write_text(text(ModuleOp([standalone])))
counts=rewrite_ordered_fma_contractions(module,output_tile=8,pre_widen_operands='both',assume_rne=True,assume_finite_intermediates=True)
for p in source.iterdir():
 if p.name!='model.mlir' and not(bundle/p.name).exists():(bundle/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
(bundle/'model.mlir').write_text(text(module));(w/'tap.c').write_text(''.join(c))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(schema='source_exact_group_tap_derivation_v1',original_source_path=str(source/'model.mlir'),original_source_sha256=sha(source/'model.mlir'),derived_source_sha256=sha(bundle/'model.mlir'),reference_source_sha256=sha(w/'source_group_exact.mlir'),group_count=len(groups),selected_group_ordinal=0,source_operations=len(group.operations),source_contractions=len(group.contractions),input_count=len(outlined.inputs),internalized_constant_tensors=len(outlined.internalized_constants),omitted_unread_initializers=len(outlined.omitted_unread_initializers),records=input_records,ordered_rewrite=counts,tap_source_sha256=sha(w/'tap.c'),scope='Unchanged actual source group plus explicit native read-only input/endpoint taps. Original full1600 gate must still close before fixtures are accepted.',token_usage_available=False)
(w/'source_binding.json').write_text(json.dumps(r,indent=2)+'\n');print('DERIVED',len(outlined.inputs),counts,flush=True)
