from pathlib import Path
from collections import Counter
import json,hashlib
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups,validate_group_source
W=Path(__file__).resolve().parent;p=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle/model.mlir')
module=parse_mlir_text(p.read_text());module.verify();print('PARSED_SOURCE',flush=True)
groups=analyze_ordered_fma_groups(module);print('GROUPS',len(groups),flush=True);families=Counter();records=[]
for ordinal,g in enumerate(groups):
 validate_group_source(g)
 rec={'group_ordinal':ordinal,'contractions':len(g.contractions),'source_operations':len(g.operations),'closed_bf16_endpoints':g.closed_bf16_endpoints,'bf16_output_types':[str(v.type)for v in g.bf16_outputs],'live_f32_outputs':[str(v.type)for v in g.live_f32_outputs],'unsupported_uses':[(str(v.type),u.name,i)for v,u,i in g.unsupported_uses],'operation_classes':dict(Counter(op.name for op in g.operations)),'input_types':[str(v.type)for v in g.inputs]}
 records.append(rec);families[(rec['contractions'],rec['source_operations'],rec['closed_bf16_endpoints'],tuple(rec['bf16_output_types']),tuple(rec['live_f32_outputs']))]+=1
r={'schema':'ordered_fma_source_group_closure_census_v1','source_path':str(p),'source_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'groups':len(groups),'contractions':sum(x['contractions']for x in records),'families':[{'contractions':a,'source_operations':b,'closed_bf16_endpoints':c,'bf16_outputs':d,'live_f32_outputs':e,'groups':n}for(a,b,c,d,e),n in families.most_common()],'records':records,'scope':'Source graph closure and mutation witness only. Numerical certificate, fenv and physical ownership proof and emitter binding remain separate obligations.','token_usage_available':False}
(W/'group_census.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['families'],indent=2),flush=True)
