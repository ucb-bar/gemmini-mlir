"""Run existing exact target matchers on immutable prepared post-offload source."""
from dataclasses import asdict
from pathlib import Path
import collections
import hashlib
import json
from mlir_oot.frontend.parse import parse_module
from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.golden_requant import match as match_requant

W = Path(__file__).resolve().parent
source = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/model.prepared.mlir')
module = parse_module(source.read_text())
contractions, epilogues = [], []
for ordinal, op in enumerate(module.walk()):
    if op.name not in {'linalg.generic','linalg.matmul','linalg.batch_matmul'}:
        continue
    dims = match_integer_gemm(op)
    if dims is not None:
        contractions.append(dict(ordinal=ordinal, dimensions=asdict(dims), types=[str(v.type) for v in op.operands],
                                 result_types=[str(v.type) for v in op.results],
                                 consumers=[dict(name=u.operation.name, operand=u.index, result_types=[str(v.type)for v in u.operation.results]) for u in op.results[0].uses]))
    if op.name == 'linalg.generic' and op.operands and 'i32>' in str(op.operands[0].type):
        rec = dict(ordinal=ordinal, operand_types=[str(v.type)for v in op.operands], result_types=[str(v.type)for v in op.results])
        try:
            candidate = match_requant(op)
            rec.update(status='matched', proof=candidate.proof)
        except ValueError as error:
            rec.update(status='refused', reason=str(error))
        epilogues.append(rec)
out = dict(schema='existing_target_capability_source_match_census_v1', source_path=str(source),
           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
           integer_contractions=contractions, direct_i32_consumer_match_results=epilogues,
           direct_requant_matches=sum(x['status']=='matched' for x in epilogues),
           refusal_counts=dict(collections.Counter(x.get('reason') for x in epilogues if x['status']=='refused')),
           scope='Prepared source already replaced integer contractions with calls: zero standalone integer matches here does not indicate lost offload. Existing target requant matcher applied to155 actual i32 consumers, without mutation or new lowering. Source ordinals identify bindings only, not strategy. No cycle estimate; float residuals/nonlinear and f32 contraction acceleration require additional contracts.',
           token_usage_available=False)
(W/'existing_capabilities.json').write_text(json.dumps(out,indent=2)+'\n')
print('EXISTING_CAPABILITY_CENSUS',len(contractions),len(epilogues),out['direct_requant_matches'],out['refusal_counts'],flush=True)
