from pathlib import Path
import hashlib,json,subprocess
from merlin.llvmlower.scalar_contraction import RUNNER_PRELUDE
J=Path(__file__).resolve().parent
source=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_host_abi/model.mlir')
runner='''import sys,json
from torch_mlir import ir
'''+RUNNER_PRELUDE.split('_SC_MARKER =',1)[0]+'''
ctx=ir.Context()
module=ir.Module.parse(open(sys.argv[1]).read(),ctx)
records=[]
def walk(op):
 for region in op.regions:
  for block in region.blocks:
   for inner in block.operations:
    spec=_scalar_contraction_spec(inner)
    if spec is not None:
     maps,extents,mul,add=spec
     row,col=len(extents)-3,len(extents)-2
     legal=(row>=0 and extents[row]%2==0 and extents[col]%4==0 and row in maps[0] and col not in maps[0] and col in maps[1] and row not in maps[1])
     records.append(dict(positions=maps,extents=extents,mul_order=mul,add_order=add,rectangular_map_extent_eligibility=legal,types=list(map(str,(v.type for v in inner.operands)))))
    walk(inner.operation)
walk(module.operation)
print(json.dumps(records))
'''
(J/'source_census_runner.py').write_text(runner)
p=subprocess.run(['/scratch/agustin/projects/model2MLIR/.venv/bin/python',str(J/'source_census_runner.py'),str(source)],capture_output=True,text=True,check=True)
(J/'source_census_corrected.log').write_text(p.stdout+p.stderr)
records=json.loads(p.stdout);assert len(records)==45 and all(r['rectangular_map_extent_eligibility'] for r in records)
r={'schema':'original_typed_scalar_rectangular_source_census_v1','source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'actual_source_operations':45,'recurring_QK_PV_operations':44,'additional_eligible_outer_product_operations':1,'selected_by_source_types_maps_scalar_body_only':True,'source_ordinal_or_symbol_policy':False,'records':records,'observation_only':'Original typed bodies inspected without mutation; actual normal compiler and whole native/strict qualification close emission. Empty/default pipelines unchanged; family labels not policy.'}
(J/'source_census.json').write_text(json.dumps(r,indent=2)+'\n');print('ORIGINAL_TYPED_RECTANGULAR_SOURCE_CENSUS_PASS',len(records),flush=True)
