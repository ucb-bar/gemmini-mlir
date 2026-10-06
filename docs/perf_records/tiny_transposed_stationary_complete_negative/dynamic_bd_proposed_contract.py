from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
root=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
w=Path(__file__).resolve().parent
files=[root/x for x in ['mlir_oot/ir/gemmini_dialect.py','mlir_oot/golden_device_lower.py','mlir_oot/golden_gemm.py','mlir_oot/codegen/builder.py','mlir_oot/execute_wave_estimator.py','mlir_oot/golden_compiler_export.py','mlir_oot/tables/isa.py']]
files += [core/'src/merlin/llvmlower/static_llvm_cfg.py',core/'src/merlin/llvmlower/llvm_loop_metadata.py',w/'command_census.json',w/'qualification.json',Path(__file__)]
r={
 'schema':'planned_dynamic_stationary_operand_edit_contract_v1',
 'status':'unimplemented_planned','recorded_utc':datetime.now(timezone.utc).isoformat(),
 'ownership':{'Merlin':'Reusable exact LLVM CFG traversal and ordinary-loop metadata. No target instruction interpretation; no new generic implementation currently needed.','OOT':'Typed dynamic stationary scratchpad row, exact ISA packing, resource/bank/disjointness proof, counted cached-K schedule and target command decoding.'},
 'missing_abstraction':'gemmini.preload accepts staticBD or dynamicCrow only. CachedB/wideA statically expandsK. Actual128K schedule contains1024staticPRELOAD plus1024staticCOMPUTE sites including repeated body/final drain.',
 'policy':'Explicit constructor-only default-off counted cachedK. Eligibility from semantic/types/layout/resources; no model/source ordinal/provenance/golden selector. Existing static/dynamicC byteidentity.',
 'primitive':'One explicit dynamicBD i64row, fixed dimensions/staticC destination, declared exact row bounds/reservedscratchpadspan; no accumulator flags or dynamicC ambiguity. Pack row into primitive ISA rs1 only.',
 'proof_obligations':[
  'module verification before mutation',
  'complete actual CFG trace, every dynamic declaration reachable/contained; partial samples refuse',
  'all dynamicBD and pairedA rows resolve exactly within reserved scratchpad, extents/banks/disjoint reservations checked',
  'destination accumulator bank/extents and firstinitialize/lateraccumulate source order preserved',
  'unknown control/memory/SSA arithmetic, negative/wrapped addresses, mutations refuse transactionally',
  'complete primitive command trace identical to static baseline including prefetch split/Ktail'],
 'schedule':'Static firstK initialization; retained ordinaryK loop interiors split at existing prefetch point; static incompleteK tail. Same DMA, output layout, aliases/lifetimes.',
 'phase1_edit_surface':['explicit schedule','typed induction-derived row expression','scratchpad/accumulator reservations and range witness','ordinary-loop retention','source/schedule/module fingerprints','independent completed command trace and finalnoFSM'],
 'phase2_costs':{'known':['dynamic command counts','requested byte/row intervals','static instruction sites/object footprint','actual complete paired capsule cycles when measured'],'unknown':['physical DMA/DRAM timing','cmd.valid/hazard events','Icache effects absent actual attribution','whole gain before independent hardware gate']},
 'validation_plan':['legacy/default bytes','unrelatedM/N/K tails,reuseB,cachedB,wideA/nonwideA,pipeline','unknown/unreachable/wrongtype/negative/wrap/bank/range mutations before lowering','realxDSLroundtrip/actualISA packing','complete staticcommand equivalence','compiled fullsignedi8/cancellation/tail/dirtyguard/noFSM','current largest complete capsule including runtime copies; no synthetic wholeprojection'],
 'dependencies':'Parent instructs implementation after current stationary pair and full2x4 gate close.2x4 nowPASS, pairlive. No queue/promotion.',
 'token_usage_available':False,
 'pins':{str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in files}}
(w/'dynamic_bd_proposed_contract.json').write_text(json.dumps(r,indent=2)+'\n')
print('PLANNED_DYNAMIC_BD_CONTRACT',len(files))
