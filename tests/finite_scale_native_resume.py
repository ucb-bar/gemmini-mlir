"""Correct immediate fenv sampling; reuse already frozen compiled whole objects.

The prior attempt sampled flags after NumPy integer guard comparisons. Those
host ufuncs clear flags, so it is retained as a harness failure, not hidden or
reported as a source numerical failure. No compiler/object/table is rebuilt.
"""
import ast
from pathlib import Path
import finite_scale_whole_prepare as module

OLD=module.OLD
OUT=module.OUT
core=module.CORE
base=module.load('native_resume_base',OLD/'tests/source_continuation_whole_build.py')
module.sys.modules['source_continuation_whole_build']=base
qualifier=module.load('native_resume_qualifier',OLD/'tests/source_continuation_whole_qualify.py')
effects=module.IntervalEffectContract(True,True,True,True,True)
typed=base.N/'typed_prepacket.generic.mlir'
proofs,refused=module.find_closed_scalar_i8_observers(module.parse_mlir_text(typed.read_text()),effects=effects)
previous=module.json.loads((module.CONTROL/'selected/target/host_llvm/source_binding.json').read_text())
helpers=module.bind_finite_scale_helpers(module.SOURCE.read_text(),routes=previous['source_bindings']['routes'],observers=proofs,effects=effects,immutable_inputs=True,fresh_disjoint_output=True)
assert len(helpers)==22 and not refused
# Actual prior compile commands are independently retained through normal hook
# receipts; source object identities are pinned before and after this resume.
before={str(p):module.sha(p) for arm in ('control','selected') for p in (OUT/arm).rglob('*') if p.is_file()}
namespace=dict(vars(module));namespace.update(base=base,qualifier=qualifier,effects=effects,typed=typed,proofs=proofs,helpers=helpers,all_commands=[],guards={})
tree=ast.parse(Path(module.__file__).read_text())
body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='feature' for t in n.targets))
program=ast.Module(body=body[start:],type_ignores=[])
exec(compile(ast.fix_missing_locations(program),str(Path(__file__)),'exec'),namespace)
assert before=={str(p):module.sha(p) for p in map(Path,before)}
module.save(OUT/'native_resume.json',{'schema':'source_finite_broadcast_native_measurement_correction_v1','status':'pass','first_failed_script':str(OUT/'preparation_first_attempt.py'),'first_failed_script_sha256':module.sha(OUT/'preparation_first_attempt.py'),'correction':'Sample flags immediately after native call, before NumPy guard ufuncs; compiled inputs/outputs unchanged.','actual_compiled_objects_unchanged':before,'qualification_path':str(OUT/'qualification.json'),'qualification_sha256':module.sha(OUT/'qualification.json'),'resume_script':str(Path(__file__).resolve()),'resume_script_sha256':module.sha(__file__)})
