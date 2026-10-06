from pathlib import Path
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import FEATURE
from merlin.llvmlower.late_quant_rne import rewrite
W=Path(__file__).resolve().parent;R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');case=W/'cost_m2';case.mkdir(exist_ok=True)
source=(R/'source_capsule.mlir').read_text().replace('tensor<1x8x5632','tensor<1x2x5632')
source=source.replace(') -> tensor<1x2x5632xi8> {',') -> tensor<1x2x5632xi8> attributes {llvm.emit_c_interface} {')
(case/'source.mlir').write_text(source)
ll=lower_to_llvm_ir(source,workdir=case/'lower',features={FEATURE,'lower_fma_to_intrinsic'}).replace('forward','baseline').replace('dealloc_helper','baseline_dealloc_helper')
for mode,isa in [('target','rv64gc'),('native','portable')]:
 selected,receipt=rewrite(ll,host_isa=isa,combine_clamp=(mode=='target'));assert len(receipt['routes'])==2
 (case/f'partial_base.{mode}.ll').write_text(selected)
 # Actual same scalar source SSA names are recorded explicitly by upstream.
 print(mode,receipt['routes'],flush=True)
# Copy experiment runner with solely typed domain and data extent changes.
s=(W/'build_capsule.py').read_text().replace("assert sha(R/'capsule/baseline.target.ll')==q['cases'][0]['target_sha256']", "assert (W/'partial_base.target.ll').is_file()").replace("assert sha(R/'capsule/baseline.native.ll')==q['cases'][0]['native_sha256']", "assert (W/'partial_base.native.ll').is_file()")
s=s.replace("expected=np.load(R/'capture/expected.npy')", "args[0]=args[0][:,:2,:].copy();args[2]=args[2][:,:2,:].copy();expected=np.load(R/'capture/expected.npy')[:,:2,:].copy()")
s=s.replace("(W/'observer.c').read_text()", "(W.parent/'observer.c').read_text()")
s=s.replace("(R/f'capsule/baseline.{mode}.ll').read_text()", "(W/f'partial_base.{mode}.ll').read_text()")
s=s.replace("main=(R/'capsule/main.c').read_text()", "main=(R/'capsule/main.c').read_text().replace('45056','11264').replace('45184','11392').replace('{1,8,5632}','{1,2,5632}')")
s=s.replace("sha(W/'rational_native_screen.json')", "sha(W.parent/'rational_native_screen.json')")
s=s.replace("raise SystemExit(0)\n",'').replace("actual_original_tap_words=45056", "actual_original_tap_words=11264")
s=s.replace("source_sha256=sha(R/'source_capsule.mlir')", "source_sha256=sha(W/'source.mlir'),full_original_source_sha256=sha(R/'source_capsule.mlir')")
s=s.replace("original45056source body", "original first2rows/all5632channels pure scalar source body")
(case/'build.py').write_text(s)
