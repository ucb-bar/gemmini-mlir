"""Original normal full48 native execution with unchanged model/bridge objects."""
from pathlib import Path
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants';H=W/'normal_native';H.mkdir(exist_ok=False)
original=B/'out/exact_row_normal/drivers/validate_native.py';source=original.read_text();start=source.index('lib=C.CDLL(')
prefix='''import ctypes as C,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args
'''
prefix+=f'w=Path({str(B/"out/exact_row_normal")!r});h=Path({str(H)!r});frozen=Path({str(W/"candidate/native_numeric")!r})\n'
prefix+='root=Path("/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005");sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
prefix+='runtime=root/"host_outlined/model.so"; prior=w/"native_v1";commands=[]\n'
prefix+='for name in ("model.o","bridge.o","products.o","math.o"): (h/name).symlink_to(prior/name)\n'
prefix+='cmd=["/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang","-shared",*[str(h/x)for x in ("model.o","bridge.o","products.o","math.o")],str(frozen/"provider.so"),str(runtime),"-Wl,--wrap=expf","-Wl,--wrap=_mlir_ciface_source_attention_frontier_fallback","-lm","-o",str(h/"model.so")];subprocess.run(cmd,check=True);commands.append(cmd)\n'
driver=H/'run.py';driver.write_text(prefix+source[start:]);exec(compile(driver.read_text(),str(driver),'exec'))
