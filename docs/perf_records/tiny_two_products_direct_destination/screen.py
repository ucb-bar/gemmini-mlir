"""Only alter the upstream pass ordering; retain the frozen typed source."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np
from merlin.llvmlower.abi import HostModel

W = Path(__file__).resolve().parent
T = W.parent / 'tiny-two-multiplications-capsule-20261006'
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
recipe = json.loads((T / 'packet_lower/lowering_recipe.json').read_text())
argv = list(recipe['commands'][0]['argv'])
assert argv[2] == str(T / 'packet_lower/model.mlir')
assert argv[3] == str(T / 'packet_lower/model.ll')
passes = argv[4].split(',')
hits = [i for i, p in enumerate(passes) if p.startswith('one-shot-bufferize')]
assert len(hits) == 1
i = hits[0]
assert passes[i + 1].startswith('buffer-results-to-out-params')
passes.insert(i + 1, 'canonicalize')
argv[3] = str(W / 'module.ll')
argv[4] = ','.join(passes)
with (W / 'lower.log').open('w') as log:
    subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
raw = (W / 'module.ll').read_text().replace('forward', 'exposed').replace('dealloc_helper', 'exposed_dealloc_helper')
(W / 'exposed.ll').write_text(raw)
so = W / 'exposed.so'
subprocess.run([str(L / 'clang'), '-O3', '-shared', '-fPIC', '-ffp-contract=off', str(W / 'exposed.ll'), '-lm', '-o', str(so)], check=True)
inputs = [np.load(T / 'capture' / (name + '.npy')) for name in ['residual', 'accumulation', 'scale']]
expected = np.load(T / 'capture/expected.npy')
out = np.empty_like(expected)
HostModel.load(str(so), name='exposed')([(v.ctypes.data, v.shape) for v in [*inputs, out]])
assert np.array_equal(out.view('u4'), expected.view('u4'))
np.save(W / 'output.npy', out)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = {
    'schema': 'upstream_loop_carried_destination_identity_screen_v1',
    'source_sha256': sha(T / 'capsule_abi.mlir'),
    'frozen_runner_sha256': sha(argv[1]),
    'actual_argv': argv,
    'actual_added_pass': 'canonicalize after one-shot-bufferize before buffer-results-to-out-params',
    'malloc_requests_bytes': [int(v) for v in re.findall(r'call ptr @malloc\(i64 (\d+)\)', raw)],
    'llvm_memcpy_calls': raw.count('call void @llvm.memcpy.'),
    'original_native_words_exact': 16384,
    'llvm_sha256': sha(W / 'exposed.ll'), 'so_sha256': sha(so),
    'cost_measured': False, 'token_usage_available': False,
}
(W / 'screen.json').write_text(json.dumps(report, indent=2) + '\n')
print('DESTINATION_IDENTITY_SCREEN', report, flush=True)
