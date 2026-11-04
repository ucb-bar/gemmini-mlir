"""Opt-in stock memory-value CSE after exact host/device helper inlining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def optimize(native: Path, *, install: Path, source: Path):
    from merlin.compile.tuning_knobs import knob
    if knob('HOST_MEMORY_CSE', default='off', values=('off', 'on'),
            root=Path(__file__).resolve().parents[1]) != 'on':
        return None
    native, install, source = Path(native), Path(install), Path(source)
    before = native.with_name('before-memory-cse.ll')
    selected = native.with_name('memory-cse.ll')
    before.write_bytes(native.read_bytes())
    command = [install / 'bin' / 'opt', '-S', '-verify-each',
        '--passes=function(early-cse<memssa>,gvn,instcombine,simplifycfg,lower-switch)',
        before, '-o', selected]
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise ValueError('stock host memory CSE failed: ' + result.stderr[-6000:])
    def identity(path):
        return dict(path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = dict(schema='submission.host_memory_cse.v1', selection='on',
        source=identity(source), before=identity(before), selected=identity(selected),
        tool=identity(command[0]), command=list(map(str, command)),
        arithmetic='ordinary LLVM legality; no fastmath, reassociation or new source premises',
        effects='existing alias and MemorySSA analysis; opaque device asm memory clobbers retained',
        hardware_cycles='not_measured')
    native.write_bytes(selected.read_bytes())
    (native.parent / 'host-memory-cse.json').write_text(json.dumps(report, indent=2))
    return report
