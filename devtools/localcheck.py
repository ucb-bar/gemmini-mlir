"""Local, simulator-free sweep: run the four entrypoints over every capsule on disk.

Development aid, not part of any declared command: it only invokes the package's own CLI.
"""
import json, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TOOL = str(HERE.parent / "gemmini-opt")
roots = [a for a in sys.argv[1:] if not a.startswith("-")] or ["isa", "layers", "model_slices", "model"]
out = pathlib.Path("/tmp/localcheck"); out.mkdir(exist_ok=True)
rows = []
for r in roots:
    for p in sorted((ROOT / r).glob("*/capsule.interface.mlir")):
        name = p.parent.name
        t0 = time.time()
        cbp, art = out / f"{name}.json", out / f"{name}.mlir"
        pr = subprocess.run([TOOL, "--convert-iface-to-gemmini", f"--emit-command-buffer={cbp}",
                             "--emit-target-artifact", str(p)], capture_output=True, text=True)
        dt = time.time() - t0
        status, note = "ok", ""
        if pr.returncode != 0:
            status = "ERROR"
            note = (pr.stderr or "").strip().splitlines()[-1][:160] if pr.stderr else ""
        else:
            art.write_text(pr.stdout)
            cb = json.loads(cbp.read_text())
            if "declined" in cb:
                status, note = "declined", cb["declined"]["reason"][:160]
        rows.append((r, name, status, pr.stdout.count("llvm.inline_asm"), round(dt, 2), note))
w = max(len(x[1]) for x in rows)
for r, name, status, n, dt, note in rows:
    if status != "ok" or "-v" in sys.argv:
        print(f"{r:13s} {name:{w}s} {status:9s} {n:7d} {dt:6.2f}s {note}")
import collections
print(collections.Counter(x[2] for x in rows))
print("biggest:", sorted(rows, key=lambda x: -x[3])[:5])
