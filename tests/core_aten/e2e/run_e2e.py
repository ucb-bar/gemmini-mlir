"""End-to-end: model2MLIR quantized graph -> gemmini-mlir Core ATen catalog -> Spike.

For every captured case directory (``capture.py``) this parses ``model.mlir``, binds and
compiles every matched region through ``mlir_oot.core_aten.catalog``, wires the kernels by
the manifest's operand sources (function arguments, earlier kernel results), links one
bare-metal ELF and runs it.  The function result is checked against the independent source-graph
exact-rational reference and the instruction model, with equality of every output code.
Eager float output is logged separately as informational divergence. Unsupported
regions are reported as device refusals.

    python run_e2e.py CAPTURE_DIR [--only name ...] [--no-spike] [--batch NAME] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "device"))
sys.path.insert(0, str(HERE.parent))
from exact_oracle import evaluate, host_guard_values

import runner  # noqa: E402
from mlir_oot.core_aten.catalog import build, compile_catalog, _execution_provider  # noqa: E402
from mlir_oot.core_aten.stream import simulate  # noqa: E402
from mlir_oot.frontend.parse import parse_module  # noqa: E402


def _key(src: dict) -> tuple:
    return ("arg", src["func_arg"]) if "func_arg" in src else ("op", src["op_ordinal"], src["result"])


def _return_sources(text: str) -> list[tuple]:
    module = parse_module(text)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    fn = [op for op in module.walk() if op.name == "func.func"][0]
    ret = [op for op in fn.walk() if op.name == "func.return"][0]
    args = list(fn.body.blocks[0].args)
    out = []
    for v in ret.operands:
        out.append(("arg", args.index(v)) if v in args else ("op", ordinals[v.owner], list(v.owner.results).index(v)))
    return out


class DeviceRefusal(Exception):
    pass


def prepare(case_dir: Path, prefix: str, wd: Path):
    from fixture_source import read
    text = read(case_dir / "model.mlir")
    api = _execution_provider()
    package = HERE.parents[2]
    policy = api._selected_numeric_policy(case_dir)
    proof = api.admit_numeric_schedule(text, package=package, numeric_policy=policy, require_full_coverage=False)
    if proof is None or proof.get('status') != 'proven':
        raise DeviceRefusal(f"{case_dir.name}: admission refused: {proof}")
    selected_readout = api._readout_options(policy)
    _, manifest, matches = build(text, readout_options=selected_readout)
    if not manifest["bindings"]:
        raise DeviceRefusal(f"{case_dir.name}: no exact region matched; declines={manifest['declines']}")
    # rename every kernel symbol per case so one ELF can hold many cases
    inputs = dict(np.load(case_dir / "inputs.npz"))
    if (case_dir / "expected.npy").exists():
        expected_list = [np.load(case_dir / "expected.npy")]
    else:
        expected_list = [np.load(p) for p in sorted(case_dir.glob("expected[0-9]*.npy"))]
    kern = {k["symbol"]: k for k in manifest["kernels"]}
    values: dict[tuple, str] = {}
    model_values: dict[tuple, np.ndarray] = {}
    buffers, calls, outputs = {}, [], set()
    for i, arr in sorted(inputs.items()):
        key = ("arg", int(i[3:]))
        values[key] = f"{prefix}_{i}"
        buffers[values[key]] = arr
        model_values[key] = arr
    for i, (key, array) in enumerate(host_guard_values(text, inputs).items()):
        values[key] = f"{prefix}_guard{i}"
        model_values[key] = array
        buffers[values[key]] = array
    for bi, (b, m) in enumerate(zip(manifest["bindings"], matches)):
        k = kern[b["symbol"]]
        names, sim_inputs = [], {}
        for a in k["args"]:
            if a["role"] == "input":
                src = _key(b["operands"][a["name"]])
                if src not in values:
                    raise DeviceRefusal(f"{case_dir.name}: host work remains; declines={manifest['declines']}")
                names.append(values[src])
                sim_inputs[a["name"]] = model_values[src]
            else:
                bn = f"{prefix}_b{bi}_{a['name']}"
                dt = np.int32 if a["dtype"] == "i32" else np.int8
                buffers[bn] = np.zeros(a["shape"], dt)
                names.append(bn)
                if a["role"] == "output":
                    outputs.add(bn)
                    (oname, osrc), = b["result"].items()
                    values[_key(osrc)] = bn
        model = simulate(m.kernel.stream, sim_inputs)
        exact = evaluate(m, sim_inputs)
        for output_name, expected in exact.items():
            if not np.array_equal(model[output_name], expected):
                raise AssertionError(f"{case_dir.name}: device model differs from exact arithmetic")
        (oname, osrc), = b["result"].items()
        model_values[_key(osrc)] = model[oname]
        calls.append((b["symbol"], names))
    checks, worst_max, worst_exact = [], 0, 1.0
    for k, (ret, expected) in enumerate(zip(_return_sources(text), expected_list)):
        if ret not in values:
            raise DeviceRefusal(f"{case_dir.name}: returned value remains on host; declines={manifest['declines']}")
        out_buf = values[ret]
        model_out = model_values[ret]
        e = np.abs(model_out.astype(np.int64).reshape(-1) - expected.astype(np.int64).reshape(-1))
        worst_max, worst_exact = max(worst_max, int(e.max())), min(worst_exact, float((e == 0).mean()))
        tag = case_dir.name if len(expected_list) == 1 else f"{case_dir.name}.out{k}"
        checks += [runner.Check(out_buf, model_out.reshape(-1).astype(buffers[out_buf].dtype), 0, f"{tag}_vs_exact_arithmetic")]
    summary = dict(ops=[b["op"] for b in manifest["bindings"]], kernels=len(kern),
                   instructions=sum(k["instructions"] for k in kern.values()),
                   fences=sum(k["fences"] for k in kern.values()),
                   model_vs_eager_max=worst_max, model_vs_eager_exact=worst_exact,
                   eager_float_divergence=dict(informational=True, max_code_difference=worst_max, exact_fraction=worst_exact),
                   declines=manifest["declines"])
    work = wd / prefix
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / 'model.mlir'
    prepared.write_text(text)
    compiled = compile_catalog(prepared, work, llvm_bin=api.toolchain.llvm_install() / 'bin',
                               numeric_schedule_proof=proof, readout_options=selected_readout)
    if compiled['status'] != 'compiled':
        raise DeviceRefusal(f"{case_dir.name}: compilation refused: {compiled.get('reason')}")
    obj = work / 'core_aten.o'
    return Path(obj), buffers, calls, checks, outputs, summary


def run(case_dirs, batch: str, spike=True) -> dict:
    wd = runner.work_root() / batch
    objects, buffers, calls, checks, outputs, summary = [], {}, [], [], set(), {}
    for i, d in enumerate(case_dirs):
        t0 = time.time()
        prefix = f"e{i}_{d.name}"
        try:
            o, b, c, ch, out, sm = prepare(d, prefix, wd)
        except DeviceRefusal as exc:
            summary[d.name] = dict(status="REFUSED", reason=str(exc))
            print("device refusal:", d.name, str(exc), flush=True)
            continue
        sm["build_s"] = round(time.time() - t0, 1)
        # one object per case: make its symbols unique by prefixing at the object level
        objects.append(runner.rename_symbols(o, wd / f"{prefix}_r.o", {s: f"{prefix}__{s}" for s, _ in c}))
        buffers.update(b)
        calls += [(f"{prefix}__{s}", a) for s, a in c]
        checks += ch
        outputs |= out
        summary[d.name] = sm
    if not calls:
        return dict(cases=summary, device_status="REFUSED")
    elf, r = runner.replay(runner.DeviceTest(batch, objects, buffers, calls, checks, outputs), wd, spike)
    result = dict(elf=str(elf), cases=summary)
    if r is not None:
        result["spike"] = dict(status=r.status, passed=r.passed, failed=r.failed, info=r.info, perf=r.perf,
                              seconds=round(r.seconds, 1), exit=r.exit_code, tail=None if r.ok else r.log[-3000:])
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-spike", action="store_true")
    ap.add_argument("--batch", default="e2e")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    dirs = sorted(p for p in a.capture.iterdir() if (p / "model.mlir").exists())
    if a.only:
        dirs = [d for d in dirs if d.name in a.only]
    res = run(dirs, a.batch, spike=not a.no_spike)
    text = json.dumps(res, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    spike = res.get("spike")
    if spike:
        print("spike", spike["status"], "passed", len(spike["passed"]), "failed", spike["failed"], spike["seconds"], "s")
        for line in spike["info"]:
            print("  ", line)
        if spike["tail"]:
            print(spike["tail"][-2000:])
    return 0 if spike is None or spike["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
