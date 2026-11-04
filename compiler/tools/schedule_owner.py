"""Ordinary manifest emission and broker custody for dense schedule exploration.

No cycle surrogate is provided. Pending requests remain unranked; only complete,
numerically passing ordinary broker feedback can become an observation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("SCHEDULE_WORKSPACE", next(
    (root for root in PACKAGE.parents if (root / "agent_selfcheck.py").is_file()), PACKAGE.parent))).resolve()
sys.path[:0] = [str(PACKAGE / "compiler_library"), str(PACKAGE)]
_LOCK = threading.Lock()
_DOMAINS = {}
_PROGRAMS = {}


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path.read_bytes())}


def library_pin(package=PACKAGE):
    from merlin.compile.scheduling.library_sources import shared_library_sources
    return sha(shared_library_sources(version=20))


def locate(case, workspace=WORKSPACE):
    suite = workspace / "public_sources/suite.json"
    if suite.is_file():
        entry = next((row for row in json.loads(suite.read_text())["cases"] if row["name"] == case), None)
        if entry is not None:
            source = suite.parent / entry["source"]
            from merlin.targetgen.contract.interface_emit import parse_interface_mlir
            cb = parse_interface_mlir(source.read_text())
            entry = dict(entry, inputs=[dict(name=name, **tensor) for name,tensor in cb["tensors"].items()])
            return suite, source, entry
    for family in ("layers", "projections", "calibration", "model_slices"):
        directory = workspace / family / case
        declaration = directory / "capsule.yaml"
        if declaration.is_file():
            from merlin.common.yaml import load_yaml
            capsule = load_yaml(declaration)
            return declaration, directory / capsule.get("interface_mlir", "capsule.interface.mlir"), capsule
    raise ValueError(f"no public capsule for {case}")


def capsule_shape(capsule):
    if capsule.get("operation", {}).get("op") != "matmul":
        return None
    attrs = capsule["operation"].get("attributes", {})
    operands = {v["name"]: tuple(v["shape"]) for v in capsule["inputs"]}
    lhs = operands.get(attrs.get("lhs"))
    rhs = operands.get(attrs.get("weight"))
    if lhs and rhs and len(lhs) == len(rhs) == 2 and lhs[1] == rhs[0]:
        return lhs[0], lhs[1], rhs[1]
    return None


def select_capsule(shape):
    from merlin.common.yaml import load_yaml
    selected = os.environ.get("SCHEDULE_CASES", "").split(",")
    selected = [name for name in selected if name]
    if not selected:
        selected = sorted(path.parent.name for path in (WORKSPACE / "layers").glob("*/capsule.yaml"))
        selected += [row["name"] for row in json.loads((WORKSPACE / "public_sources/suite.json").read_text())["cases"]]
    for case in selected:
        declaration, source, capsule = locate(case)
        if capsule_shape(capsule) == tuple(shape):
            return case, declaration, source, capsule
    raise ValueError(f"no original public matmul capsule has shape {tuple(shape)}; set SCHEDULE_CASES")


def manifest_emit(package, source, schedule=None, panel_shape=None):
    """Invoke the exact ordinary stdout command used by the grader."""
    from merlin.common.yaml import load_yaml
    package, source = Path(package).resolve(), Path(source).resolve()
    manifest = load_yaml(package / "manifest.yaml")
    tool = package / manifest["entrypoints"]["tool"]
    argv = [word.format(tool=str(tool), input_mlir=str(source))
            for word in manifest["commands"]["lower_target_to_llvm"]["argv"]]
    env = dict(os.environ, PYTHONPATH=str(package / "compiler_library"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("DENSE_SCHEDULE", None)
    env.pop("DENSE_PANEL_SHAPE", None)
    env.pop("DENSE_RESIDENT_READOUT", None)
    if schedule is not None:
        env["DENSE_SCHEDULE"] = schedule
    if panel_shape is not None:
        env["DENSE_PANEL_SHAPE"] = panel_shape
    process = subprocess.run(argv, cwd=package, env=env, capture_output=True, timeout=120)
    if process.returncode or not process.stdout:
        raise ValueError(f"ordinary manifest emission failed: {process.stderr.decode()[-2000:]}")
    return process.stdout, {"argv": argv, "source": pin(source), "manifest": pin(package / "manifest.yaml"),
        "tool": pin(tool), "llvm_sha256": sha(process.stdout), "schedule": schedule, "panel_shape": panel_shape,
        "numerics": "NOT_CHECKED", "cycles": "UNKNOWN"}


def binding(shape, caps):
    from dense import plan_dense
    from merlin.transforms.pipeline_state import PipelineContext
    case, declaration, source_path, capsule = select_capsule(shape)
    source_bytes = source_path.read_bytes()
    attrs = capsule["operation"].get("attributes", {})
    bias = bool(attrs.get("bias") or attrs.get("bias_tensor") or "bias_add" in attrs.get("epilogue", []))
    _, original = plan_dense(*shape, bias=bias, alternative="compact_mn")
    modes = {id(original): "compact_mn"}
    mapping = {"original": "compact_mn", "identical": "compact_mn", "batched": "retained", "ready": "epochs"}

    def transform(result, mode):
        if source_path.read_bytes() != source_bytes:
            raise ValueError("original source changed during exploration")
        alternative = mapping[mode]
        if mode in ("original", "identical"):
            return original
        _, chosen = plan_dense(*shape, bias=bias, alternative=alternative)
        modes[id(chosen)] = alternative
        return chosen

    def emit(result):
        result.revalidate()
        alternative = modes[id(result)]
        payload, evidence = manifest_emit(PACKAGE, source_path, alternative)
        evidence.update(case=case, actual_checked_kernel_sha256=result.kernel.digest(),
                        requested_schedule=alternative, recipe_aliases=mapping)
        _PROGRAMS[sha(payload)] = {"schedule": alternative, "case": case, "source": str(source_path),
                                  "evidence": evidence, "shape": tuple(shape),
                                  "suite": "scalability" if source_path.parent.name == "public_sources" else "core"}
        return payload.decode(), evidence

    context = PipelineContext(hardware=sha((PACKAGE / "data/facts.json").read_bytes()),
        resources=sha(caps), numerical_policy="i8-i8-modular-i32:source-epilogue",
        execution="ordinary-capsule-Spike-complete-call", library=library_pin())
    return dict(source=original, transform=transform, emit=emit, shapes=((shape[0], shape[1]), (shape[1], shape[2])),
        context=context, case=case, declaration=str(declaration), original_source=str(source_path),
        capsule=capsule, source_sha256=sha(source_bytes))


def identity(shape, bound):
    from merlin.transforms.evaluated_exploration import ExplorationIdentity, exploration_context_identity
    capsule = bound["capsule"]
    # Input/reference pins identify the exact immutable public generation recipe;
    # realized executable/inputs and model scope remain with the ordinary broker.
    domain = ExplorationIdentity(bound["source_sha256"], sha((PACKAGE / "data/mlir_oot_backend_contract.yaml").read_bytes()),
        sha({"inputs": capsule["inputs"], "stimulus": capsule.get("stimulus_range"),
             "capsule": pin(bound["declaration"])}),
        sha({"numeric_policy": capsule.get("numeric_policy"), "capsule": pin(bound["declaration"])}),
        sha((PACKAGE / "data/facts.json").read_bytes()), library_pin(),
        sha({"ordinary_owner": pin(WORKSPACE / "agent_selfcheck.py"), "adapter": pin(__file__), "suite":
             "scalability" if Path(bound["original_source"]).parent.name == "public_sources" else "core", "sim": "spike"}),
        sha({"case": bound["case"], "shape": tuple(shape), "scope": "whole ordinary capsule invocation"}),
        sha({"roi": "complete measured events for ordinary capsule call", "provider": "Spike timing model"}),
        exploration_context_identity(bound["context"]), "llvm-mlir", "modeled_roi")
    _DOMAINS[domain.digest()] = bound
    return domain


def _snapshot(package, destination, schedule, panel_shape=None):
    def ignore(directory, names):
        return [name for name in names if name in {"compiler_library", "__pycache__", "build", "out", "qa", "evidence", "selfcheck_out"}]
    shutil.copytree(package, destination, ignore=ignore)
    # The ordinary broker refuses source symlinks. Materialize the selected
    # reviewed library into this immutable submission before asking it to run.
    shutil.copytree((package / "compiler_library").resolve(), destination / "compiler_library",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    settings = json.loads((destination / "DENSE_SCHEDULE.json").read_text())
    settings["schedule"] = schedule
    if panel_shape is not None:
        settings["dense_panel_shape"] = panel_shape
    (destination / "DENSE_SCHEDULE.json").write_text(json.dumps(settings) + "\n")


def evaluate(program, receipt, domain):
    """Submit one immutable ordinary snapshot; return None while feedback is pending."""
    from merlin.transforms.evaluated_exploration import CompletedObservation
    key = sha(program.data)
    meta = _PROGRAMS[key]
    bound = _DOMAINS[domain.digest()]
    if meta["case"] != bound["case"] or pin(meta["source"])["sha256"] != domain.source_sha256:
        raise ValueError("emitted program/source custody differs from evaluation identity")
    root = WORKSPACE / "ab" / "schedule-owner" / domain.digest()
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / (key + ".json")
    with _LOCK:
        if not record_path.exists():
            variant = WORKSPACE / "variants" / ("schedule-" + sha([domain.digest(), key])[:24])
            variant.parent.mkdir(exist_ok=True)
            if variant.exists():
                raise ValueError("existing untracked schedule snapshot; retain it for diagnosis")
            _snapshot(PACKAGE, variant, meta["schedule"], meta.get("panel_shape"))
            actual, evidence = manifest_emit(variant, meta["source"])
            if actual != program.data:
                raise ValueError("snapshot ordinary emission differs from exact candidate LLVM")
            command = [sys.executable, str(WORKSPACE / "agent_selfcheck.py"), "--submission",
                       str(variant.relative_to(WORKSPACE)), "--suite", meta["suite"], "--sim", "spike",
                       "--capsules", meta["case"], "--wait-budget", "5"]
            done = subprocess.run(command, cwd=WORKSPACE, capture_output=True, timeout=45)
            response = json.loads(done.stdout)
            request_id = response.get("request_id") or response.get("selfcheck_request_id")
            if not request_id:
                raise ValueError(f"ordinary broker did not retain request: {response}")
            request = WORKSPACE / ".qa_channel" / f"req_{request_id}.json"
            record = {"schema": "authored.schedule_owner.v1", "program_sha256": key,
                "evaluation_identity": domain.digest(), "request_id": request_id, "request": pin(request),
                "request_body": json.loads(request.read_text()), "variant": str(variant),
                "emission": evidence, "source": pin(meta["source"]), "case": meta["case"],
                "status": "pending", "numerics": "UNKNOWN", "timing": "UNKNOWN"}
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        record = json.loads(record_path.read_text())
    feedback = WORKSPACE / ".qa_channel" / f"resp_{record['request_id']}.json"
    if not feedback.exists():
        return None
    response = json.loads(feedback.read_text())
    if response.get("error"):
        record.update(status="rejected_by_ordinary_broker", feedback=pin(feedback), error=response["error"])
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return None
    if response.get("requested_submission_sha256") != record["request_body"]["submission_sha256"]:
        raise ValueError("feedback describes different compiler snapshot")
    rows = response.get("per_capsule", response.get("cases", []))
    if isinstance(rows, dict):
        row = rows.get(meta["case"])
    else:
        row = next((r for r in rows if r.get("capsule") == meta["case"]), None)
    if row is None:
        return None
    detail = row.get("detail", {})
    if detail.get("path"):
        path = WORKSPACE / detail["path"]
        if not path.is_file():
            return None
        payload = path.read_bytes()
        if detail.get("sha256") and sha(payload) != detail["sha256"]:
            raise ValueError("feedback detail custody changed")
        row = json.loads(payload)
    performance = row.get("performance", {})
    cycles = performance.get("estimated_whole_soc_cycles")
    numeric = "pass" if row.get("numerical_execution_passed") is True else "fail" if row.get("numerical_execution_passed") is False else "unknown"
    # Scalability feedback exposes ordinary execution/comparison as tier L2;
    # core feedback supplies the explicit numerical boolean instead.
    if numeric == "unknown" and row.get("tiers", {}).get("L2") in ("pass", "fail"):
        numeric = row["tiers"]["L2"]
    complete = (performance.get("complete_event_accounting") is True
                and performance.get("fast_estimate") is not True
                and not performance.get("unresolved_events")
                and isinstance(cycles, int) and cycles > 0)
    record.update(status="completed" if complete else "pending_timing", feedback=pin(feedback),
                  numerical_status=numeric, complete_roi=complete, cycles=cycles if complete else None,
                  actual_provider_scope={k: performance.get(k) for k in ("comparison_domain_sha256", "model_sha256", "scope_sha256", "capsule_sha256", "facts_sha256", "elf_sha256")})
    if complete:
        shared_scope = {k: performance.get(k) for k in
            ("comparison_domain_sha256", "model_sha256", "scope_sha256", "capsule_sha256", "facts_sha256")}
        scope_path = root / "provider-domain.json"
        with _LOCK:
            if scope_path.exists() and json.loads(scope_path.read_text()) != shared_scope:
                raise ValueError("ordinary feedback provider/input/scope differs across paired candidates")
            if not scope_path.exists():
                scope_path.write_text(json.dumps(shared_scope, indent=2) + "\n")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    if numeric == "unknown" or not complete:
        return None
    return CompletedObservation(domain, key, sha(feedback.read_bytes()), numeric, "complete", True, cycles)
