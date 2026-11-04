"""Authored source-to-ordinary-emitter binding for dense panel epoch trials."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def _workspace(package):
    return next(root for root in Path(package).resolve().parents
                if (root / "agent_selfcheck.py").is_file())


def source(case, *, package):
    workspace = _workspace(package)
    public = workspace / "public_sources" / (case + ".mlir")
    if public.is_file():
        return public
    for family in ("layers", "projections", "calibration", "model_slices"):
        path = workspace / family / case / "capsule.interface.mlir"
        if path.is_file():
            return path
    raise ValueError(f"no public interface source for {case}")


def prepare(source, *, package, mechanism):
    package = Path(package).resolve()
    sys.path[:0] = [str(package / "compiler_library"), str(package)]
    from frontend import parse_source
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.transforms.pipeline_state import PipelineSession, PipelineContext, PassSpec
    from merlin.transforms.pipeline_policy import PolicyFacts, RegisteredPass, PipelinePolicy, PipelineVariant
    from dense import plan_dense
    from merlin.compile.scheduling.library_sources import shared_library_sources

    if mechanism not in (None, "dense_panel_epochs", "shared_rhs_epochs", "dense_panel_transfers", "dense_panel_transfers_dma", "dense_retained_transfers_dma", "dense_ready_readout_dma"):
        raise ValueError(f"unregistered dense mechanism {mechanism}")
    text = Path(source).read_text()
    parse_source(text)
    cb = parse_interface_mlir(text)
    contractions = []
    residents = {}
    for command in cb["commands"]:
        operands = command.get("operands", {})
        code = command["opcode"]
        if code == "RES_PACK":
            residents[operands["dst"]] = operands["src"]
        elif code in ("MATMUL", "MATMUL_RESIDENT"):
            lhs = cb["tensors"][operands["lhs"]]["shape"]
            rhs = cb["tensors"][residents.get(operands["rhs"], operands["rhs"])]["shape"]
            if len(lhs) != 2 or len(rhs) != 2 or lhs[1] != rhs[0]:
                raise ValueError("dense schedule trial requires rank-two logical matrices")
            contractions.append((lhs[0], lhs[1], rhs[1]))
    if not contractions:
        raise ValueError("source has no supported dense contraction")
    digest = lambda data: hashlib.sha256(data).hexdigest()
    context = PipelineContext(hardware=digest((package / "data/facts.json").read_bytes()),
        numerical_policy="ordinary-compiler-source-epilogue", resources=digest((package / "data/command_ordering.json").read_bytes()),
        execution="ordinary-manifest-backend", library=digest(json.dumps(
            shared_library_sources(version=20),sort_keys=True).encode()))
    def verify(value):
        parse_source(value)
    session = PipelineSession(text, context=context, ir_level="interface_source",
        fingerprint=lambda value: digest(value.encode()), verify_ir=verify)
    facts = PolicyFacts(session.stamp(), shapes=tuple((m,k,n) for m,k,n in contractions),
                        operations=frozenset({"dense_contraction"}))
    incumbent = json.loads((package / "DENSE_SCHEDULE.json").read_text())["schedule"]

    def select(alternative):
        def action(current):
            # These are the checked shared schedule constructors used by the
            # ordinary backend; legality is derived from the selected RTL laws.
            plans = [plan_dense(m,k,n,alternative=alternative)[1] for m,k,n in contractions]
            for plan in plans:
                plan.revalidate()
            return {"selected_dense_schedule": alternative,
                    "checked_dense_plans": plans}
        return action

    def emit(current):
        from backend import compile_source
        alternative = current.analysis("selected_dense_schedule")
        previous = os.environ.get("DENSE_SCHEDULE")
        os.environ["DENSE_SCHEDULE"] = alternative
        try:
            _, llvm = compile_source(current.module)
        finally:
            if previous is None:
                os.environ.pop("DENSE_SCHEDULE", None)
            else:
                os.environ["DENSE_SCHEDULE"] = previous
        return {"target_product": llvm if llvm.endswith("\n") else llvm + "\n"}

    registry = {}
    for name, alternative in (("dense_incumbent", incumbent), ("dense_panel_epochs", "epochs"),
                              ("shared_rhs_epochs", "shared_rhs_epochs"),
                              ("dense_panel_transfers", "epochs_wide"),
                              ("dense_panel_transfers_dma", "epochs_wide_dma"),
                              ("dense_retained_transfers_dma", "retained_wide_dma"),
                              ("dense_ready_readout_dma", "epochs_ready_dma")):
        registry[name] = RegisteredPass(PassSpec(name,0,
            produces=("selected_dense_schedule", "checked_dense_plans"),
            mutates_ir=False,ir_level="interface_source"), select(alternative))
    registry["ordinary_target_emitter"] = RegisteredPass(PassSpec("ordinary_target_emitter",1,
        requires=("selected_dense_schedule", "checked_dense_plans"), produces=("target_product",),
        mutates_ir=False,ir_level="interface_source"), emit)
    branch = mechanism if mechanism else "dense_incumbent"
    return dict(session=session, facts=facts, registry=registry,
        policy=PipelinePolicy((PipelineVariant(branch,(branch,"ordinary_target_emitter")),), branch),
        output="target_product")
