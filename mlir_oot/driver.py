"""``gemmini-opt`` -- the package's single CLI, exposing the four certification entrypoints.

    gemmini-opt --verify-diagnostics                       IN.mlir
    gemmini-opt --convert-iface-to-gemmini                 IN.mlir
    gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=OUT.json IN.mlir
    gemmini-opt --convert-iface-to-gemmini --emit-target-artifact         IN.mlir

The last two flags may be given together (``emit_analysis_bundle``): the pipeline runs ONCE and
writes the command buffer to the named file while the target artifact goes to stdout.
"""

from __future__ import annotations

import copy
import sys
import traceback
from io import StringIO
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from . import cmdbuf, validate
from .codegen import scalar_lane
from .codegen.llvm_emit import CodegenError, LLVMEmitter, to_text
from .target.isa import EncodingError
from .frontend import iface_dialect
from .frontend.extract import InterfaceError, extract
from .ir.workload import Workload
from .codegen.host_lane import HostLaneError
from .lowering.lanes import MixedLaneProgram
from .codegen.scalar_lane import ScalarLaneError
from .lowering.contraction import LoweringRefusal
from .lowering.epilogue import UnsupportedEpilogue
from .lowering import lanes, quantize, rowscale, softmaxfuse, sumsq, widen
from .lowering.iface_to_gemmini import IfaceToGemmini


class Options:
    def __init__(self) -> None:
        self.verify_only = False
        self.convert = False
        self.emit_cb: str | None = None
        self.emit_artifact = False
        self.input: str | None = None


def parse_argv(argv: list[str]) -> Options:
    o = Options()
    for a in argv:
        if a == "--verify-diagnostics":
            o.verify_only = True
        elif a.startswith("--convert-iface-to-"):
            o.convert = True
        elif a.startswith("--emit-command-buffer="):
            o.emit_cb = a.split("=", 1)[1]
        elif a == "--emit-command-buffer":
            o.emit_cb = "command_buffer.json"
        elif a in ("--emit-target-artifact", "--lower-target-to-llvm"):
            o.emit_artifact = True
        elif a.startswith("-"):
            raise SystemExit(f"gemmini-opt: unknown flag {a!r}")
        else:
            o.input = a
    return o


def _context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(iface_dialect.MerlinIface)
    return ctx


def _print(module: ModuleOp) -> str:
    s = StringIO()
    Printer(stream=s).print_op(module)
    return s.getvalue() + "\n"


def read_workload(text: str) -> Workload:
    """Parse the input, whichever of the two frozen grammars it is written in."""
    from .frontend import linalg_reader

    if linalg_reader.is_linalg_on_tensors(text):
        return linalg_reader.read(text)
    module = Parser(_context(), text).parse_module()
    module.verify()
    return extract(module)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = parse_argv(argv)
    if opts.input is None:
        print("gemmini-opt: no input file", file=sys.stderr)
        return 2
    text = Path(opts.input).read_text()

    try:
        wl = read_workload(text)
    except Exception as exc:  # parse / verify diagnostics
        print(f"gemmini-opt: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if opts.verify_only and not (opts.convert or opts.emit_cb or opts.emit_artifact):
        return 0

    declined, lowered, host_module = _lower(wl)

    # The dialect lowering can succeed on a program this target's INSTRUCTION ENCODING still cannot
    # carry (a derived field wider than the rs1 slot that holds it). That failure used to surface
    # only while writing the artifact -- after the command buffer had already been written declaring
    # commands -- so the two outputs disagreed and the entrypoint died with a traceback. Build the
    # artifact FIRST, so an encoding refusal becomes one stated decline that both outputs carry.
    artifact_text: str | None = None
    if (opts.emit_artifact or opts.emit_cb is not None) and declined is None:
        try:
            artifact_text = _artifact_text(wl, lowered, host_module)
        except (EncodingError, CodegenError, ValueError, KeyError) as exc:
            declined, lowered, host_module, artifact_text = _decline(wl, exc), None, None, None
            wl.quant = None

    if declined is not None:
        wl.lane_placement = []
    if opts.emit_cb is not None:
        cb = cmdbuf.build(
            wl,
            kernel_abi=(lowered.kernel_abi if lowered is not None else None),
            declined=declined,
        )
        probs = validate.problems(cb)
        if probs:
            print("gemmini-opt: command buffer is not well formed:", file=sys.stderr)
            for p in probs:
                print(f"  {p}", file=sys.stderr)
            return 1
        cmdbuf.write(cb, opts.emit_cb)

    if opts.emit_artifact:
        sys.stdout.write(
            artifact_text if artifact_text is not None else to_text(_empty_kernel(wl))
        )
    elif opts.convert and opts.emit_cb is None:
        if lowered is None:
            # A decline is stated in the command buffer, not by failing the entrypoint: the tool
            # ran, it just has no target command for this program.
            print(
                f"gemmini-opt: declined: {declined.get('reason') if declined else 'unknown'}",
                file=sys.stderr,
            )
            sys.stdout.write(_print(_empty_target_module()))
            return 0
        sys.stdout.write(_print(lowered.module))
    return 0


def _lower(wl: Workload):
    """Pick this program's admission route and produce what that route owes.

    Route A -- the interface->target rewrite, an accelerator command stream.
    Route Q -- a contraction the mesh has no operand port for, asked for in the ACCELERATOR ABI:
               the compiler derives a per-tensor scale at run time, stages the operand as i8 and
               issues the contraction on the mesh (`lowering.quantize`).
    Route H -- a region whose operands the mesh has no encoding for and which route Q cannot carry:
               the placement is declared AND compiled onto the scalar lane, because a declared
               placement that computes nothing still drops the output tensor the program commits.
    Route D -- a construct no route can express, stated as a decline with its reason.
    """
    if wl.declined is not None:
        return wl.declined, None, None

    try:
        refusal = lanes.mesh_refusal(wl) if wl.ops else None
    except MixedLaneProgram as exc:
        # Route D: the program's regions belong on different lanes and this package cannot yet
        # sequence them in one kernel. Stated as a decline, never as a lane route that would leave
        # the array idle for a region the capsule asks it to carry.
        wl.lane_placement = []
        return _decline(wl, exc), None, None
    if refusal is not None:
        plan = quantize.plan(wl)
        if plan is not None:
            try:
                # The rewrite runs on a COPY. The command buffer must keep describing the program
                # the interface declared -- float operands, float readout -- because that is the
                # program whose arithmetic is being certified; the staging and the integer readout
                # are how this backend ISSUES it, and they are stated in `params`, not by restating
                # the workload as something it is not.
                staged_wl = copy.deepcopy(wl)
                quantize.apply(staged_wl, plan)
                lowered = IfaceToGemmini().run(staged_wl)
                wl.quant = plan
                wl.lane_placement = lanes.placement(wl, lanes.MESH_LANE, plan.reason())
                return None, lowered, None
            except (LoweringRefusal, UnsupportedEpilogue, InterfaceError, ValueError,
                    KeyError, CodegenError):
                wl.quant = None  # fall through to the host lane, which is what it had before
    if refusal is not None:
        hoist = rowscale.plan(wl)
        fuse = softmaxfuse.plan(wl)
        plans = widen.plan(wl)
        if hoist is not None or fuse is not None or plans is not None:
            try:
                # The rewrite runs on a COPY for the same reason route Q's does: the command buffer
                # must go on describing the program the interface declared. The digit split is how
                # this backend ISSUES that program, and it is stated in `params`, not by restating
                # the workload as something it is not.
                split_wl = copy.deepcopy(wl)
                reasons = []
                fuse_on_copy = softmaxfuse.plan(split_wl)
                if fuse_on_copy is not None:
                    softmaxfuse.apply(split_wl, fuse_on_copy)
                    reasons.append(fuse_on_copy.reason())
                hoist_on_copy = rowscale.plan(split_wl)
                if hoist_on_copy is not None:
                    rowscale.apply(split_wl, hoist_on_copy)
                    reasons.append(hoist_on_copy.reason())
                split_plans = widen.plan(split_wl)
                if split_plans:
                    widen.apply(split_wl, split_plans)
                    reasons.append(widen.reason(split_plans))
                if not reasons:
                    raise LoweringRefusal("neither reassociation applies to this program")
                lowered = IfaceToGemmini().run(split_wl)
                wl.lane_placement = lanes.placement(wl, lanes.MESH_LANE, "; ".join(reasons))
                return None, lowered, None
            except (LoweringRefusal, UnsupportedEpilogue, InterfaceError, ScalarLaneError,
                    ValueError, KeyError, CodegenError):
                pass    # fall through to the host lane, which is what it had before

    if refusal is not None:
        # Route S -- a normalisation whose SCALE this readout cannot apply, but whose row REDUCTION
        # is a contraction of two declared i8 operands. The reduction is issued on the array and
        # only the scale runs off it, inside the same kernel, rather than sending a matrix
        # accelerator's only multiply-accumulate work to the host (`lowering.sumsq`).
        staged = sumsq.plan(wl)
        if staged is not None:
            try:
                # The rewrite runs on a COPY, for the reason routes Q and the reassociations do:
                # the command buffer must go on describing the program the interface declared.
                split_wl = copy.deepcopy(wl)
                plan_on_copy = sumsq.plan(split_wl)
                if plan_on_copy is None:
                    raise LoweringRefusal("the staged reduction does not replan on the copy")
                sumsq.apply(split_wl, plan_on_copy)
                lowered = IfaceToGemmini().run(split_wl)
                wl.lane_placement = lanes.placement(
                    wl, lanes.MESH_LANE, plan_on_copy.reason()
                )
                return None, lowered, None
            except (LoweringRefusal, UnsupportedEpilogue, InterfaceError, ScalarLaneError,
                    ValueError, KeyError, CodegenError):
                pass    # fall through to the host lane, which is what it had before

    if refusal is not None:
        try:
            wl.lane_placement = lanes.host_placement(wl, refusal)
            lowered = IfaceToGemmini()._host_lane(wl)
            module = scalar_lane.compile_workload(wl, lowered.arg_tensors)
            wl.ops = []
            return None, lowered, module
        except (ScalarLaneError, LoweringRefusal, UnsupportedEpilogue, InterfaceError,
                ValueError, KeyError) as exc:
            wl.lane_placement = []
            return _decline(wl, exc), None, None

    try:
        lowered = IfaceToGemmini().run(wl)
        host_module = None
        if wl.lane_placement and wl.host_entry is not None:
            # Route H has to PRODUCE what it commits. Compile the placement here, so a region this
            # lane cannot express becomes a stated decline rather than a kernel that returns
            # without writing its output.
            host_module = _host_lane_module(wl, lowered)
        return None, lowered, host_module
    except (LoweringRefusal, UnsupportedEpilogue, InterfaceError, HostLaneError,
            ScalarLaneError, ValueError, KeyError) as exc:
        return _decline(wl, exc), None, None
    except Exception as exc:  # unexpected: still a decline, never a silent empty program
        traceback.print_exc(file=sys.stderr)
        return _decline(wl, exc), None, None


def _artifact_text(wl: Workload, lowered, host_module) -> str:
    """Render the LLVM-dialect artifact for whichever route :func:`_lower` admitted."""
    if host_module is not None:
        return to_text(host_module)
    if lowered is None:
        return to_text(_empty_kernel(wl))
    lane_wl = lowered.workload or wl
    resident_matmul = (any(op.kind == "matmul" for op in wl.ops)
                       and all(op.kind in ("resident_pack", "matmul", "commit", "evict")
                               for op in wl.ops))
    return to_text(LLVMEmitter().emit(
        lowered.module, quant=wl.quant,
        lane_program=(lane_wl, lowered.lane_stages) if lowered.lane_stages else None,
        scratch=_scratch_bytes(lane_wl, lowered.scratch),
        unroll=1 if resident_matmul else 4))


def _scratch_bytes(wl: Workload, names: list[str]) -> dict[str, int]:
    """How many bytes the kernel must allocate for each of its own internal buffers."""
    from .lowering.iface_to_gemmini import dtype_bytes
    from .target import layout

    out: dict[str, int] = {}
    for n in names:
        t = wl.tensors[n]
        out[n] = layout.row_count(t.shape) * layout.row_pitch(t.shape) * dtype_bytes(t.dtype)
    return out


def wl_kernel_abi(wl: Workload):
    return None


def _host_lane_module(wl: Workload, lowered) -> ModuleOp:
    """Compile the host-lane region; a construct it cannot express raises and becomes a decline."""
    from .codegen.host_lane import HostLaneCompiler

    return HostLaneCompiler(wl.host_entry, list(lowered.arg_tensors), wl.tensors).compile()


def _decline(wl: Workload, exc: Exception) -> dict:
    op = next((o.kind for o in wl.ops if o.kind not in ("resident_pack", "evict")), None)
    shape: list[int] = []
    for t in wl.tensors.values():
        if t.role == "output":
            shape = list(t.shape)
            break
    d: dict = {"reason": f"{type(exc).__name__}: {exc}"}
    if op:
        d["op"] = op
    if shape:
        d["shape"] = shape
    return d


def _empty_target_module() -> ModuleOp:
    """A declined program still emits a well-formed, empty target module."""
    from xdsl.ir import Block, Region

    from .target import dialect as gd
    from xdsl.dialects.builtin import StringAttr

    blk = Block(arg_types=[])
    blk.add_op(gd.ReturnOp.build(operands=[[]], result_types=[[]], attributes={}))
    kernel = gd.KernelOp.build(
        regions=[Region([blk])],
        attributes={"sym_name": StringAttr("gemmini_kernel"),
                    "command_shape": StringAttr("declined")},
    )
    return ModuleOp([kernel])


def _empty_kernel(wl: Workload) -> ModuleOp:
    """A declared decline still defines the symbol the harness links against, and emits no command."""
    from xdsl.dialects import llvm
    from xdsl.ir import Block, Region

    ptr = llvm.LLVMPointerType()
    n = len([t for t in wl.tensors.values() if t.role != "intermediate"])
    blk = Block(arg_types=[ptr] * max(1, n))
    blk.add_op(llvm.ReturnOp())
    fn = llvm.FuncOp(
        "gemmini_kernel",
        llvm.LLVMFunctionType([ptr] * max(1, n)),
        linkage=llvm.LinkageAttr("external"),
        body=Region([blk]),
    )
    return ModuleOp([fn])
